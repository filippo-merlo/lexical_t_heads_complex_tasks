"""Llama-3.1-8B-Instruct with its fitted ``J^z`` and ``W_O``: forward passes, readouts, patching.

The model wrapper, the lens, the ``o_proj`` hooks and both per-head readouts are
head_level_jlens's, used unchanged (as in read_inctx_param_heads/heads_reading.py). Every
prompt is run alone, without padding, so the reading and the scored answer come from the
same computation.

Notation (proposal Sec. 3): ``z`` is a layer's ``o_proj`` input at the last prompt token,
``[n_heads * head_dim]``; head ``h`` owns ``z[h * head_dim:(h + 1) * head_dim]``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from lth import LENS_KEY, READOUTS

TOP_K = 10


@dataclass
class LM:
    model: object
    spec: object
    jz: dict[int, torch.Tensor]
    w_o: dict[int, torch.Tensor]
    eos: frozenset[int]

    @classmethod
    def load(cls) -> LM:
        from transformers import GenerationConfig

        from src.config import LENSES, jz_path
        from src.heads import o_proj_weight
        from src.model import load_jz, load_model

        spec = LENSES[LENS_KEY]
        model = load_model(spec)
        lens = load_jz(spec, jz_path(spec, n_prompts=spec.jz_n_prompts))
        jz = {l: lens.jacobians[l].to(model.input_device) for l in range(spec.n_layers)}
        w_o = {l: o_proj_weight(model, l) for l in range(spec.n_layers)}
        eos = GenerationConfig.from_pretrained(spec.hf_model_id).eos_token_id
        return cls(model, spec, jz, w_o, frozenset(eos if isinstance(eos, list) else [eos]))

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def device(self):
        return self.model.input_device

    # ---- tokenization ------------------------------------------------------------------

    def encode_raw(self, text: str) -> list[int]:
        """Raw text with BOS, as Yang et al. tokenize their prompts."""
        return self.tokenizer(text).input_ids

    def encode_chat(self, text: str) -> list[int]:
        """One user turn under the chat template, ready for the reply (as in Fu et al.)."""
        return list(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
            return_dict=True)["input_ids"])

    def first_token(self, text: str) -> int:
        """First token of an answer string (Yang et al.: ``" " + output``)."""
        return self.tokenizer(text, add_special_tokens=False).input_ids[0]

    # ---- forward passes ----------------------------------------------------------------

    @torch.no_grad()
    def next_token(self, ids: list[int], *, edits: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy next token after ``ids`` and ``z`` at the last token, ``[n_layers, d_source]``.

        ``edits`` (``{layer: fn(z) -> z}``) patch ``o_proj``'s input during the pass; the
        returned ``z`` is recorded before the edit.
        """
        tokens, z = self._prefill(torch.tensor([ids], device=self.device), edits)
        return tokens[0], z[0]

    @torch.no_grad()
    def next_tokens(self, ids: list[int], rows: int, *, edits: dict) -> list[int]:
        """Greedy next token after ``ids`` copied over ``rows`` batch rows, under ``edits``."""
        tokens, _ = self._prefill(torch.tensor([ids] * rows, device=self.device), edits)
        return tokens.tolist()

    @torch.no_grad()
    def generate(self, ids: list[int], max_new_tokens: int, *, edits: dict | None = None,
                 rows: int = 1) -> tuple[list[str], torch.Tensor]:
        """Greedy replies to ``ids`` (copied over ``rows`` batch rows) and the prefill ``z``.

        ``edits`` apply to the prompt's forward pass only, as in Yang et al.'s patching;
        the reply is then decoded from the patched cache. Each reply stops at an EOS token
        and is decoded without special tokens and stripped, as in Fu et al.'s vLLM path.
        """
        input_ids = torch.tensor([ids] * rows, device=self.device)
        cache = {}
        first, z = self._prefill(input_ids, edits, cache)
        hf = self.model.hf_model
        generated = [first]
        done = torch.tensor([t.item() in self.eos for t in first], device=self.device)
        past = cache["past"]
        for _ in range(max_new_tokens - 1):
            if bool(done.all()):
                break
            out = hf(input_ids=generated[-1][:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1].argmax(-1)
            generated.append(token)
            done |= torch.tensor([t.item() in self.eos for t in token], device=self.device)
        tokens = torch.stack(generated, dim=1).tolist()
        replies = []
        for row in tokens:
            cut = next((i for i, t in enumerate(row) if t in self.eos), len(row))
            replies.append(self.tokenizer.decode(row[:cut], skip_special_tokens=True).strip())
        return replies, z

    @torch.no_grad()
    def generate_batch(self, prompts: list[list[int]], max_new_tokens: int, *,
                       edits: dict | None = None) -> list[list[int]]:
        """Greedy replies to several prompts at once, as token ids up to the first EOS.

        For reasoning mode, where replies run to thousands of tokens. Prompts are padded on
        the left with an attention mask, so the last prompt token is the last column.
        ``edits`` stay on for the whole generation: at the last prompt token and at every
        generated token (proposal Sec. 5.3, reasoning mode). Readings are not taken here
        but from an unpadded pass over the prompt and the reply (:meth:`states`).
        """
        from src.hooks import ZinPatcher

        pad = self.tokenizer.pad_token_id
        pad = min(self.eos) if pad is None else pad
        width = max(len(p) for p in prompts)
        input_ids = torch.full((len(prompts), width), pad, device=self.device)
        mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(prompts):
            input_ids[row, width - len(ids):] = torch.tensor(ids, device=self.device)
            mask[row, width - len(ids):] = 1
        patcher = ZinPatcher(self.model.layers, edits) if edits else contextlib.nullcontext()
        with patcher:
            out = self.model.hf_model.generate(
                input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=pad, eos_token_id=sorted(self.eos))
        replies = []
        for row in out[:, width:].tolist():
            cut = next((i for i, t in enumerate(row) if t in self.eos), len(row))
            replies.append(row[:cut])
        return replies

    def decode(self, ids: list[int]) -> str:
        """A reply's text, as Fu et al.'s vLLM path returns it: no special tokens, stripped."""
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def states(self, ids: list[int], positions: list[int]) -> list[torch.Tensor]:
        """``z`` at each of ``positions`` from one unpadded pass over ``ids``."""
        from src.hooks import AnchorRecorder

        layers = range(self.spec.n_layers)
        with AnchorRecorder(self.model.layers, zin_at=layers) as recorder:
            self.model.forward(torch.tensor([ids], device=self.device))
        return [torch.stack([recorder.activations[("zin", l)][0, p] for l in layers]) for p in positions]

    def _prefill(self, input_ids, edits, cache=None):
        from src.hooks import AnchorRecorder, ZinPatcher

        layers = range(self.spec.n_layers)
        with AnchorRecorder(self.model.layers, zin_at=layers) as recorder:
            patcher = ZinPatcher(self.model.layers, edits) if edits else None
            if patcher is not None:
                patcher.__enter__()
            try:
                out = self.model.hf_model(input_ids=input_ids, use_cache=cache is not None,
                                          logits_to_keep=1)
            finally:
                if patcher is not None:
                    patcher.__exit__(None, None, None)
        if cache is not None:
            cache["past"] = out.past_key_values
        z = torch.stack([recorder.activations[("zin", l)][:, -1] for l in layers], dim=1)
        return out.logits[:, -1].argmax(-1), z

    # ---- readouts ----------------------------------------------------------------------

    @torch.no_grad()
    def top_tokens(self, z: torch.Tensor, k: int = TOP_K) -> torch.Tensor:
        """Top-``k`` token ids of every head, ``[len(READOUTS), n_layers * n_heads, k]``.

        Logit-head reads ``W_O^h z_h`` and J-head ``J_{l,h} z_h``, both through the final
        norm and the unembedding (``model.unembed``); the norm scales each head's vector by
        a positive number, so the ranking is that of ``q_v^T`` in proposal Sec. 3.
        """
        from src.readout import all_heads_transported

        out = []
        for readout in READOUTS:
            vectors = torch.cat([
                all_heads_transported(self.jz[l], z[l], self.spec, arm=readout, w_o=self.w_o[l])
                for l in range(self.spec.n_layers)])
            out.append(self.model.unembed(vectors).topk(k, dim=-1).indices)
        return torch.stack(out).to(torch.int32).cpu()

    # ---- patching ----------------------------------------------------------------------

    def mean_patch(self, heads: list[tuple[int, int]], mean_z: torch.Tensor,
                   alphas: list[float]) -> dict:
        """Edits that set each head in ``heads`` to ``alpha * mean_z`` at the last token.

        Batch row ``i`` gets ``alphas[i]`` (Yang et al., ``set_activation_rank_logit_average``).
        """
        alpha = torch.tensor(alphas, device=self.device, dtype=torch.bfloat16)[:, None]
        by_layer: dict[int, list[int]] = {}
        for layer, head in heads:
            by_layer.setdefault(layer, []).append(head)
        edits = {}
        for layer, layer_heads in by_layer.items():
            cols = torch.cat([torch.arange(h * self.spec.head_dim, (h + 1) * self.spec.head_dim)
                              for h in layer_heads]).to(self.device)
            target = mean_z[layer].to(self.device, torch.bfloat16)[cols]

            def edit(z, cols=cols, target=target):
                z = z.clone()
                z[:, -1, cols] = alpha * target
                return z

            edits[layer] = edit
        return edits
