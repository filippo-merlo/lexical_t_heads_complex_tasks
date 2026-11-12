"""Check 1 with Davidson et al.'s instruction function vectors instead of mean head patching.

Davidson, Gureckis, Lake and Williams (2025, arXiv 2505.12075, Sec. 2, App. B, E) extract
a function vector (FV) from instruction prompts, following Todd et al. (2024):

- mean head outputs: each head's output at the last token, averaged over 100 correct
  instruction prompts, 20 for each of the 5 best instructions;
- head scores: the instruction of a prompt is replaced by an uninformative one, the query
  kept; a head's causal indirect effect (CIE) is the gain in probability of the answer's
  first token when that head alone is set to its mean output; three kinds of
  uninformative instruction (token sequences sampled to match the instruction's
  conditional probabilities, WikiText-103 text, other tasks' instructions), matched in
  token length and log-probability, five per instruction, 25 prompts per kind; the CIE is
  averaged over the kinds, and over tasks, and the 20 heads with the largest mean form one
  set for all tasks;
- the FV of a task is the sum of those heads' mean outputs (in the residual basis), added
  to the residual stream after layer |L|/3 (layer 11 on Llama-3.1-8B) at the last token.

Here the instructions are our rule wordings inst1-inst4, the templates of the form
"<instruction> Q: {input}\\nA:" (Davidson et al.'s instruction prompt; the cloze template
inst0 has no instruction apart from the query); 25 correct prompts each give the mean
(100 prompts), and 5 more each the CIE prompts (20 per kind). "Correct" is the flexible
score, as in the rest of the study. The heads are chosen over the four tasks of this run.

The FV is then tested in our check 1 (``scripts/patch.py``, family ``simple``), unchanged
apart from the intervention: the zero-shot prompts of the first 100 items answered
correctly 1-shot, replies generated and scored as there, baseline = their unpatched
accuracy; alpha times the FV is added at the last prompt token only (alpha = 1 is
Davidson et al.'s intervention; the other alphas are our check's grid). Control: the same
heads' mean outputs under our check 1's control mean (the control task's 1-shot prompts at
its items correct under inst1).

``--n-heads`` changes the number of heads (20 is theirs); ``--cie-from`` reuses another
run's CIE, which does not depend on it. Writes ``results/davidson_fv/`` (or ``--out``): ``uninformative.json`` (every corrupted instruction),
``cie.npz`` (CIE per task and kind, ``[n_layers, n_heads]``), ``heads.json`` (the 20 heads),
``fv.npz`` (task and control FVs), ``patching_simple.csv`` and
``answers/patching_simple.jsonl.gz`` (same columns as ``patch.py``'s).

    .venv/bin/python scripts/davidson_fv.py --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, gpu, simple  # noqa: E402

TASKS = ("landmark-country", "product-company", "english-french", "present-past")
INSTRUCTIONS = ("inst1", "inst2", "inst3", "inst4")
# E14 (``--instructions all``): also the 10 new templates of the question-answer form
# (``lth.simple.E4B_WORDINGS``, cells qa-named and qa-unnamed), which end in the same query.
E4B_QA = tuple(w for w in simple.E4B_WORDINGS if w.startswith(simple.E4B_PREFIX + "qa-"))
QUERY = " Q: {input}\nA:"
N_MEAN = 25  # correct prompts per instruction for the mean (100 in all; Davidson et al.: 20 x 5)
N_CIE = 5  # uninformative instructions per instruction and kind, each with its own query
N_HEADS = 20  # |A|
LAYER = 11  # |L|/3 for Llama-3.1-8B, as Davidson et al. state
KINDS = ("equiprobable", "wikitext", "other_task")
TAU = 0.1  # equiprobable sampling: starting tolerance and its increment (log-probability)
WIKI_LINES = 20_000  # WikiText-103 lines sampled for candidate prefixes
WIKI_POOL = 100  # N: candidates of about the instruction's length
OTHER_POOL = 20  # candidates of about the instruction's length among other tasks' instructions
SEED = 0


def instruction(task: str, wording: str) -> str:
    template = (simple.e4b_templates()[task][wording] if wording.startswith(simple.E4B_PREFIX)
                else simple.instruction_dict()[task][wording[4:]])
    assert template.endswith(QUERY), template
    return template[: -len(QUERY)]


def pad_batch(lm, rows: list[list[int]]):
    import torch

    width = max(len(r) for r in rows)
    pad = min(lm.eos)
    ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows], device=lm.device)
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=lm.device)
    return ids, mask, (mask.cumsum(-1) - 1).clamp(min=0)


def token_logprobs(lm, rows: list[list[int]], batch: int = 50) -> list[list[float]]:
    """``log p(t_i | t_<i)`` for every token after the first (BOS) of each row."""
    import torch

    out = []
    for b in range(0, len(rows), batch):
        chunk = rows[b:b + batch]
        ids, mask, positions = pad_batch(lm, chunk)
        with torch.no_grad():
            logits = lm.model.hf_model(input_ids=ids, attention_mask=mask, position_ids=positions).logits
        logp = logits.float().log_softmax(-1)
        width = ids.shape[1]
        for r, row in enumerate(chunk):
            start = width - len(row)
            out.append([logp[r, start + i - 1, row[i]].item() for i in range(1, len(row))])
    return out


def equiprobable(lm, q_ids: list[int], target: list[float], n: int, rng) -> list[list[int]]:
    """``n`` token sequences of the instruction's length, token ``l`` drawn uniformly among
    the tokens whose log-probability given the sequence so far is within ``tau`` of the
    instruction's ``l``-th token's (``tau`` raised by ``TAU`` until one qualifies); added
    vocabulary excluded."""
    import torch

    added = sorted(lm.tokenizer.get_added_vocab().values())
    rows = [[q_ids[0]] for _ in range(n)]  # BOS
    for l in range(len(target)):
        ids = torch.tensor(rows, device=lm.device)
        with torch.no_grad():
            logp = lm.model.hf_model(input_ids=ids, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
        logp[:, added] = float("-inf")
        for r in range(n):
            tau = TAU
            while True:
                candidates = ((logp[r] - target[l]).abs() <= tau).nonzero().flatten().tolist()
                if candidates:
                    break
                tau += TAU
            rows[r].append(rng.choice(candidates))
    return [r[1:] for r in rows]


def closest(lm, candidates: list[list[int]], length: int, logprob: float, pool: int, n: int, rng
            ) -> list[list[int]]:
    """The ``n`` candidates with log-probability closest to ``logprob``, among ``pool`` of
    them (drawn at random if more) of the length nearest ``length``: the window ``|len -
    length| <= k`` is widened until it holds ``pool``, or every candidate."""
    k = 0
    while True:
        window = [c for c in candidates if abs(len(c) - length) <= k]
        if len(window) >= min(pool, len(candidates)):
            break
        k += 1
    if len(window) > pool:
        window = rng.sample(window, pool)
    scores = [sum(lp) for lp in token_logprobs(lm, [[lm.tokenizer.bos_token_id] + c for c in window])]
    order = sorted(range(len(window)), key=lambda i: abs(scores[i] - logprob))
    return [window[i] for i in order[:n]]


def wiki_prefixes(lm, rng) -> dict[int, list[list[int]]]:
    """Token length -> prefixes of WikiText-103 lines that end before a space, one random
    prefix per line and length (headings and empty lines left out)."""
    import datasets

    lines = [t.strip() for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")["text"]]
    lines = [t for t in lines if t and not t.startswith("=")]
    by_length: dict[int, list[list[int]]] = {}
    for text in rng.sample(lines, WIKI_LINES):
        cuts = [i for i, ch in enumerate(text) if ch == " "]
        seen = set()
        for cut in cuts:
            ids = lm.tokenizer(text[:cut], add_special_tokens=False).input_ids
            if len(ids) > 70:
                break
            if len(ids) not in seen:
                seen.add(len(ids))
                by_length.setdefault(len(ids), []).append(ids)
    return by_length


def uninformative(lm, task: str, rng, wiki, wordings=INSTRUCTIONS, new_others: bool = False) -> dict:
    """For each instruction of ``task`` in ``wordings``: its tokens, log-probability, and ``N_CIE``
    uninformative instructions of each kind (token ids). ``new_others`` (E14): the other tasks'
    new question-answer instructions join the pool of the ``other_task`` kind."""
    own = {instruction(task, w) for w in wordings}
    others = []
    for other, templates in simple.instruction_dict().items():
        if other == task:
            continue
        for template in list(templates.values()) + ([simple.e4b_templates()[other][w] for w in E4B_QA]
                                                     if new_others and other in simple.e4b_templates() else []):
            if template.endswith(QUERY) and template[: -len(QUERY)] not in own:
                others.append(template[: -len(QUERY)])
    others = [lm.tokenizer(t, add_special_tokens=False).input_ids for t in dict.fromkeys(others)]
    wiki_all = [c for cs in wiki.values() for c in cs]
    out = {}
    for wording in wordings:
        q = lm.tokenizer(instruction(task, wording)).input_ids  # with BOS
        target = token_logprobs(lm, [q])[0]
        logprob, length = sum(target), len(q) - 1
        near = [c for k in range(length - 3, length + 4) for c in wiki.get(k, [])]
        out[wording] = dict(
            instruction=instruction(task, wording), length=length, logprob=logprob,
            equiprobable=equiprobable(lm, q, target, N_CIE, rng),
            wikitext=closest(lm, near or wiki_all, length, logprob, WIKI_POOL, N_CIE, rng),
            other_task=closest(lm, others, length, logprob, OTHER_POOL, N_CIE, rng))
    return out


def answer_prob(lm, rows: list[list[int]], targets: list[int], edits=None):
    """Probability of ``targets[r]`` as the next token after row ``r``, under ``edits``."""
    import contextlib

    import torch

    from src.hooks import ZinPatcher

    ids, mask, positions = pad_batch(lm, rows)
    with torch.no_grad(), (ZinPatcher(lm.model.layers, edits) if edits else contextlib.nullcontext()):
        logits = lm.model.hf_model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                   logits_to_keep=1).logits[:, -1]
    p = logits.float().softmax(-1)
    return p[torch.arange(len(rows)), torch.tensor(targets, device=p.device)]


def cie(lm, rows: list[list[int]], targets: list[int], mean_z):
    """CIE of every head, ``[n_layers, n_heads]``: mean over rows of the answer's probability
    with that head set to its mean at the last token, minus the probability unpatched."""
    import torch

    n_heads, dim = lm.spec.n_heads, lm.spec.head_dim
    clean = answer_prob(lm, rows, targets)
    out = torch.zeros(lm.spec.n_layers, n_heads)
    for layer in range(lm.spec.n_layers):
        target = mean_z[layer].to(lm.device)
        head_of_row = torch.arange(n_heads, device=lm.device).repeat_interleave(len(rows))

        def edit(z, target=target, head_of_row=head_of_row):
            z = z.clone()
            for h in range(n_heads):
                cols = slice(h * dim, (h + 1) * dim)
                z[head_of_row == h, -1, cols] = target[cols].to(z.dtype)
            return z

        p = answer_prob(lm, rows * n_heads, targets * n_heads, {layer: edit})
        out[layer] = (p.view(n_heads, len(rows)) - clean[None]).mean(1).cpu()
    return out


def function_vector(lm, heads, mean_z):
    """Sum over ``heads`` of ``W_O^h`` times the head's mean ``z``, float32, ``[d_model]``."""
    import torch

    dim = lm.spec.head_dim
    fv = torch.zeros(lm.w_o[0].shape[0], device=lm.device)
    for layer, head in heads:
        cols = slice(head * dim, (head + 1) * dim)
        fv += lm.w_o[layer][:, cols].float() @ mean_z[layer][cols].to(lm.device, torch.float32)
    return fv


class ResidualAdd:
    """Adds ``alphas[row] * fv`` to the residual stream after ``LAYER`` at the last token of
    multi-token passes (the prompt), leaving single-token passes (the reply) alone."""

    def __init__(self, lm, fv, alphas):
        import torch

        self.block = lm.model.layers[LAYER]
        self.add = torch.tensor(alphas, device=lm.device, dtype=torch.float32)[:, None] * fv[None]

    def hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] == 1:
            return output
        hidden = hidden.clone()
        hidden[:, -1] = (hidden[:, -1].float() + self.add).to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def generate_fv(lm, prompts, fv, alphas, max_new_tokens, max_rows, stop):
    """``(prompt index, alpha)`` -> greedy reply with ``alpha * fv`` added, as
    ``patch.generate_patched``."""
    pairs = sorted(((p, a) for p in range(len(prompts)) for a in alphas), key=lambda pa: len(prompts[pa[0]]))
    replies = {}
    for batch in lm.token_batches([len(prompts[p]) for p, _ in pairs], max_new_tokens, max_rows):
        chosen = [pairs[r] for r in batch]
        with ResidualAdd(lm, fv, [a for _, a in chosen]):
            replies.update(zip(chosen, lm.generate_batch([prompts[p] for p, _ in chosen], max_new_tokens,
                                                         stop=stop)))
    return replies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--n-heads", type=int, default=N_HEADS, help="heads in the FV (default: theirs, 20)")
    parser.add_argument("--out", default="davidson_fv", help="output folder under results/")
    parser.add_argument("--cie-from", nargs="+", default=None,
                        help="folders under results/ whose cie.npz to reuse, merged (the items must be the "
                             "same); the uninformative instructions and the CIE are then not recomputed")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS),
                        help="simple tasks (default: the four of the first run; 'all' for all 14); the "
                             "heads are chosen over these")
    parser.add_argument("--cie-only", action="store_true", help="stop once cie.npz is written")
    parser.add_argument("--check-tasks", nargs="+", default=None,
                        help="tasks to run check 1 on (default: --tasks); the heads are still chosen over --tasks")
    parser.add_argument("--half", choices=("A", "B"), default=None,
                        help="E4 fold (results/heldout/split.json): means, CIE and control items from this "
                             "half, check-1 items from the other")
    parser.add_argument("--instructions", choices=("yang", "all"), default="yang",
                        help="E14: 'all' adds the 10 new question-answer templates per task to Yang et al.'s "
                             "inst1-inst4 (an instruction with fewer than N_CIE + 1 correct prompts is left out)")
    args = parser.parse_args()
    tasks = list(simple.TASKS) if args.tasks == ["all"] else args.tasks
    instructions = INSTRUCTIONS + (E4B_QA if args.instructions == "all" else ())
    halves = json.loads((RESULTS / "heldout" / "split.json").read_text()) if args.half else None

    def chosen(task, items):  # the items of the choosing half, in order
        return items if halves is None else [i for i in items if i in set(halves[task][args.half])]

    def tested(task, items):  # the items of the testing half, in order
        return items if halves is None else [i for i in items if i not in set(halves[task][args.half])]
    gpu.pin(args.gpu)

    import numpy as np
    import torch

    from patch import ALPHAS, PATCH_ROWS, simple_mean
    from run_outputs import SIMPLE_MAX_TOKENS, simple_answer

    from lth.evaluation import flexible_simple
    from lth.model import LM
    from lth.store import load, readings_dir

    def records(task, wording):
        return load(readings_dir("simple", task), wording)["records"]

    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    lm = LM.load()
    reuse = [RESULTS / d for d in args.cie_from] if args.cie_from else None
    wiki = None if reuse else wiki_prefixes(lm, rng)

    # Items: the patched ones (check 1), then per instruction the mean and CIE items.
    plan, means, corrupt = {}, {}, {}
    for task in tasks:
        fixed = tested(task, simple.correct_items(records(task, simple.FIXING)))[: simple.CAP]
        per = {}
        parts = []
        for wording in instructions:
            free = [i for i in chosen(task, simple.correct_items(records(task, wording))) if i not in set(fixed)]
            if len(free) <= N_CIE:
                print(f"{task} {wording}: only {len(free)} correct prompts, left out", flush=True)
                continue
            if len(free) >= N_MEAN + N_CIE:
                per[wording] = dict(mean=free[:N_MEAN], cie=free[N_MEAN:N_MEAN + N_CIE])
            else:  # too few correct prompts (next_item): CIE first, the rest for the mean
                per[wording] = dict(mean=free[:-N_CIE], cie=free[-N_CIE:])
                print(f"{task} {wording}: only {len(free)} correct prompts", flush=True)
            ids = simple.token_ids(lm.tokenizer, simple.prompts(task, wording)[0])
            parts.append((simple_mean(lm, ids, per[wording]["mean"]), len(per[wording]["mean"])))
        plan[task] = dict(fixed=fixed, items=per)
        means[task] = sum(m * n for m, n in parts) / sum(n for _, n in parts)  # over all mean prompts
        if not reuse:
            corrupt[task] = uninformative(lm, task, rng, wiki, list(per), args.instructions == "all")
        print(f"{task}: mean ready", flush=True)
    (out / "items.json").write_text(json.dumps(plan, indent=1))

    # CIE per task and kind; heads = top n_heads of the mean over kinds and tasks.
    if reuse:
        done = {}
        for d in reuse:
            done.update(json.loads((d / "items.json").read_text()))
        assert all(done[task] == json.loads(json.dumps(plan[task])) for task in tasks)
        scores = {k: v for d in reuse for k, v in np.load(d / "cie.npz").items()}
    else:
        (out / "uninformative.json").write_text(json.dumps(
            {task: {w: {**u, **{k: [lm.tokenizer.decode(c) for c in u[k]] for k in KINDS}}
                    for w, u in corrupt[task].items()} for task in tasks}, indent=1, ensure_ascii=False))
        scores = {}
    for task in tasks if not reuse else ():
        data = simple.dataset(task)
        for kind in KINDS:
            rows, targets = [], []
            for wording in plan[task]["items"]:
                for c, item in zip(corrupt[task][wording][kind], plan[task]["items"][wording]["cie"]):
                    query = lm.tokenizer(QUERY.format(input=data[item]["input"]), add_special_tokens=False).input_ids
                    rows.append([lm.tokenizer.bos_token_id] + c + query)
                    targets.append(lm.first_token(" " + data[item]["output"]))
            scores[f"{task}/{kind}"] = cie(lm, rows, targets, means[task]).numpy()
            print(f"{task} {kind}: CIE done", flush=True)
    if not reuse:
        np.savez(out / "cie.npz", **scores)
    if args.cie_only:
        return
    total = np.mean([scores[f"{t}/{k}"] for t in tasks for k in KINDS], axis=0)
    order = np.argsort(-total, axis=None)[:args.n_heads]
    heads = [tuple(int(x) for x in np.unravel_index(i, total.shape)) for i in order]
    (out / "heads.json").write_text(json.dumps(
        dict(layer=LAYER, tasks=tasks, cie_from=args.cie_from, heads=[[l, h, float(total[l, h])] for l, h in heads]), indent=1))

    # Check 1 with the FV.
    rows_out, log, fvs = [], [], {}

    def stop(text):
        return simple_answer(text) != text

    for task in args.check_tasks or tasks:
        fixed = plan[task]["fixed"]
        prompts, answers = simple.prompts(task, simple.ZERO)
        corrupt_ids = simple.token_ids(lm.tokenizer, prompts)
        unpatched = {int(r["item"]): r for r in records(task, simple.ZERO)}
        rows_out.append(dict(family="simple-fv", task=task, heads="", source="none", mean_wording="", layer="",
                             n_heads=0, n_prompts=len(fixed), n_mean=0, alpha="",
                             accuracy=sum(unpatched[i]["correct"] for i in fixed) / len(fixed),
                             first_token=sum(unpatched[i]["first_token"] for i in fixed) / len(fixed)))
        log += [dict(family="simple-fv", task=task, heads="", source="none", mean_wording="", alpha="",
                     item=str(i), expected=answers[i], predicted=unpatched[i]["answer"],
                     correct=unpatched[i]["correct"], first_token=unpatched[i]["first_token"]) for i in fixed]
        control = simple.CONTROL[task]
        items = chosen(control, simple.correct_items(records(control, simple.CONTROL_ITEMS["rule"])))
        ids = simple.token_ids(lm.tokenizer, simple.prompts(control, simple.CONTROL_PROMPTS)[0])
        kept = plan[task]["items"]
        sources = [("task", "+".join(kept), means[task], sum(len(v["mean"]) for v in kept.values())),
                   ("control", simple.CONTROL_PROMPTS, simple_mean(lm, ids, items), len(items))]
        patched = [corrupt_ids[i] for i in fixed]
        first = [lm.first_token(answers[i]) for i in fixed]
        for source, wording, mean, n_mean in sources:
            fv = function_vector(lm, heads, mean)
            fvs[f"{task}/{source}"] = fv.cpu().numpy()
            condition = dict(family="simple-fv", task=task, heads="davidson_fv", source=source, mean_wording=wording)
            replies = generate_fv(lm, patched, fv, ALPHAS, SIMPLE_MAX_TOKENS, PATCH_ROWS["simple"], stop)
            for alpha in ALPHAS:
                hits = tokens = 0
                for p, i in enumerate(fixed):
                    reply = replies[p, alpha]
                    text = lm.tokenizer.decode(reply, skip_special_tokens=True)
                    answer = simple_answer(text)
                    correct = flexible_simple({"answer_text": answer, "reference": answers[i]})
                    first_token = reply[:1] == [first[p]]
                    hits += correct
                    tokens += first_token
                    log.append(dict(**condition, alpha=alpha, item=str(i), expected=answers[i], predicted=answer,
                                    correct=correct, first_token=first_token, reply=text))
                rows_out.append(dict(**condition, layer=LAYER, n_heads=len(heads), n_prompts=len(fixed),
                                     n_mean=n_mean, alpha=alpha, accuracy=hits / len(fixed),
                                     first_token=tokens / len(fixed)))
        print(f"{task}: check 1 done", flush=True)
    np.savez(out / "fv.npz", **fvs)
    with (out / "patching_simple.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    (out / "answers").mkdir(exist_ok=True)
    with gzip.open(out / "answers" / "patching_simple.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in log)


if __name__ == "__main__":
    main()
