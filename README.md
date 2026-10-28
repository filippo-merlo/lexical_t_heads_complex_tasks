# Lexical task heads beyond simple tasks

Code for the project described in `proposal-minimal.tex`. It identifies the lexical task
heads of Yang et al. (2026) in Llama-3.1-8B-Instruct, on 14 simple tasks and on the five
rule-defined tasks of Fu et al. (2026). Each attention head is read with two readouts:
the per-head logit lens (Logit-head) and Head-JLens (J-head). The identified heads are
then tested by mean activation patching.

## Setup

```bash
uv sync                      # needs ../head_level_jlens next to this repository
scripts/fetch_sources.sh     # Yang et al. and Fu et al. code (pinned), Fu et al.'s data
```

The GPT-5.5 steps call an Azure OpenAI deployment and need `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_ENDPOINT` set in the environment.

## Run

| step | command | output |
|---|---|---|
| rule wordings (GPT-5.5) | `scripts/generate_wordings.py` | `data/review/wordings.md`, then hand-checked into `materials/complex/wordings.json` |
| term lists (GPT-5.5) | `scripts/generate_terms.py` | `data/review/terms.md`, then hand-checked into `materials/complex/terms.json` |
| E0 and readings (GPU) | `scripts/run_all.sh runs` (`run_simple.py`, `run_complex.py`) | `data/runs/` |
| E0 table | `scripts/e0_table.py` | `results/e0_accuracy.csv` |
| head sets and measures | `scripts/identify_heads.py` | `results/heads.json`, `measures.csv`, `overlaps.csv` |
| causal check (GPU) | `scripts/run_all.sh patching` (`patch.py`) | `results/patching_{simple,complex}.csv` |
| results tables and figures | `scripts/results_tex.py` | `tables/`, `figures/` (proposal, Results section) |

Every GPU script skips work already on disk, so an interrupted run resumes.

## Layout

- `lth/`: the shared code.
  - `model.py`: forward passes, both readouts and patching.
  - `simple.py`: Yang et al.'s tasks and prompts.
  - `fu.py`: Fu et al.'s tasks, run through their own code.
  - `terms.py`: term lists and their token ids.
  - `store.py`: what each run keeps.
- `materials/`: the generation prompts (proposal App. B) and the checked wordings and term lists.
- `third_party/`, `data/`: fetched sources and run outputs (not tracked).
