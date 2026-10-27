"""What each run keeps per task and wording, and where it is stored.

For each wording (one prompt template, shot count or prompt variant) a run keeps:
- every item's behaviour (``records``: item, correct, straight away, prediction);
- the identification prompts: the first ``CAP`` answered correctly and straight away
  (P+, as Yang et al.'s ``exp_size``), with every head's top-10 tokens under both readouts;
- the mean set: the first ``CAP`` such prompts whose item is not patched in the causal
  check, with the sum of their ``z`` (Yang et al.'s holdout and ``average_size``).
The fixing wording (1-shot, or the original rules prompt) defines the patched items: its
first ``CAP`` prompts in P+.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from lth import DATA

CAP = 100
RUNS = Path(os.environ.get("LTH_RUNS", DATA / "runs"))


def run_dir(family: str, task: str) -> Path:
    return RUNS / family / task.replace("/", "__")


class Collector:
    def __init__(self, lm, *, fixed: set[str] | None = None, fixing: bool = False,
                 require_straight: bool = True, fix_cap: int = CAP):
        self.lm = lm
        self.fix_cap = fix_cap
        self.fixing = fixing
        self.require_straight = require_straight
        self.fixed: list[str] = []
        self.known_fixed = set(fixed or ())
        self.records: list[dict] = []
        self.id_items: list[str] = []
        self.top: list[torch.Tensor] = []
        self.mean_items: list[str] = []
        self.z_sum = torch.zeros(lm.spec.n_layers, lm.spec.d_source, dtype=torch.float64)

    def add(self, item: str, z: torch.Tensor, *, correct: bool, straight: bool = True, **record):
        self.records.append({"item": item, "correct": correct, "straight": straight, **record})
        if not self.wants(item, correct=correct, straight=straight):
            return
        if len(self.id_items) < CAP:
            self.id_items.append(item)
            self.top.append(self.lm.top_tokens(z))
        if self.fixing and len(self.fixed) < self.fix_cap:
            self.fixed.append(item)
        elif item not in self.known_fixed and len(self.mean_items) < CAP:
            self.mean_items.append(item)
            self.z_sum += z.double().cpu()

    def wants(self, item: str, *, correct: bool, straight: bool = True) -> bool:
        """Whether a prompt joins P+ and is still needed (identification, fixing or mean set).

        In reasoning mode (``require_straight=False``) P+ asks only for a correct answer.
        """
        if not correct or (self.require_straight and not straight):
            return False
        return (len(self.id_items) < CAP or (self.fixing and len(self.fixed) < self.fix_cap)
                or (item not in self.known_fixed and len(self.mean_items) < CAP))

    def save(self, directory: Path, wording: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        top = torch.stack(self.top).numpy() if self.top else np.zeros((0, 2, 0, 10), np.int32)
        np.savez(directory / f"{wording}.npz", id_items=np.array(self.id_items, dtype=str),
                 top=top, mean_items=np.array(self.mean_items, dtype=str),
                 z_sum=self.z_sum.numpy(), fixed=np.array(self.fixed, dtype=str))
        (directory / f"{wording}.json").write_text(json.dumps(self.records, ensure_ascii=False))


def done(directory: Path, wording: str) -> bool:
    return (directory / f"{wording}.npz").exists() and (directory / f"{wording}.json").exists()


def load(directory: Path, wording: str) -> dict:
    arrays = dict(np.load(directory / f"{wording}.npz"))
    arrays["records"] = json.loads((directory / f"{wording}.json").read_text())
    return arrays


def fixed_items(directory: Path, wording: str) -> set[str]:
    return set(np.load(directory / f"{wording}.npz")["fixed"].tolist())
