"""Lexical task heads (Yang et al., 2026) on simple and complex tasks, read with Logit-head and J-head."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# Smoke runs read and write elsewhere: LTH_RUNS, LTH_RESULTS and LTH_MATERIALS override
# the run, result and checked-materials folders.
RESULTS = Path(os.environ.get("LTH_RESULTS", ROOT / "results"))
MATERIALS = Path(os.environ.get("LTH_MATERIALS", ROOT / "materials"))
THIRD_PARTY = ROOT / "third_party"
YANG = THIRD_PARTY / "Lexical_Task_Representations"
FU = THIRD_PARTY / "rules-vs-examples"

LENS_KEY = "llama3.1-8b-it"
READOUTS = ("logithead", "jhead")
