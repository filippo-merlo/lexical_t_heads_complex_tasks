"""Lexical task heads (Yang et al., 2026) on simple and complex tasks, read with Logit-head and J-head."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MATERIALS = ROOT / "materials"
THIRD_PARTY = ROOT / "third_party"
YANG = THIRD_PARTY / "Lexical_Task_Representations"
FU = THIRD_PARTY / "rules-vs-examples"

LENS_KEY = "llama3.1-8b-it"
READOUTS = ("logithead", "jhead")
