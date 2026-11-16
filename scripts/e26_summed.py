"""E26: every listed set patched with Davidson et al.'s summed intervention instead of head by head.

``scripts/patch.py`` unchanged (same items, the five rule means of the choosing half, control mean,
alphas, reply length, scoring, every reply saved), except that a set is not patched head by head: its
heads' mean outputs, each through its W_O, are summed into one vector and alpha times that vector is
added to the residual stream after layer 11 at the last prompt token (``davidson_fv.function_vector``,
``generate_fv``; as ``scripts/e13_fv_residual.py``). Only the intervention differs from the
head-by-head runs; Davidson et al.'s own pooled mean (E13's ``fvsum``) is not used. Takes
``scripts/patch.py``'s arguments.

    LTH_RESULTS=results/e26_summed/fold1 CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python scripts/e26_summed.py \\
        --gpu 0 --family simple --set-list --same-style --alphas 1 2 --half A --max-new-tokens 16 \\
        --task-only random --tasks antonym --part a
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch  # noqa: E402


def generate_summed(lm, prompts, heads, mean, alphas, max_new_tokens, max_rows, stop=None):
    """As ``patch.generate_patched`` (``(prompt index, alpha)`` -> reply token ids), with the heads'
    means summed into one vector added after layer 11 instead of patched head by head."""
    from davidson_fv import function_vector, generate_fv

    return generate_fv(lm, prompts, function_vector(lm, heads, mean), list(alphas), max_new_tokens, max_rows, stop)


if __name__ == "__main__":
    patch.generate_patched = generate_summed
    patch.main()
