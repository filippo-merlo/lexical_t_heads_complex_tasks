"""Head roles from readings (paper/head-roles-design.tex, stage 1; CPU only, no forward pass).

For every head, task and half, six role scores from the J-head top-10 tokens at the last
token of the five instruction templates (``data/readings/heldout/<half>``, the readings the
rebuilt design identifies on). Every score is a rate minus a control rate, so 0 means "no
more than chance":

- ``answer``        own answer decoded      - another item's answer (mean over the half's other items)
- ``query``         own query decoded       - another item's query
- ``answer_letter`` answer's first letter   - another item's answer letter (items whose answer
                                              and query start with different letters)
- ``query_letter``  query's first letter    - another item's query letter (same items)
- ``category``      a token of the task's category set (not this item's) on the task's prompts
                    - the median rate of the same set on the other 13 tasks' prompts
- ``task_word``     a word of the task's word set on the task's prompts - the median rate of the
                    same set on the other tasks' prompts

Item tokens (``answer``, ``query``): the first token of the word with or without a leading space
and in three casings, kept if it is the whole word or a prefix of at least 4 characters
(shorter pieces such as ' CAN' for Canada match too much). With ``--sets specific`` (default)
the category set is the task's answer tokens (same rule) minus those of tasks of other
families, and the word set is the fold's 30-word list minus words in the lists of tasks of
other families and minus the task's answer and query tokens. ``--sets shared`` keeps the
other families' tokens and words and takes category prefixes of 2 characters: the first
pilot, whose task-level roles fail the control. Also per head: the generic score of the design doc
(Sec. 5.5 of head-sets-simple-design, mean cosine of the top-10 token counts over task
pairs) and, for task pairs with shared inputs, the cosine of the head's token counts between
the two tasks.

Failure control (``--control``): every prompt is scored against another item of its half and every
task against the category and word sets of a task of another family; the roles should then
vanish.

Writes ``results/roles/readings/{scores,scores_shared}[_control].npz`` (scores[half, role, task, head],
generic[half, head], pair_cos[half, pair, head], pair_peak[half, pair, 2, head]), the sets
used (``*_sets.json``) and ``meta.json``.

    .venv/bin/python scripts/roles_readings.py
    .venv/bin/python scripts/roles_readings.py --control
    .venv/bin/python scripts/roles_readings.py --sets shared [--control]
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, RESULTS, simple  # noqa: E402

OUT = RESULTS / "roles" / "readings"
HALVES = ("A", "B")
ROLES = ("answer", "query", "answer_letter", "query_letter", "category", "task_word")
TEMPLATES = tuple(f"inst{k}" for k in range(5))
PAIRS = (("next_item", "prev_item"), ("country-capital", "country-currency"),
         ("english-french", "english-german"), ("english-french", "english-spanish"),
         ("english-german", "english-spanish"), ("antonym", "synonym"))
J = 1  # READOUTS = ("logithead", "jhead")
N_HEADS = 1024


def tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


class Vocab:
    def __init__(self, tok):
        self.tok = tok
        self.V = len(tok)
        raw = [tok.decode([i]) for i in range(self.V)]
        self.norm = np.array([r.strip().lower() for r in raw], dtype=object)
        self.letter = np.full(self.V, -1)
        for i, r in enumerate(raw):
            m = re.fullmatch(r"[\s\-_]?([a-zA-Z])", r)
            if m:
                self.letter[i] = ord(m.group(1).lower()) - 97

    def first_ids(self, word: str, min_prefix: int) -> set[int]:
        w = word.strip().lower()
        out = set()
        for form in {word.strip(), word.strip().lower(), word.strip().capitalize()}:
            for pre in ("", " "):
                enc = self.tok.encode(pre + form, add_special_tokens=False)
                if not enc:
                    continue
                piece = self.norm[enc[0]]
                if piece == w or (len(piece) >= min_prefix and w.startswith(piece)):
                    out.add(enc[0])
        return out

    def mask(self, ids) -> np.ndarray:
        m = np.zeros(self.V, bool)
        if ids:
            m[list(ids)] = True
        return m


def first_letter(word: str) -> int:
    c = str(word).strip()[:1].lower()
    return ord(c) - 97 if "a" <= c <= "z" else -1


def csr(sets: list[set[int]], V: int):
    """token -> items whose set contains it, as (indptr, indices)."""
    rows = [[] for _ in range(V)]
    for item, s in enumerate(sets):
        for t in s:
            rows[t].append(item)
    indptr = np.zeros(V + 1, np.int64)
    indptr[1:] = np.cumsum([len(r) for r in rows])
    indices = np.fromiter((i for r in rows for i in r), np.int64, count=int(indptr[-1]))
    return indptr, indices


def other_items_hit(top: np.ndarray, indptr, indices, n_items: int, own_mask: np.ndarray, exclude: int) -> np.ndarray:
    """Per head, the number of distinct items (not ``exclude``) whose set is in the head's top 10,
    ignoring top tokens of the prompt's own item (``own_mask``)."""
    flat = top.reshape(-1)
    keep = ~own_mask[flat]
    pos = np.nonzero(keep & (indptr[flat + 1] > indptr[flat]))[0]
    if len(pos) == 0:
        return np.zeros(N_HEADS)
    tk = flat[pos]
    starts, lens = indptr[tk], indptr[tk + 1] - indptr[tk]
    heads = np.repeat(pos // 10, lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    items = indices[np.repeat(starts, lens) + offs]
    ok = items != exclude
    key = np.unique(heads[ok] * n_items + items[ok])
    return np.bincount(key // n_items, minlength=N_HEADS).astype(float)


def load_half(task: str, half: str):
    tops, items = [], []
    for w in TEMPLATES:
        z = np.load(DATA / "readings" / "heldout" / half / "simple" / task / f"{w}.npz")
        if len(z["id_items"]):
            tops.append(z["top"][:, J])
            items += [int(i) for i in z["id_items"]]
    return np.concatenate(tops), np.array(items)


FAMILY_OF = (("antonym", "synonym"), ("singular-plural", "present-past"),
             ("country-capital", "country-currency", "landmark-country", "park-country", "product-company"),
             ("next_item", "prev_item"), ("english-french", "english-german", "english-spanish"))


def wrong_family_partner() -> dict[str, str]:
    """Failure control: each task's category and word sets replaced by those of a task of the next family."""
    out = {}
    for f, fam in enumerate(FAMILY_OF):
        nxt = FAMILY_OF[(f + 1) % len(FAMILY_OF)]
        for i, t in enumerate(fam):
            out[t] = nxt[i % len(nxt)]
    return out


def derangement(n: int, rng) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--control", action="store_true",
                        help="failure control: every prompt scored against another item of its half (a fixed "
                             "derangement), every task against the category and word sets of a task of another "
                             "family; writes scores_control.npz. Item and task roles should then vanish.")
    parser.add_argument("--sets", choices=("specific", "shared"), default="specific",
                        help="specific (default): a task's category tokens (whole word or >= 4-character prefix) "
                             "and list words exclude those of tasks of other families; shared: all of them, "
                             "category tokens with >= 2-character prefixes (the first pilot, which fails the "
                             "control for the task-level roles)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    voc = Vocab(tokenizer())
    V = voc.V
    tasks = list(simple.TASKS)
    partner = wrong_family_partner() if args.control else {t: t for t in tasks}
    split = json.loads((RESULTS / "heldout" / "split.json").read_text())
    data = {t: simple.dataset(t) for t in tasks}
    strict_a = {t: [voc.first_ids(str(d["output"]), 4) for d in data[t]] for t in tasks}
    strict_q = {t: [voc.first_ids(str(d["input"]), 4) for d in data[t]] for t in tasks}
    loose_a = {t: [voc.first_ids(str(d["output"]), 2) for d in data[t]] for t in tasks}
    loose_q = {t: [voc.first_ids(str(d["input"]), 2) for d in data[t]] for t in tasks}
    family = {t: f for f, fam in enumerate(FAMILY_OF) for t in fam}
    if args.sets == "shared":
        cat_ids = {t: set().union(*loose_a[t]) for t in tasks}
    else:
        strict_all = {t: set().union(*strict_a[t]) for t in tasks}
        cat_ids = {t: strict_all[t] - set().union(*(strict_all[u] for u in tasks if family[u] != family[t]))
                   for t in tasks}
    cat_mask = {t: voc.mask(cat_ids[t]) for t in tasks}

    scores = np.full((2, len(ROLES), len(tasks), N_HEADS), np.nan)
    generic = np.zeros((2, N_HEADS))
    pair_cos = np.zeros((2, len(PAIRS), N_HEADS))
    pair_peak = np.zeros((2, len(PAIRS), 2, N_HEADS))
    n_prompts = np.zeros((2, len(tasks)), int)
    n_letter_items = np.zeros((2, len(tasks)), int)

    for hi, half in enumerate(HALVES):
        fold = 1 if half == "A" else 2  # fold 1 identifies on A, fold 2 on B
        lists = json.loads((RESULTS / "rebuild" / "additions" / f"fold{fold}" / "word_lists.json").read_text())["tasks"]
        tops = {t: load_half(t, half) for t in tasks}
        # task-word masks: list words without any answer or query token of the task
        word_mask = {}
        for t in tasks:
            words = set(lists[t]["words"])
            if args.sets == "specific":
                words -= set().union(*(lists[u]["words"] for u in tasks if family[u] != family[t]))
            m = np.array([n in words for n in voc.norm])
            m &= ~voc.mask(set().union(*loose_a[t], *loose_q[t]))
            word_mask[t] = m
        # cross-task rates of every task's category and word sets
        cross_cat = np.zeros((len(tasks), len(tasks), N_HEADS))   # [set task, prompt task, head]
        cross_word = np.zeros((len(tasks), len(tasks), N_HEADS))
        for pi, pt in enumerate(tasks):
            top = tops[pt][0]
            for si, st in enumerate(tasks):
                cross_cat[si, pi] = cat_mask[st][top].any(-1).mean(0)
                cross_word[si, pi] = word_mask[st][top].any(-1).mean(0)
        counts = {}
        for ti, t in enumerate(tasks):
            top, items = tops[t]
            members = sorted(split[t][half])
            n = len(members)
            perm = derangement(n, rng) if args.control else np.arange(n)
            index = {item: int(perm[k]) for k, item in enumerate(members)}
            A = [strict_a[t][i] - strict_q[t][i] for i in members]
            Q = [strict_q[t][i] - strict_a[t][i] for i in members]
            own_loose = [loose_a[t][i] | loose_q[t][i] for i in members]
            la = np.array([first_letter(data[t][i]["output"]) for i in members])
            lq = np.array([first_letter(data[t][i]["input"]) for i in members])
            diff = (la >= 0) & (lq >= 0) & (la != lq)
            ia, xa = csr(A, V)
            iq, xq = csr(Q, V)
            acc = np.zeros((len(ROLES), N_HEADS))
            n_letter = 0
            cat_own = np.zeros(N_HEADS)
            for p in range(len(top)):
                k = index[int(items[p])]
                t10 = top[p]
                own = voc.mask(own_loose[k])
                a_own = voc.mask(A[k])[t10].any(-1)
                q_own = voc.mask(Q[k])[t10].any(-1)
                a_oth = other_items_hit(t10, ia, xa, n, own, k) / (n - 1)
                q_oth = other_items_hit(t10, iq, xq, n, own, k) / (n - 1)
                acc[0] += a_own - a_oth
                acc[1] += q_own - q_oth
                cat_own += (cat_mask[partner[t]] & ~own)[t10].any(-1)
                if diff[k]:
                    L = voc.letter[t10]
                    has = np.zeros((N_HEADS, 26), bool)
                    for j in range(10):
                        v = L[:, j]
                        ok = v >= 0
                        has[np.nonzero(ok)[0], v[ok]] = True
                    others = np.delete(np.arange(n), k)
                    others = others[diff[others]]
                    acc[2] += has[:, la[k]] - has[:, la[others]].mean(1)
                    acc[3] += has[:, lq[k]] - has[:, lq[others]].mean(1)
                    n_letter += 1
            P = len(top)
            scores[hi, 0, ti] = acc[0] / P
            scores[hi, 1, ti] = acc[1] / P
            if n_letter >= 20:
                scores[hi, 2, ti] = acc[2] / n_letter
                scores[hi, 3, ti] = acc[3] / n_letter
            others = [j for j in range(len(tasks)) if j != ti]
            si = tasks.index(partner[t])  # the task itself unless --control
            scores[hi, 4, ti] = cat_own / P - np.median(cross_cat[si, others], 0)
            scores[hi, 5, ti] = cross_word[si, ti] - np.median(cross_word[si, others], 0)
            n_prompts[hi, ti] = P
            n_letter_items[hi, ti] = n_letter
            # token counts per head for the generic score and the pair contrast
            key = (np.arange(N_HEADS)[None, :, None] * V + top).reshape(-1)
            u, c = np.unique(key, return_counts=True)
            counts[t] = (u, c / P)
            print(half, t, P, "prompts,", n_letter, "letter prompts", flush=True)

        def cosine(a, b):
            (ua, ca), (ub, cb) = a, b
            na = np.bincount(ua // V, weights=ca ** 2, minlength=N_HEADS)
            nb = np.bincount(ub // V, weights=cb ** 2, minlength=N_HEADS)
            common, i1, i2 = np.intersect1d(ua, ub, return_indices=True)
            dot = np.bincount(common // V, weights=ca[i1] * cb[i2], minlength=N_HEADS)
            return dot / np.sqrt(na * nb + 1e-12)

        def peak(a):
            u, c = a
            out = np.zeros(N_HEADS)
            np.maximum.at(out, u // V, c)
            return out

        pairs = [(a, b) for i, a in enumerate(tasks) for b in tasks[i + 1:]]
        generic[hi] = np.mean([cosine(counts[a], counts[b]) for a, b in pairs], 0)
        for k, (a, b) in enumerate(PAIRS):
            pair_cos[hi, k] = cosine(counts[a], counts[b])
            pair_peak[hi, k, 0], pair_peak[hi, k, 1] = peak(counts[a]), peak(counts[b])

    OUT.mkdir(parents=True, exist_ok=True)
    name = ("scores" if args.sets == "specific" else "scores_shared") + ("_control" if args.control else "")
    np.savez(OUT / f"{name}.npz", scores=scores, generic=generic,
             pair_cos=pair_cos, pair_peak=pair_peak, n_prompts=n_prompts, n_letter_items=n_letter_items)
    if args.control:
        (OUT / "control.json").write_text(json.dumps({"item_derangement_seed": 0, "task_partner": partner}, indent=1))
    (OUT / f"{name}_sets.json").write_text(json.dumps({
        "category_tokens": {t: int(cat_mask[t].sum()) for t in tasks},
        "task_words_fold2_list": {t: sorted(set(voc.norm[np.nonzero(word_mask[t])[0]])) for t in tasks}}, indent=1))
    if args.control:
        return
    (OUT / "meta.json").write_text(json.dumps({
        "halves": HALVES, "roles": ROLES, "tasks": tasks, "pairs": PAIRS, "templates": TEMPLATES,
        "readout": "jhead", "top_k": 10, "readings": "data/readings/heldout/<half>/simple",
        "word_lists": "results/rebuild/additions/fold<k>/word_lists.json (fold 1 = half A, fold 2 = half B)",
        "letter_min_prompts": 20,
    }, indent=1))


if __name__ == "__main__":
    main()
