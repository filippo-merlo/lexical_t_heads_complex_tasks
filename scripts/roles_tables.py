"""LaTeX tables of paper/head-roles-design.tex (stage 1) from results/roles/readings.

    .venv/bin/python scripts/roles_tables.py
writes paper/tables/roles_classes.tex, roles_control.tex, roles_examples.tex
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import PAPER, RESULTS  # noqa: E402

DIR = RESULTS / "roles" / "readings"
OUT = PAPER / "tables"
NAMES = {"answer": "item answer", "query": "item query", "answer_letter": "answer's first letter",
         "query_letter": "query's first letter", "category": "answer category", "task_word": "task word"}


def tex(s: str) -> str:
    s = s.replace("\\", "\x00")  # backslash last, so that its braces are not escaped
    return (s.replace("{", "\\{").replace("}", "\\}").replace("\x00", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
            .replace("#", "\\#").replace("$", "\\$")
            .replace("^", "\\^{}").replace("~", "\\~{}"))


def classes() -> str:
    S = json.loads((DIR / "summary.json").read_text())
    lines = []
    for r, name in NAMES.items():
        cells = []
        for h in "AB":
            d = S["halves"][h]["roles"][r]
            cells += [str(d["heads"]), str(d["constant"]), str(d["broad"]), str(d["family"]), str(d["single"]),
                      f"{d['held_on_other_half']:.2f}"]
        corr = S["halves"]["A"]["score_correlation_active"][r]
        lines.append(f"{name} & " + " & ".join(cells) + f" & {corr:.2f} \\\\")
    body = "\n".join(lines)
    a, b = S["halves"]["A"], S["halves"]["B"]
    return rf"""\begin{{table}}[tbp]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}l rrrrrr rrrrrr r@{{}}}}
\toprule
 & \multicolumn{{6}}{{c}}{{identified on half $A$}} & \multicolumn{{6}}{{c}}{{identified on half $B$}} & \\
\cmidrule(lr){{2-7}}\cmidrule(lr){{8-13}}
Role & heads & const. & broad & fam. & single & held & heads & const. & broad & fam. & single & held & $r$ \\
\midrule
{body}
\midrule
any role & {a['heads_with_a_role']} & \multicolumn{{5}}{{l}}{{{a['several_primary_roles']} with several primary roles}} & {b['heads_with_a_role']} & \multicolumn{{5}}{{l}}{{{b['several_primary_roles']} with several primary roles}} & \\
\bottomrule
\end{{tabular}}
\caption{{Heads with each role at $\tau=0.3$, by how many tasks the role recurs in (Sec.~\ref{{sec:classes}}): constant (at least 85\% of the tasks where the role can be measured), broad (two or more families), family (two or more tasks of one family), single (one task). \emph{{held}}: share of the (head, task) roles found again at $\tau$ on the other half. $r$: correlation of the scores between halves over the (head, task) pairs at $\tau$ on either half. Source: \texttt{{results/roles/readings/summary.json}}.}}
\label{{tab:classes}}
\end{{table}}
"""


def control() -> str:
    def row(f):
        S = json.loads((DIR / f).read_text())
        return [max(S["halves"][h]["max_heads_in_a_task"][r] for h in "AB") for r in NAMES]

    real = row("summary.json")
    rows = [("roles, specific sets (as used)", real),
            ("control, specific sets", row("summary_control.json")),
            ("control, shared sets (first pilot)", row("summary_shared_control.json")),
            ("control, specific sets, $\\tau=0.5$", row("summary_tau0.5_control.json"))]
    body = "\n".join(f"{name} & " + " & ".join(str(v) for v in vals) + " \\\\" for name, vals in rows)
    head = " & ".join(("answer", "query", "ans.\\ letter", "qry.\\ letter", "category", "task word"))
    return rf"""\begin{{table}}[tbp]
\centering\small
\begin{{tabular}}{{@{{}}l rrrrrr@{{}}}}
\toprule
 & {head} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Failure control: the largest number of heads that pass $\tau$ (0.3 unless stated) in any one task and half. In the control every prompt is scored against another item of its half and every task against the category and word sets of a task of another family (Sec.~\ref{{sec:checks}}); the gate for stage~2 is at most 10. The first row gives the same count for the real scores. Sources: \texttt{{results/roles/readings/summary*.json}}, field \texttt{{max\_heads\_in\_a\_task}}.}}
\label{{tab:control}}
\end{{table}}
"""


def examples() -> str:
    rows = list(csv.DictReader(open(DIR / "examples.csv")))
    lines = []
    last = None
    for r in rows:
        # Latin Modern has no Cyrillic or CJK glyphs: tokens in other scripts are left out
        toks = [t for t in r["tokens"].split(" | ") if all(ord(c) <= 0x024F for c in t)][:6]
        cell = ", ".join(tex(t) for t in toks)
        head = r["head"] if r["head"] != last else ""
        last = r["head"]
        lines.append(f"\\texttt{{{head}}} & {tex(r['task'])} & \\texttt{{\\small {cell}}} \\\\")
    body = "\n".join(lines)
    return rf"""\begin{{table}}[tbp]
\centering\small
\begin{{tabularx}}{{\textwidth}}{{@{{}}llL@{{}}}}
\toprule
Head & Task & Most frequent tokens in the top 10 (share of prompts) \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\caption{{What some heads decode, on half $A$ (J-head, last token, five instruction templates). Source: \texttt{{results/roles/readings/examples.csv}} (\texttt{{scripts/roles\_examples.py}}), the first six of the eight most frequent tokens that are in Latin script. Item-level roles (letters, answers) change from item to item and show only weakly in counts pooled over items.}}
\label{{tab:examples}}
\end{{table}}
"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, fn in (("roles_classes", classes), ("roles_control", control), ("roles_examples", examples)):
        (OUT / f"{name}.tex").write_text(fn())
        print("wrote", OUT / f"{name}.tex")


if __name__ == "__main__":
    main()
