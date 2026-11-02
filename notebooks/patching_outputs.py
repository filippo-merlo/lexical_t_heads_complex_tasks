import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", app_title="Patched replies on the simple tasks")


@app.cell
def _():
    import sys

    import altair as alt
    import marimo as mo
    import pandas as pd

    ROOT = mo.notebook_dir().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from lth import replies, simple

    _ = alt.data_transformers.disable_max_rows()
    return alt, mo, pd, replies, simple


@app.cell
def _(mo):
    mo.md(r"""
    # Patched replies on the simple tasks

    What Llama-3.1-8B-Instruct answers when lexical task heads are patched, for every
    manipulation of the two causal checks of E1 and E1b (proposal Secs. 5.4 and 5.4.1).

    **Manipulations.** *Procedure*: how the head set was identified (E1: the most accurate
    template; E1b: heads that pass under at least $k$ of the five templates; the J-head $k=1$
    union split into heads that other tasks' sets also contain, *shared*, and those no other
    set contains, *unshared*; only present--past was patched with the unshared heads).
    *Readout*: Logit-head, J-head, or the heads only J-head finds (E1 only). *Head style*:
    rule or example (E1 only). *Mean*: the task's mean head output from one source wording
    (five instruction templates, five shot counts) or the control task's mean (check 1 only; one
    per mean style, so the wording choice does not apply), scaled by $\alpha$; $\alpha = 0$ sets
    the heads to zero.

    **Checks.** *Check 1*: the zero-shot prompt (`" accelerate:"`), 100 items answered
    correctly 1-shot. *Check 2*: the 1-shot prompt (`" zoom: zoomed; accelerate:"`) of up
    to 100 items answered wrongly; baseline 0.

    **Reply kinds**, checked in this order (`lth/replies.py`): *correct* (the flexible score);
    *contains the answer*, but not at the start, so the flexible score rejects it (`"opposite of
    solid"`); *names the task* instead of doing it (`"past tense of assess"`); *repeats the
    input*; *other form of the input* (same stem, e.g. `"-ing"`); *answer of another item* of
    the task (e.g. a capital of another country); *other text*; *empty*. The rules are
    heuristic: check the replies in the tables below before reading a share.
    """)
    return


@app.cell
def _(mo, replies, simple):
    task = mo.ui.dropdown(options=list(simple.TASKS), value="present-past", label="Task")
    family = mo.ui.radio(options={v: k for k, v in replies.FAMILIES.items()}, value=replies.FAMILIES["simple"],
                         label="Check")
    readout = mo.ui.dropdown(options={v: k for k, v in replies.HEAD_SETS.items()}, value="J-head",
                             label="Head set")
    chosen = mo.ui.multiselect(options=replies.procedures(), value=[p for p in replies.procedures()
                                                                      if p in ("E1", "k=1", "k=5")],
                               label="Procedures")
    head_style = mo.ui.radio(options=["rule", "example"], value="rule", label="Head style (example: E1 only)")
    source = mo.ui.radio(options={"task mean": "task", "control mean": "control"}, value="task mean",
                         label="Mean (control: check 1 only)")
    mean_style = mo.ui.radio(options=["rule", "example"], value="rule", label="Mean style")
    mo.vstack([mo.hstack([task, readout, chosen], justify="start", gap=2),
               mo.hstack([family, head_style, source, mean_style], justify="start", gap=2)])
    return chosen, family, head_style, mean_style, readout, source, task


@app.cell
def _(mean_style, mo, simple):
    wording = mo.ui.dropdown(options=["all (average)", *simple.WORDINGS[mean_style.value]],
                             value="all (average)", label="Source wording of the mean")
    wording
    return (wording,)


@app.cell
def _(chosen, family, head_style, mean_style, mo, pd, readout, replies, source, task, wording):
    def _condition(frame):
        keep = ((frame["heads"] == readout.value) & (frame["style"] == head_style.value)
                & (frame["source"] == source.value) & (frame["mean_style"] == mean_style.value))
        if wording.value != "all (average)" and source.value == "task":  # one control mean per style
            keep &= frame["mean_wording"] == wording.value
        return frame[keep]

    _frames, _missing = [], []
    baseline = None
    for _p in chosen.value:
        _f = replies.load(_p, family.value, task.value)
        if _f is None:
            _missing.append(f"{_p}: task not patched")
            continue
        if baseline is None:
            baseline = _f[_f["source"] == "none"].assign(procedure="unpatched")
        _c = _condition(_f)
        if _c.empty:
            _missing.append(f"{_p}: no {replies.HEAD_SETS[readout.value]} {head_style.value}-style set"
                            f" with this mean")
            continue
        _frames.append(_c.assign(procedure=_p))
    patched = pd.concat(_frames, ignore_index=True) if _frames else None
    mo.md("**Not shown:** " + "; ".join(_missing)) if _missing else None
    return baseline, patched


@app.cell
def _(chosen, head_style, mo, pd, readout, replies, task):
    def _sets():
        rows, e1 = [], None
        for p in chosen.value:
            sets = replies.head_sets(p, task.value)
            if readout.value == "jhead_only":
                logit, j = sets.get((head_style.value, "logithead"), []), sets.get((head_style.value, "jhead"), [])
                heads = sorted(set(j) - set(logit)) if j else []
            else:
                heads = sets.get((head_style.value, readout.value), [])
            if p == "E1":
                e1 = set(heads)
            rows.append({"procedure": p, "description": replies.PROCEDURES[p][1], "heads": len(heads),
                         "in E1 set": len(set(heads) & e1) if e1 is not None else None,
                         "layers": ", ".join(f"L{l}H{h}" for l, h in heads)})
        return pd.DataFrame(rows)

    mo.vstack([mo.md(f"## Head sets: {task.value}, {replies.HEAD_SETS[readout.value]}, {head_style.value} style"),
               mo.ui.table(_sets(), selection=None, wrapped_columns=["layers"])])
    return


@app.cell
def _(alt, baseline, mo, patched, pd, replies):
    # One fixed colour per procedure (the reference palette's eight slots, then neutral grey).
    PROC_COLORS = dict(zip(replies.PROCEDURES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                                "#6250d6", "#e34948", "#898781"]))

    def _curves():
        acc = (patched.groupby(["procedure", "alpha"])
               .agg(accuracy=("correct", "mean"),
                    top_reply=("predicted", lambda s: s.value_counts(normalize=True).iloc[0]))
               .reset_index())
        procs = [p for p in replies.PROCEDURES if p in set(acc["procedure"])]
        color = alt.Color("procedure:N", scale=alt.Scale(domain=procs, range=[PROC_COLORS[p] for p in procs]),
                          legend=alt.Legend(orient="top", title=None))
        x = alt.X("alpha:Q", title="α", axis=alt.Axis(values=sorted(acc["alpha"].unique())))
        base_line = alt.Chart(pd.DataFrame({"y": [baseline["correct"].mean()]})).mark_rule(
            strokeDash=[4, 3], color="#898781").encode(y="y:Q")

        def line(field, title, fmt):
            return alt.Chart(acc).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36)).encode(
                x=x, y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                color=color, tooltip=["procedure", "alpha", alt.Tooltip(f"{field}:Q", format=fmt)],
            ).properties(width=420, height=220)

        return alt.hconcat(line("accuracy", "accuracy", ".0%") + base_line,
                           line("top_reply", "share of the most common reply", ".0%"))

    mo.vstack([
        mo.md("## Accuracy and constant answers against α\n"
              "Accuracy over the patched items (averaged over the selected source wordings); dashed: unpatched. "
              "Right: the share of the most frequent reply, high when a patch makes the model give one answer "
              "to every item."),
        _curves() if patched is not None else mo.md("*No condition selected.*"),
    ])
    return


@app.cell
def _(alt, baseline, mo, patched, pd, replies):
    KIND_COLORS = dict(zip(replies.KINDS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                           "#c3c2b7", "#e1e0d9"]))

    def _kind_shares():
        both = pd.concat([baseline.assign(alpha=float("nan")), patched], ignore_index=True)
        both["α"] = both["alpha"].map(lambda a: "unpatched" if pd.isna(a) else f"{a:g}")
        shares = (both.groupby(["procedure", "α"]).kind.value_counts(normalize=True)
                  .rename("share").reset_index())
        shares["order"] = shares["kind"].map(replies.KINDS.index)
        return shares

    def _chart(shares):
        alphas = ["unpatched", *[f"{a:g}" for a in sorted(patched["alpha"].unique())]]
        procs = ["unpatched", *[p for p in replies.PROCEDURES if p in set(shares["procedure"])]]
        return alt.Chart(shares).mark_bar(stroke="#fcfcfb", strokeWidth=1).encode(
            x=alt.X("α:N", sort=alphas, title="α", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", stack="zero", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                    title="share of replies"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=list(replies.KINDS), range=list(KIND_COLORS.values())),
                            legend=alt.Legend(orient="right", title="reply")),
            order=alt.Order("order:Q"),
            tooltip=["procedure", "α", "kind", alt.Tooltip("share:Q", format=".0%")],
        ).properties(width=alt.Step(22), height=200).facet(
            spacing=28, column=alt.Column("procedure:N", sort=procs, title=None,
                              header=alt.Header(labelFontWeight="bold", labelAnchor="start"))
        ).resolve_scale(x="independent")

    if patched is None:
        _out = mo.md("*No condition selected.*")
    else:
        _shares = _kind_shares()
        _table = (_shares.pivot_table(index=["procedure", "α"], columns="kind", values="share", fill_value=0)
                  .reindex(columns=[k for k in replies.KINDS if k in set(_shares["kind"])]).mul(100).round(0)
                  .astype(int).reset_index())
        _out = mo.ui.tabs({"chart": _chart(_shares), "table (%)": mo.ui.table(_table, selection=None,
                                                                              page_size=40)})
    mo.vstack([mo.md("## What the model replies\nShares of reply kinds per α, all selected source wordings "
                     "pooled; the unpatched replies are the same for every procedure."), _out])
    return


@app.cell
def _(mo, patched, replies, wording):
    _alphas = sorted(patched["alpha"].unique()) if patched is not None else [0.0]
    alpha = mo.ui.dropdown(options={f"{a:g}": a for a in _alphas},
                           value=f"{4.0:g}" if 4.0 in _alphas else f"{_alphas[0]:g}", label="α")
    _wordings = sorted(patched["mean_wording"].unique()) if patched is not None else []
    item_wording = mo.ui.dropdown(options=_wordings, value=_wordings[0] if _wordings else None,
                                  label="Source wording" + (" (one, for the items)" if wording.value ==
                                                            "all (average)" else ""))
    kinds = mo.ui.multiselect(options=list(replies.KINDS), value=[], label="Only kinds (any procedure)")
    differ = mo.ui.checkbox(value=False, label="Only items where procedures disagree on correctness")
    mo.vstack([mo.md("## Item by item\nOne row per patched item: the patched prompt's reply without and with each "
                     "procedure's patch, at one α and one source wording."),
               mo.hstack([alpha, item_wording, kinds, differ], justify="start", gap=2)])
    return alpha, differ, item_wording, kinds


@app.cell
def _(alpha, baseline, differ, item_wording, kinds, mo, patched, pd):
    def _items():
        cond = patched[(patched["alpha"] == alpha.value) & (patched["mean_wording"] == item_wording.value)]
        table = baseline.set_index("item")[["input", "expected", "predicted", "kind"]].rename(
            columns={"predicted": "unpatched", "kind": "unpatched kind"})
        table["expected"] = table["expected"].str.strip()
        procs = list(dict.fromkeys(cond["procedure"]))
        for p in procs:
            c = cond[cond["procedure"] == p].set_index("item")
            table[p] = c["predicted"].str.strip().str.slice(0, 80)
            table[f"{p} kind"] = c["kind"]
        if kinds.value:
            table = table[table[[f"{p} kind" for p in procs]].isin(kinds.value).any(axis=1)]
        if differ.value and len(procs) > 1:
            correct = pd.DataFrame({p: cond[cond["procedure"] == p].set_index("item")["correct"] for p in procs})
            table = table[correct.reindex(table.index).nunique(axis=1) > 1]
        return table.reset_index()

    item_table = mo.ui.table(_items(), selection="single", page_size=20) if patched is not None else None
    item_table
    return (item_table,)


@app.cell
def _(baseline, family, item_table, item_wording, mo, patched, pd, replies, task):
    def _detail(item):
        rows = patched[(patched["item"] == item) & (patched["mean_wording"] == item_wording.value)]
        grid = rows.pivot_table(index="alpha", columns="procedure", values="reply", aggfunc="first")
        grid = grid.map(lambda r: r.strip().replace("\n", " ⏎ ")[:140] if isinstance(r, str) else r)
        unpatched = baseline.loc[baseline["item"] == item, "reply"].iloc[0]
        return mo.vstack([
            mo.md(f"**Item {item}.** Patched prompt: `{replies.patched_prompt(task.value, family.value, item)!r}`; "
                  f"expected `{rows['expected'].iloc[0].strip()}`; unpatched reply: "
                  f"`{unpatched.strip()[:140]!r}`"),
            mo.ui.table(grid.reset_index().rename(columns={"alpha": "α"}), selection=None, page_size=20),
        ])

    _rows = item_table.value if item_table is not None else None
    if _rows is None or len(_rows) == 0:
        _out = mo.md("*Select an item in the table above to see its full replies at every α.*")
    else:
        _out = _detail(int(pd.DataFrame(_rows)["item"].iloc[0]))
    mo.vstack([mo.md("### One item at every α"), _out])
    return


if __name__ == "__main__":
    app.run()
