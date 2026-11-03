"""HeadVis on our prompts: what one attention head attends to and writes, Neuronpedia-style.

    uv run marimo run  notebooks/headvis.py --headless --port 2719   # app view
    uv run marimo edit notebooks/headvis.py --headless --port 2719   # with the code

From a laptop: ``ssh -N -L 2719:localhost:2719 <node>``, then open http://localhost:2719.
Or open it in VS Code's marimo extension, with this repository's ``.venv`` as the kernel.
Pick the GPU in the notebook (the model needs about 20 GB); with ``LTH_GPU`` (an
nvidia-smi index) set, the model loads on that GPU at once.
The computation lives in ``lth/headvis.py``; this file is widgets and layout.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", app_title="HeadVis")


@app.cell(hide_code=True)
def _():
    import os
    import re
    import sys

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    ROOT = mo.notebook_dir().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    _ = alt.data_transformers.disable_max_rows()
    return alt, mo, np, os, pd, re, sys


@app.cell(hide_code=True)
def _(mo, os):
    import subprocess

    _rows = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    _gpus = [[x.strip() for x in row.split(",")] for row in _rows.strip().splitlines()]
    _labels = {i: f"{i}: {name}, {int(used) / 1024:.0f} of {int(total) / 1024:.0f} GB in use"
               for i, name, used, total in _gpus}
    _default = os.environ.get("LTH_GPU", min(_gpus, key=lambda g: int(g[2]))[0])
    gpu_choice = mo.ui.dropdown(options={v: k for k, v in _labels.items()}, value=_labels[_default], label="GPU")
    load = mo.ui.switch(value="LTH_GPU" in os.environ, label="Load the model (about a minute, 20 GB)")
    mo.hstack([mo.md("## HeadVis · Llama-3.1-8B-Instruct"), gpu_choice, load], justify="start", gap=2,
              align="center")
    return gpu_choice, load


@app.cell(hide_code=True)
def _(gpu_choice, load, mo, os, sys):
    mo.stop(not load.value, mo.md("*Pick a free GPU and switch on **Load the model**.*"))
    # The GPU pin must come before torch is imported, so the torch-touching modules are
    # imported here, after it. Once torch is in, the kernel stays on that GPU.
    from lth import gpu

    if "torch" in sys.modules:
        _on = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")
        mo.stop(_on != gpu_choice.value,
                mo.md(f"**This kernel already uses GPU {_on}.** Pick it again, or restart the kernel to change GPU."))
    else:
        gpu.pin(int(gpu_choice.value))

    from lth import fu, headvis, simple
    from lth.model import LM
    from lth.store import load_outputs, outputs_path

    return LM, fu, headvis, load_outputs, outputs_path, simple


@app.cell(hide_code=True)
def _(LM, headvis, mo):
    with mo.status.spinner("Loading the model and its J^z (about a minute)"):
        lm = LM.load()
        induction = headvis.induction_scores(lm).rename(columns={"induction": "induction (random)"})
    return induction, lm


@app.cell(hide_code=True)
def _(mo):
    get_head, set_head = mo.state((25, 7))
    return get_head, set_head


@app.cell(hide_code=True)
def _(fu, headvis, mo, simple):
    source = mo.ui.dropdown(options=["simple task", "complex task", "custom text"], value="simple task",
                            label="Prompts")
    custom = mo.ui.text_area(value=" hot: cold; big: small; fast:", rows=4, full_width=True,
                             label="Custom text: one prompt per block, blocks separated by a line `---`")
    chat = mo.ui.checkbox(label="chat template")
    simple_task = mo.ui.dropdown(options=list(simple.TASKS), value="antonym", label="Task")
    simple_wording = mo.ui.dropdown(options=[simple.ZERO, *simple.WORDINGS["rule"], *simple.WORDINGS["example"]],
                                    value="shot5", label="Wording")
    complex_task = mo.ui.dropdown(options=list(fu.STUDY_LEVELS), value=fu.STUDY_LEVELS[0], label="Task")
    complex_wording = mo.ui.dropdown(options=["orig", *[f"w{i}" for i in range(1, 7)], "examples", "combined",
                                              "bare"], value="orig", label="Wording")
    first = mo.ui.number(start=0, stop=599, value=0, label="From item")
    count = mo.ui.number(start=1, stop=50, value=10, label="Items")

    METRICS = {"Induction (random tokens)": "induction (random)", "Induction (these prompts)": "induction",
               "Previous token": "prev", "Attention entropy": "entropy", "Self attention": "self",
               "Attention on BOS": "bos", "Q-K distance": "distance", "Max attention": "max"}
    metric = mo.ui.dropdown(options=METRICS, value="Induction (random tokens)", label="Score")
    lowest = mo.ui.checkbox(label="lowest first")
    SETS = headvis.head_sets()
    set_choice = mo.ui.dropdown(options=list(SETS), value=next(iter(SETS)), searchable=True, label="Set")
    target_text = mo.ui.text(value="", placeholder="e.g.  cold", label="Rank a token")
    decoded_where = mo.ui.radio(options=["at the last token", "at every position"], value="at the last token",
                                inline=True)
    return (METRICS, SETS, chat, complex_task, complex_wording, count, custom, decoded_where, first, lowest, metric,
            set_choice, simple_task, simple_wording, source, target_text)


@app.cell(hide_code=True)
def _(chat, complex_task, complex_wording, count, custom, first, lm, load_outputs, mo, outputs_path, re, simple,
      simple_task, simple_wording, source):
    def _prompts():
        if source.value == "custom text":
            texts = [t for t in re.split(r"\n\s*---\s*\n", custom.value) if t.strip()]
            return [(t, lm.encode_chat(t) if chat.value else lm.encode_raw(t)) for t in texts]
        lo, hi = int(first.value), int(first.value) + int(count.value)
        if source.value == "simple task":
            return [(t, lm.encode_raw(t)) for t in simple.prompts(simple_task.value, simple_wording.value)[0][lo:hi]]
        rows = load_outputs(outputs_path("complex", complex_task.value, complex_wording.value))[lo:hi]
        return [(r["prompt"], lm.encode_chat(r["prompt"])) for r in rows]

    prompts = [(t, ids) for t, ids in _prompts() if len(ids) >= 3]
    mo.stop(not prompts, mo.md("*No prompt of at least 3 tokens.*"))
    return (prompts,)


@app.cell(hide_code=True)
def _(headvis, induction, lm, mo, pd, prompts):
    with mo.status.spinner(f"Scoring every head on {len(prompts)} prompt(s)"):
        _scans = [headvis.scan(lm, ids) for _, ids in prompts]
    all_scores = (pd.concat([x.scores for x in _scans]).groupby(["layer", "head"], as_index=False)
                  [list(headvis.SCORES)].mean().merge(induction, on=["layer", "head"]))
    all_scores.insert(0, "site", [headvis.head_label(l, h) for l, h in zip(all_scores["layer"], all_scores["head"])])
    return (all_scores,)


@app.cell(hide_code=True)
def _(get_head, lm, mo, set_head):
    layer_ui = mo.ui.dropdown(options=[str(i) for i in range(lm.spec.n_layers)], value=str(get_head()[0]),
                              label="Layer", on_change=lambda v: set_head((int(v), get_head()[1])))
    head_ui = mo.ui.dropdown(options=[str(i) for i in range(lm.spec.n_heads)], value=str(get_head()[1]),
                             label="Head", on_change=lambda v: set_head((get_head()[0], int(v))))
    return head_ui, layer_ui


@app.cell(hide_code=True)
def _(METRICS, SETS, all_scores, headvis, lowest, metric, mo, pd, set_choice, set_head):
    # Short column names, so every score fits the column; hovering a header gives the full name.
    SHORT = {"induction (random)": "ind rand", "induction": "ind", "prev": "prev", "entropy": "entropy",
             "self": "self", "bos": "BOS", "distance": "dist", "max": "max"}
    _tips = {SHORT[key]: label for label, key in METRICS.items()}
    _tips["stands out on"] = "the score on which this head ranks highest among all heads, on these prompts"

    def _pick(frame):
        if len(frame):
            set_head(headvis.parse_heads(str(frame["head"].iloc[0]))[0])

    # The score each head is most extreme on (highest among all heads on these prompts).
    _pct = all_scores[list(SHORT)].rank(ascending=False, pct=True)
    _best = _pct.idxmin(axis=1)
    _stands = {all_scores.at[i, "site"]: f"{SHORT[c]} {all_scores.at[i, c]:.2f} · top "
               + (f"{_pct.at[i, c]:.1%}" if _pct.at[i, c] < 0.01 else f"{_pct.at[i, c]:.0%}")
               for i, c in _best.items()}

    def _view(frame, every):
        if every:
            cols = ["site", metric.value, *[c for c in SHORT if c != metric.value]]
            return frame[cols].rename(columns={"site": "head", **SHORT}).round(2).reset_index(drop=True)
        out = frame[["site", metric.value]].rename(columns={"site": "head", **SHORT}).round(2).reset_index(drop=True)
        out["stands out on"] = out["head"].map(_stands)
        return out

    def _table(frame, every=False, **kw):
        view = _view(frame, every)
        return mo.ui.table(view, selection="single", on_change=_pick, show_download=False,
                           show_search=False, show_column_summaries=False, show_data_types=False,
                           freeze_columns_left=["head"],
                           header_tooltip={k: v for k, v in _tips.items() if k in view.columns}, **kw)

    _ranked = all_scores.sort_values(metric.value, ascending=lowest.value).head(8)
    _keys = pd.DataFrame(SETS[set_choice.value], columns=["layer", "head"])
    _in_set = _keys.merge(all_scores, on=["layer", "head"])
    finder = mo.ui.tabs({
        "By score": mo.vstack([mo.hstack([metric, lowest], justify="start", align="center"),
                               _table(_ranked, pagination=False)]),
        "Identified sets": mo.vstack([set_choice, _table(_in_set, page_size=12)]),
        "All heads": _table(all_scores, every=True, page_size=12),
    })
    return (finder,)


@app.cell(hide_code=True)
def _(all_scores, get_head, headvis, lm, mo, np, prompts):
    head = get_head()
    with mo.status.spinner(f"Attention of {headvis.head_label(*head)}"):
        detail = [headvis.scan(lm, ids, [head], patterns=True) for _, ids in prompts]
    head_scores = all_scores[(all_scores["layer"] == head[0]) & (all_scores["head"] == head[1])].iloc[0]
    query_max = np.concatenate([s.query_max[head][0][1:] for s in detail])
    distances = np.concatenate([np.arange(1, len(s.ids)) - s.query_max[head][1][1:] for s in detail])
    order = sorted(range(len(detail)), key=lambda p: -float(detail[p].query_max[head][0][1:].max()))
    return detail, distances, head, head_scores, order, query_max


@app.cell(hide_code=True)
def _(detail, head, headvis, lm, mo):
    with mo.status.spinner("Decoding the head at every position"):
        decodes = [headvis.decode(lm, s.ids, head) for s in detail]
    return (decodes,)


@app.cell(hide_code=True)
def _(decoded_where, decodes, headvis, lm, mo):
    top_decoded = headvis.top_decoded(decodes, lm.tokenizer, last_only=decoded_where.value == "at the last token")
    decoded = mo.vstack([
        mo.hstack([mo.md("**Top decoded tokens** · % of prompts (positions) with it in the head's top 10"), decoded_where], justify="start", gap=2, align="center"),
        mo.hstack([mo.vstack([mo.md("Logit-head"), mo.Html(headvis.decoded_chips_html(top_decoded["logithead"]))]),
                   mo.vstack([mo.md("J-head"), mo.Html(headvis.decoded_chips_html(top_decoded["jhead"]))])],
                  widths="equal", gap=2, align="start"),
    ], gap=0.25)
    return decoded, top_decoded


@app.cell(hide_code=True)
def _(METRICS, head_scores, mo, np):
    def _tile(label, key):
        v = head_scores[key]
        shown = "—" if np.isnan(v) else f"{v:.1f}" if key == "distance" else f"{v:.3f}"
        return (f'<div style="border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:3px 8px">'
                f'<div style="font-size:11px;opacity:.7">{label}</div>'
                f'<div style="font-size:16px;font-weight:600">{shown}</div></div>')

    tiles = mo.Html('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(128px,1fr));gap:6px">'
                    + "".join(_tile(label, key) for label, key in METRICS.items()) + "</div>")
    return (tiles,)


@app.cell(hide_code=True)
def _(alt, distances, mo, pd, query_max):
    def _hist(values, field, title, step, extent=None):
        b = alt.Bin(step=step, extent=extent) if extent else alt.Bin(step=step)
        return alt.Chart(pd.DataFrame({field: values})).mark_bar(
            color="#2a78d6", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(f"{field}:Q", bin=b, title=title),
            y=alt.Y("count():Q", title="queries"),
            tooltip=[alt.Tooltip(f"{field}:Q", bin=b, title="bin"), alt.Tooltip("count():Q", title="queries")],
        ).properties(width=220, height=100)

    _dstep = max(1, int(distances.max()) // 25)
    hist_max = _hist(query_max, "max", "max non-BOS attention per query", 0.05, [0, 1])
    hist_dist = _hist(distances, "distance", "Q-K distance (to the top key)", _dstep)
    return hist_dist, hist_max


@app.cell(hide_code=True)
def _(decodes, detail, head, headvis, lm, mo, order):
    chips = mo.vstack([
        mo.md("**Top query tokens**"),
        mo.Html(headvis.chips_html(headvis.top_tokens(detail, head, side="query", n=10, unique=True), "query")),
        mo.md("**Top key tokens**"),
        mo.Html(headvis.chips_html(headvis.top_tokens(detail, head, side="key", n=10, unique=True), "key")),
    ], gap=0.25)
    viewer = headvis.viewer_html(detail, head, order, decodes=decodes, tokenizer=lm.tokenizer)
    return chips, viewer


@app.cell(hide_code=True)
def _(detail, head, headvis, lm, mo, order, pd, target_text):
    _target = lm.tokenizer(target_text.value, add_special_tokens=False).input_ids[0] if target_text.value else None
    _rows = []
    with mo.status.spinner("Readouts"):
        for _p in order:
            _s = detail[_p]
            _r = headvis.readouts(lm, _s.ids, len(_s.ids) - 1, [head], k=5, target=_target).set_index("readout")
            _row = {"prompt": _p, "ends with": "".join(_s.tokens[-4:]),
                    "Logit-head top 5": _r.loc["logithead", "top tokens"], "J-head top 5": _r.loc["jhead", "top tokens"]}
            if _target is not None:
                _row["rank (Logit / J)"] = (f'{int(_r.loc["logithead", "target rank"])} / '
                                            f'{int(_r.loc["jhead", "target rank"])}')
            _rows.append(_row)
    readout_table = pd.DataFrame(_rows)
    readout_note = (f"Ranked: {lm.tokenizer.decode([_target])!r} (0 = top)." if _target is not None else "")
    return readout_note, readout_table


@app.cell(hide_code=True)
def _(chat, chips, complex_task, complex_wording, count, custom, decoded, decodes, detail, finder, first, head,
      head_scores, headvis, head_ui, hist_dist, hist_max, layer_ui, lm, mo, prompts, readout_note, readout_table,
      simple_task, simple_wording, source, target_text, tiles, top_decoded, viewer):
    if source.value == "custom text":
        _prompt_row = mo.vstack([mo.hstack([source, chat], justify="start", gap=2, align="center"), custom])
    elif source.value == "simple task":
        _prompt_row = mo.hstack([source, simple_task, simple_wording, first, count], justify="start", gap=2)
    else:
        _prompt_row = mo.hstack([source, complex_task, complex_wording, first, count], justify="start", gap=2)

    _site = headvis.head_label(*head)
    _lengths = [len(ids) for _, ids in prompts]
    _header = (f"<h1>HeadVis: {_site}</h1>"
               + head_scores.drop(["layer", "head"]).to_frame().T.round(3).to_html(index=False)
               + "".join(f"<h2>Top decoded tokens, {name}</h2>" + top_decoded[r].round(3).to_html(index=False)
                         for r, name in (("logithead", "Logit-head"), ("jhead", "J-head")))
               + "<h2>Last-token readouts</h2>" + readout_table.to_html(index=False) + "<h2>Prompts</h2>")
    _download = mo.download(data=headvis.viewer_html(detail, head, order=list(range(len(detail))), header=_header,
                                                     decodes=decodes, tokenizer=lm.tokenizer).encode(),
                            filename=f"headvis_{_site}.html", label="Save view")
    _defs = "\n".join(f"- **{k}**: {v}." for k, v in headvis.SCORES.items())

    _sidebar = mo.vstack([mo.hstack([layer_ui, head_ui], justify="start", gap=1),
                          mo.md("**Find a head** · tick a row to select it"),
                          finder], gap=0.75).style({"min-width": "0", "overflow-x": "auto"})
    _main = mo.vstack([
        mo.hstack([mo.md(f"## {_site}"),
                   mo.md(f"{len(prompts)} prompt(s), {min(_lengths)}–{max(_lengths)} tokens"), _download],
                  justify="start", gap=2, align="center"),
        tiles,
        mo.hstack([mo.vstack([hist_max, hist_dist], gap=0.5), chips], justify="start", gap=2, widths=[1, 1.6]),
        decoded,
        mo.md("**Top sequences by max attention** · hovering a token also shows what the head writes there"),
        mo.iframe(viewer, height="480px"),
        mo.hstack([mo.md("**What it writes at the last token** (Logit-head and J-head, top 5)"), target_text,
                   mo.md(readout_note)], justify="start", gap=2, align="center"),
        mo.ui.table(readout_table, selection=None, show_download=False, show_search=False,
                    show_column_summaries=False, show_data_types=False, pagination=False,
                    wrapped_columns=["Logit-head top 5", "J-head top 5"]),
        mo.accordion({"What the scores are": mo.md(
            "Means over the prompts. Scores skip the first query (BOS can attend only to itself); *max* scores "
            "and the token view skip the BOS key, where most heads put most of their attention.\n\n" + _defs
            + "\n- **induction (random)**: as *induction*, on BOS + 50 distinct random tokens repeated once "
            "(Neuronpedia's score; prompt-independent).")}),
    ], gap=0.75)
    mo.vstack([_prompt_row, mo.hstack([_sidebar, _main], widths=[1.6, 3], gap=2, align="start")])
    return


if __name__ == "__main__":
    app.run()
