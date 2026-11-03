"""HeadVis on our prompts: what chosen attention heads attend to and write, Neuronpedia-style.

    LTH_GPU=0 uv run marimo run  notebooks/headvis.py --headless --port 2719   # app view
    LTH_GPU=0 uv run marimo edit notebooks/headvis.py --headless --port 2719   # with the code

From a laptop: ``ssh -N -L 2719:localhost:2719 <node>``, then open http://localhost:2719.
``LTH_GPU`` is the nvidia-smi index of a free GPU (the model needs about 20 GB).
The computation lives in ``lth/headvis.py``; this file is widgets, tables and charts.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", app_title="HeadVis")


@app.cell
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
    return alt, mo, np, os, pd, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HeadVis on our prompts

    What chosen attention heads of Llama-3.1-8B-Instruct attend to on chosen prompts, and what
    they write at a token, in the manner of
    [Neuronpedia's HeadVis](https://www.neuronpedia.org/llama3.1-8b-it/head/25/7?headFinder=true).
    Neuronpedia scores each head on its own corpus; here every number comes from a live pass
    over the prompts picked below: your own text, Yang et al.'s simple-task prompts, or Fu et
    al.'s complex-task prompts as the runs used them. Exploration only: no number here is a
    result.

    Scores skip the first query (BOS can attend only to itself). "Largest attention" scores
    and the token views skip the BOS key, where most heads put most of their attention.
    """)
    return


@app.cell
def _(mo, os):
    # The GPU pin must come before torch is imported, so the torch-touching modules are
    # imported here, after it.
    from lth import gpu

    mo.stop("LTH_GPU" not in os.environ,
            mo.md("**Set `LTH_GPU`** to a free GPU's nvidia-smi index and restart, e.g. "
                  "`LTH_GPU=0 uv run marimo run notebooks/headvis.py`."))
    gpu.pin(int(os.environ["LTH_GPU"]))

    from lth import fu, headvis, simple
    from lth.model import LM
    from lth.store import load_outputs, outputs_path

    return LM, fu, headvis, load_outputs, outputs_path, simple


@app.cell
def _(LM, headvis, mo):
    with mo.status.spinner("Loading the model and its J^z (about a minute)"):
        lm = LM.load()
        induction = headvis.induction_scores(lm).rename(columns={"induction": "induction (random)"})
    return induction, lm


@app.cell
def _(fu, headvis, mo, simple):
    source = mo.ui.radio(options=["simple task", "complex task", "custom"], value="simple task",
                         label="Prompts", inline=True)
    custom = mo.ui.text_area(value=" hot: cold; big: small; fast:", rows=6, full_width=True,
                             label="One prompt per block; separate blocks with a line `---`")
    chat = mo.ui.checkbox(label="chat template, as Fu et al. (off: raw text with BOS, as Yang et al.)")
    simple_task = mo.ui.dropdown(options=list(simple.TASKS), value="antonym", label="Task")
    simple_wording = mo.ui.dropdown(options=[simple.ZERO, *simple.WORDINGS["rule"], *simple.WORDINGS["example"]],
                                    value="shot5", label="Wording")
    complex_task = mo.ui.dropdown(options=list(fu.STUDY_LEVELS), value=fu.STUDY_LEVELS[0], label="Task")
    complex_wording = mo.ui.dropdown(options=["orig", *[f"w{i}" for i in range(1, 7)], "examples", "combined",
                                              "bare"], value="orig", label="Wording")
    first = mo.ui.number(start=0, stop=599, value=0, label="First item")
    count = mo.ui.number(start=1, stop=50, value=10, label="Items")

    SETS = headvis.head_sets()
    heads_text = mo.ui.text(value="L25H7, L13H22", label="Heads (e.g. L25H7, L13H22)", full_width=True)
    head_set = mo.ui.dropdown(options=["none", *SETS], value="none", searchable=True,
                              label="plus an identified set (results/heads.json)")
    return (SETS, chat, complex_task, complex_wording, count, custom, first, head_set, heads_text, simple_task,
            simple_wording, source)


@app.cell
def _(chat, complex_task, complex_wording, count, custom, first, head_set, heads_text, mo, simple_task,
      simple_wording, source):
    if source.value == "custom":
        _row = mo.vstack([custom, chat])
    elif source.value == "simple task":
        _row = mo.hstack([simple_task, simple_wording, first, count], justify="start", gap=2)
    else:
        _row = mo.hstack([complex_task, complex_wording, first, count], justify="start", gap=2)
    mo.vstack([source, _row, mo.hstack([heads_text, head_set], justify="start", gap=2, widths=[2, 1])])
    return


@app.cell
def _(SETS, chat, complex_task, complex_wording, count, custom, first, head_set, headvis, heads_text, lm,
      load_outputs, mo, outputs_path, re, simple, simple_task, simple_wording, source):
    def _prompts():
        if source.value == "custom":
            texts = [t for t in re.split(r"\n\s*---\s*\n", custom.value) if t.strip()]
            return [(t, lm.encode_chat(t) if chat.value else lm.encode_raw(t)) for t in texts]
        lo, hi = int(first.value), int(first.value) + int(count.value)
        if source.value == "simple task":
            return [(t, lm.encode_raw(t)) for t in simple.prompts(simple_task.value, simple_wording.value)[0][lo:hi]]
        rows = load_outputs(outputs_path("complex", complex_task.value, complex_wording.value))[lo:hi]
        return [(r["prompt"], lm.encode_chat(r["prompt"])) for r in rows]

    prompts = [(t, ids) for t, ids in _prompts() if len(ids) >= 3]
    _given = headvis.parse_heads(heads_text.value) + SETS.get(head_set.value, [])
    _bad = [h for h in _given if not (h[0] < lm.spec.n_layers and h[1] < lm.spec.n_heads)]
    heads = list(dict.fromkeys(h for h in _given if h not in _bad))
    mo.stop(not prompts, mo.md("*No prompt of at least 3 tokens.*"))
    _lengths = [len(ids) for _, ids in prompts]
    mo.md(f"{len(prompts)} prompt(s), {min(_lengths)}–{max(_lengths)} tokens; {len(heads)} head(s)."
          + (f" **Ignored:** {', '.join(headvis.head_label(*h) for h in _bad)} (no such head)." if _bad else ""))
    return heads, prompts


@app.cell
def _(headvis, induction, lm, mo, pd, prompts):
    with mo.status.spinner(f"Scanning {len(prompts)} prompt(s)"):
        scans = [headvis.scan(lm, ids) for _, ids in prompts]
    per_prompt = pd.concat([x.scores.assign(prompt=p) for p, x in enumerate(scans)], ignore_index=True)
    all_scores = (per_prompt.groupby(["layer", "head"], as_index=False)[list(headvis.SCORES)].mean()
                  .merge(induction, on=["layer", "head"]))
    all_scores.insert(0, "site", [headvis.head_label(l, h) for l, h in zip(all_scores["layer"], all_scores["head"])])
    return all_scores, per_prompt, scans


@app.cell
def _(all_scores, headvis, heads, mo, pd, per_prompt):
    _keys = pd.DataFrame(heads, columns=["layer", "head"])
    chosen_scores = _keys.merge(all_scores, on=["layer", "head"])[list(all_scores.columns)].round(3)
    _per = _keys.merge(per_prompt, on=["layer", "head"]).round(3)
    _per.insert(0, "site", [headvis.head_label(l, h) for l, h in zip(_per["layer"], _per["head"])])
    _defs = "\n".join(f"- **{k}**: {v}." for k, v in headvis.SCORES.items())
    mo.vstack([
        mo.md("## Scores of the chosen heads\nMeans over the prompts; *induction (random)* is the head's "
              "score on repeated random tokens, the prompt-independent score Neuronpedia reports."),
        mo.ui.table(chosen_scores, selection=None, page_size=20) if len(chosen_scores) else mo.md("*No head given.*"),
        mo.accordion({
            "What each score is": mo.md(_defs + f"\n- **induction (random)**: as *induction*, on BOS + "
                                        f"{headvis.INDUCTION_LENGTH} distinct random tokens repeated once, mean "
                                        f"over {headvis.INDUCTION_SEQUENCES} sequences."),
            "Per prompt": mo.ui.table(_per[["site", "prompt", *headvis.SCORES]], selection=None, page_size=20),
        }),
    ])
    return (chosen_scores,)


@app.cell
def _(headvis, mo):
    metric = mo.ui.dropdown(options=[*headvis.SCORES, "induction (random)"], value="induction (random)",
                            label="Rank all heads by")
    order = mo.ui.radio(options=["highest", "lowest"], value="highest", inline=True)
    top_n = mo.ui.number(start=1, stop=1024, value=16, label="Heads")
    return metric, order, top_n


@app.cell
def _(all_scores, metric, mo, order, top_n):
    _ranked = all_scores.sort_values(metric.value, ascending=order.value == "lowest").head(int(top_n.value))
    finder = mo.ui.table(_ranked.round(3), selection="multi", page_size=16,
                         label="Select rows to add those heads to the detail view")
    mo.vstack([mo.md("## Head finder\nEvery head of the model, scored on these prompts."),
               mo.hstack([metric, order, top_n], justify="start", gap=2), finder])
    return (finder,)


@app.cell
def _(finder, headvis, heads, mo, prompts):
    _picked = [(int(l), int(h)) for l, h in zip(finder.value["layer"], finder.value["head"])] if len(finder.value) else []
    _options = list(dict.fromkeys(heads + _picked))
    mo.stop(not _options, mo.md("*Give a head above, or select one in the finder.*"))
    detail_head = mo.ui.dropdown(options={headvis.head_label(*h): h for h in _options},
                                 value=headvis.head_label(*_options[0]), label="Head")
    prompt_index = mo.ui.number(start=0, stop=len(prompts) - 1, value=0, label="Prompt")
    mo.vstack([mo.md("## One head"), mo.hstack([detail_head, prompt_index], justify="start", gap=2)])
    return detail_head, prompt_index


@app.cell
def _(detail_head, headvis, lm, mo, prompts):
    with mo.status.spinner(f"Attention of {headvis.head_label(*detail_head.value)}"):
        detail = [headvis.scan(lm, ids, [detail_head.value], patterns=True) for _, ids in prompts]
    return (detail,)


@app.cell
def _(detail, detail_head, mo, pd):
    def _sequences():
        rows = []
        for p, s in enumerate(detail):
            value, key = s.query_max[detail_head.value]
            qpos = int(value[1:].argmax()) + 1
            rows.append({"prompt": p, "tokens": len(s.ids), "max attention": round(float(value[qpos]), 3),
                         "query": f"{s.tokens[qpos]!r} @{qpos}", "key": f"{s.tokens[int(key[qpos])]!r} @{int(key[qpos])}",
                         "end of prompt": "".join(s.tokens[-12:])})
        return pd.DataFrame(rows).sort_values("max attention", ascending=False)

    sequences = _sequences()
    mo.vstack([mo.md("### Prompts by largest attention\nEach prompt's strongest query → non-BOS key link."),
               mo.ui.table(sequences, selection=None, page_size=10)])
    return (sequences,)


@app.cell
def _(detail, mo, prompt_index):
    _n = len(detail[prompt_index.value].ids)
    query = mo.ui.slider(start=1, stop=_n - 1, value=_n - 1, show_value=True, full_width=True,
                         label="Query position (default: the last prompt token, where the readouts are taken)")
    color_by = mo.ui.radio(options={"attention from the query": "row", "largest attention each token receives": "key",
                                    "largest attention each token pays": "query"},
                           value="attention from the query", inline=True, label="Shade tokens by")
    no_bos = mo.ui.checkbox(value=True, label="leave out the BOS key in the heatmap")
    return color_by, no_bos, query


@app.cell
def _(color_by, detail, detail_head, headvis, mo, np, pd, prompt_index, query):
    view = detail[prompt_index.value]
    _a = view.patterns[detail_head.value]
    _q = query.value
    if color_by.value == "row":
        _w = _a[_q].astype(float).copy()
        _w[_q + 1:] = np.nan
    else:
        _w = (view.key_max if color_by.value == "key" else view.query_max)[detail_head.value][0].astype(float)
    token_view = headvis.token_html(view.tokens, _w, marked=_q if color_by.value == "row" else None)
    _row = _a[_q, :_q + 1]
    query_keys = (pd.DataFrame({"position": range(_q + 1), "token": [repr(t) for t in view.tokens[:_q + 1]],
                                "attention": _row.round(4)})
                  .nlargest(10, "attention").reset_index(drop=True))
    mo.vstack([
        mo.md(f"### Prompt {prompt_index.value}, query {_q} ({view.tokens[_q]!r})\n"
              f"Attention on BOS: {_row[0]:.3f}. Shading is scaled to the largest non-BOS value; hover a "
              f"token for its position and value."),
        color_by, query, mo.Html(token_view),
        mo.accordion({"Keys this query attends to most (BOS included)": mo.ui.table(query_keys, selection=None)}),
    ])
    return query_keys, token_view, view


@app.cell
def _(alt, detail_head, mo, no_bos, pd, view):
    def _heatmap():
        n = len(view.ids)
        if n > 160:
            return mo.md(f"*The heatmap is drawn for prompts up to 160 tokens (this one has {n}); "
                         "use the token view above.*")
        a = view.patterns[detail_head.value]
        lo = 1 if no_bos.value else 0
        cells = pd.DataFrame([{"query": i, "key": j, "attention": float(a[i, j]),
                               "query token": repr(view.tokens[i]), "key token": repr(view.tokens[j])}
                              for i in range(1, n) for j in range(lo, i + 1)])
        size = max(4, min(14, 640 // n))
        return alt.Chart(cells).mark_rect().encode(
            x=alt.X("key:O", title="key position", axis=alt.Axis(labelAngle=0, values=list(range(0, n, 5)))),
            y=alt.Y("query:O", title="query position", axis=alt.Axis(values=list(range(0, n, 5)))),
            color=alt.Color("attention:Q", scale=alt.Scale(range=["#cde2fb", "#0d366b"]),
                            legend=alt.Legend(title="attention", orient="right")),
            tooltip=["query", "query token", "key", "key token", alt.Tooltip("attention:Q", format=".3f")],
        ).properties(width=alt.Step(size), height=alt.Step(size))

    mo.vstack([mo.md("### Attention pattern"), no_bos, _heatmap()])
    return


@app.cell
def _(alt, detail, detail_head, mo, np, pd):
    _values = np.concatenate([_s.query_max[detail_head.value][0][1:] for _s in detail])
    _hist = alt.Chart(pd.DataFrame({"max attention": _values})).mark_bar(
        color="#2a78d6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("max attention:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="largest non-BOS attention of a query"),
        y=alt.Y("count():Q", title="queries"),
        tooltip=[alt.Tooltip("max attention:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="bin"),
                 alt.Tooltip("count():Q", title="queries")],
    ).properties(width=520, height=180)
    mo.vstack([mo.md("### Largest attention per query, over all prompts\nNeuronpedia's *max attention "
                     "distribution*: a head that mostly rests on BOS piles up near 0."), _hist])
    return


@app.cell
def _(detail, detail_head, headvis, mo):
    top_keys = headvis.top_tokens(detail, detail_head.value, side="key")
    top_queries = headvis.top_tokens(detail, detail_head.value, side="query")
    mo.vstack([mo.md("### Top key and query tokens, over all prompts\nKeys: the tokens this head attends to "
                     "most (with the query that does). Queries: the tokens that attend most (with their key)."),
               mo.hstack([mo.ui.table(top_keys.round(3), selection=None, label="keys"),
                          mo.ui.table(top_queries.round(3), selection=None, label="queries")], widths="equal")])
    return top_keys, top_queries


@app.cell
def _(mo):
    target_text = mo.ui.text(value="", label="Token of interest (its first token is ranked; mind the leading space)")
    all_heads = mo.ui.checkbox(label="all heads given above, not only this one")
    return all_heads, target_text


@app.cell
def _(all_heads, detail_head, headvis, heads, lm, mo, query, target_text, view):
    _target = lm.tokenizer(target_text.value, add_special_tokens=False).input_ids[0] if target_text.value else None
    _heads = list(dict.fromkeys([detail_head.value, *heads])) if all_heads.value else [detail_head.value]
    with mo.status.spinner("Readouts"):
        readout_table = headvis.readouts(lm, view.ids, query.value, _heads, target=_target)
    _next = lm.tokenizer.decode([lm.next_token(view.ids)[0].item()])
    mo.vstack([
        mo.md(f"### What the head writes at query {query.value}\nTop-10 tokens of the head's output under "
              f"Logit-head and J-head, from the normal (sdpa) pass, as in the runs. The model's next token after "
              f"the whole prompt: {_next!r}."
              + (f" Ranked token: {lm.tokenizer.decode([_target])!r} (0 = top)." if _target is not None else "")),
        mo.hstack([target_text, all_heads], justify="start", gap=2),
        mo.ui.table(readout_table, selection=None, wrapped_columns=["top tokens"], page_size=20),
    ])
    return (readout_table,)


@app.cell
def _(chosen_scores, detail_head, headvis, mo, prompt_index, prompts, query, query_keys, readout_table, sequences,
      token_view, top_keys, top_queries):
    def _page() -> bytes:
        import html

        site = headvis.head_label(*detail_head.value)
        parts = [
            f"<h1>HeadVis: {site}, prompt {prompt_index.value}, query {query.value}</h1>",
            f"<details><summary>Prompt</summary><pre style='white-space:pre-wrap'>"
            f"{html.escape(prompts[prompt_index.value][0])}</pre></details>",
            "<h2>Tokens</h2>", token_view,
            "<h2>Keys this query attends to most</h2>", query_keys.to_html(index=False),
            "<h2>Readouts</h2>", readout_table.to_html(index=False),
            "<h2>Scores of the chosen heads</h2>", chosen_scores.to_html(index=False),
            "<h2>Prompts by largest attention</h2>", sequences.to_html(index=False),
            "<h2>Top keys</h2>", top_keys.round(3).to_html(index=False),
            "<h2>Top queries</h2>", top_queries.round(3).to_html(index=False),
        ]
        style = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;background:#fcfcfb;color:#1f1f1e}"
                 "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #d9d8d2;padding:3px 6px}")
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>HeadVis {site}</title>"
                f"<style>{style}</style></head><body>{''.join(parts)}</body></html>").encode()

    mo.download(data=_page(), filename=f"headvis_{headvis.head_label(*detail_head.value)}_p{prompt_index.value}"
                                     f"_q{query.value}.html", label="Save this view as HTML")
    return


if __name__ == "__main__":
    app.run()
