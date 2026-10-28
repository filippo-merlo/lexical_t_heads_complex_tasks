import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Model outputs on the complex tasks")


@app.cell
def _():
    import gzip
    import html
    import json
    import sys

    import altair as alt
    import marimo as mo
    import pandas as pd

    ROOT = mo.notebook_dir().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from lth import RESULTS, fu
    from lth.store import done, run_dir

    _ = alt.data_transformers.disable_max_rows()
    return RESULTS, alt, done, fu, gzip, html, json, mo, pd, run_dir


@app.cell
def _(mo):
    mo.md(r"""
    # Model outputs on the complex tasks

    The replies of Llama-3.1-8B-Instruct to the five complex tasks of Fu et al. (2026) at the
    easy level, as scored in E0 and in the causal check of E2.

    **Modes.** *Direct*: Fu et al.'s answer limits (a few tokens), so the answer must come
    first. *Reasoning* (proposal Sec. 5.3): up to 6,000 tokens, stopped once the scored answer
    is fixed; run for Set Game, Operator Function and Tapatan.

    **Prompts.** *orig.*: the original rules prompt; *w1–w6*: the six added wordings (reasoning
    mode and the two Yes/No tasks); *examples*, *combined*: Fu et al.'s examples-only and
    rules-plus-examples prompts; *no task info*: the prompt without task information, the
    prompt patched in the causal check.

    **Outcome of a reply.** *Correct*; *wrong answer*: the scorer read an answer and it is
    wrong; *no answer*: the scorer read none (no leading Yes/No, no `\boxed{}` or an
    unparsable one, no Tapatan label: in direct mode the first line, in reasoning mode the
    first label anywhere).
    """)
    return


@app.cell
def _(fu):
    MODES = {"complex": "direct", "reasoning": "reasoning"}
    PROMPTS = {"orig": "orig.", **{f"w{i}": f"w{i}" for i in range(1, 7)},
               "examples": "examples", "combined": "combined", "bare": "no task info"}
    PROMPT_ORDER = list(PROMPTS.values())
    TASK_NAMES = {"noun_class_agreement": "Noun Class Agreement",
                  "lexical_category_inference": "Lexical Category Inference",
                  "set_game": "Set Game", "operator_func": "Operator Function", "tapatan": "Tapatan"}
    TASK_ORDER = list(TASK_NAMES.values())
    LABEL_TASKS = {"noun_class_agreement": ["Yes", "No"], "lexical_category_inference": ["Yes", "No"],
                   "tapatan": ["A win", "B win", "continue"]}

    OUTCOMES = ["correct", "wrong answer", "no answer"]
    OUTCOME_COLORS = ["#2a78d6", "#eb6834", "#a8a7a2"]
    LABEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
    NO_ANSWER_COLOR = "#a8a7a2"

    def given_answer(task, predicted):
        """The answer the scorer read from a reply, or None when it read none."""
        if task == "tapatan":
            return next((label for label in LABEL_TASKS["tapatan"]
                         if str(predicted).lower() == label.lower()), None)
        return None if predicted in (None, "", "Not boxed", "unknown") else str(predicted)

    def outcome(correct, answer):
        return "correct" if correct else "wrong answer" if answer is not None else "no answer"

    def task_name(key):
        return TASK_NAMES[fu.TASK_LEVELS[key].task]
    return (LABEL_COLORS, LABEL_TASKS, MODES, NO_ANSWER_COLOR, OUTCOMES, OUTCOME_COLORS,
            PROMPTS, PROMPT_ORDER, TASK_NAMES, TASK_ORDER, given_answer, outcome, task_name)


@app.cell
def _(MODES, PROMPTS, done, fu, given_answer, json, outcome, pd, run_dir, task_name):
    def _rows():
        for family, mode in MODES.items():
            for key in fu.STUDY_LEVELS:
                directory = run_dir(family, key)
                level = fu.TASK_LEVELS[key]
                for wording, label in PROMPTS.items():
                    if not done(directory, wording):
                        continue
                    records = {r["item"]: r for r in json.loads((directory / f"{wording}.json").read_text())}
                    for result in fu.recorded_results(directory / "fu_out" / wording):
                        item = fu.item_id(result)
                        record = records[item]
                        expected, predicted = fu.answer(level, result, reasoning=family == "reasoning")
                        answer = given_answer(level.task, predicted)
                        yield {"mode": mode, "task": task_name(key), "prompt_kind": label, "item": item,
                               "expected": str(expected), "answer": answer,
                               "outcome": outcome(record["correct"], answer),
                               "straight": record["straight"], "reply_tokens": record.get("reply_tokens"),
                               "prompt": result["prompt"], "reply": record["response"]}

    e0 = pd.DataFrame(_rows())
    return (e0,)


@app.cell
def _(mo):
    mo.md(r"""
    ## E0: outcome of every reply

    Share of items per outcome, for every task, mode and prompt. Hover a bar for counts.
    """)
    return


@app.cell
def _(OUTCOMES, OUTCOME_COLORS, PROMPT_ORDER, TASK_ORDER, alt, e0):
    _keys = ["mode", "task", "prompt_kind"]
    _counts = e0.groupby([*_keys, "outcome"]).size().rename("n").reset_index()
    _counts["share"] = _counts["n"] / _counts.groupby(_keys)["n"].transform("sum")
    _counts["rank"] = _counts["outcome"].map(OUTCOMES.index)
    _counts["panel"] = _counts["task"] + ", " + _counts["mode"]
    _panels = [f"{task}, {mode}" for task in TASK_ORDER for mode in ("direct", "reasoning")]

    alt.Chart(_counts).mark_bar(stroke="white", strokeWidth=1).encode(
        x=alt.X("share:Q", stack="zero", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5), title="share of items"),
        y=alt.Y("prompt_kind:N", sort=PROMPT_ORDER, title=None),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=OUTCOMES, range=OUTCOME_COLORS),
                        legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("rank:Q"),
        tooltip=["task", "mode", alt.Tooltip("prompt_kind", title="prompt"), "outcome", "n",
                 alt.Tooltip("share:Q", format=".0%")],
    ).properties(width=260, height=alt.Step(15)).facet(
        facet=alt.Facet("panel:N", sort=[p for p in _panels if p in set(_counts["panel"])], title=None,
                        header=alt.Header(labelFontWeight="bold", labelAnchor="start")),
        columns=2,
    ).resolve_scale(y="independent")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## E0: which answer the model gives

    For the tasks with a fixed label set: the share of each answer given, next to the share of
    each gold label over the same items (*gold*). A prompt whose bar is one colour gets the
    same answer to every item.
    """)
    return


@app.cell
def _(LABEL_COLORS, LABEL_TASKS, NO_ANSWER_COLOR, PROMPT_ORDER, TASK_NAMES, alt, e0, pd):
    def _answer_chart(task, labels):
        rows = e0[e0["task"] == TASK_NAMES[task]]
        given = rows.assign(label=rows["answer"].fillna("no answer"))
        gold = rows[rows["prompt_kind"] == "orig."].assign(prompt_kind="gold", label=lambda d: d["expected"])
        data = pd.concat([given, gold]).groupby(["mode", "prompt_kind", "label"]).size().rename("n").reset_index()
        data["share"] = data["n"] / data.groupby(["mode", "prompt_kind"])["n"].transform("sum")
        domain = [*labels, "no answer"]
        data["rank"] = data["label"].map(domain.index)
        return alt.Chart(data, title=TASK_NAMES[task]).mark_bar(stroke="white", strokeWidth=1).encode(
            x=alt.X("share:Q", stack="zero", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", tickCount=5), title="share of items"),
            y=alt.Y("prompt_kind:N", sort=["gold", *PROMPT_ORDER], title=None),
            color=alt.Color("label:N", scale=alt.Scale(domain=domain,
                                                       range=[*LABEL_COLORS[:len(labels)], NO_ANSWER_COLOR]),
                            legend=alt.Legend(orient="right", title="answer")),
            order=alt.Order("rank:Q"),
            tooltip=["mode", alt.Tooltip("prompt_kind", title="prompt"), alt.Tooltip("label", title="answer"),
                     "n", alt.Tooltip("share:Q", format=".0%")],
        ).properties(width=260, height=alt.Step(15)).facet(
            column=alt.Column("mode:N", sort=["direct", "reasoning"], title=None),
        ).resolve_scale(y="independent")

    alt.vconcat(*[_answer_chart(task, labels) for task, labels in LABEL_TASKS.items()]).resolve_scale(
        color="independent")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## E0: reply length in reasoning mode

    One tick per reply: its length in tokens, coloured by outcome. Replies stop once the
    scored answer is fixed, or at 6,000 tokens.
    """)
    return


@app.cell
def _(OUTCOMES, OUTCOME_COLORS, PROMPT_ORDER, TASK_ORDER, alt, e0):
    _reasoning = e0.loc[e0["mode"] == "reasoning", ["task", "prompt_kind", "item", "outcome", "reply_tokens"]]

    alt.Chart(_reasoning).mark_tick(thickness=2, opacity=0.6).encode(
        x=alt.X("reply_tokens:Q", title="reply length (tokens)"),
        y=alt.Y("prompt_kind:N", sort=PROMPT_ORDER, title=None),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=OUTCOMES, range=OUTCOME_COLORS),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["task", alt.Tooltip("prompt_kind", title="prompt"), "item", "outcome",
                 alt.Tooltip("reply_tokens", title="tokens")],
    ).properties(width=560, height=alt.Step(15)).facet(
        row=alt.Row("task:N", sort=TASK_ORDER, title=None,
                    header=alt.Header(labelAngle=0, labelOrient="top", labelAnchor="start",
                                      labelFontWeight="bold")),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## E0: read the replies

    Pick a mode, task and prompt, then a row of the table: its prompt and reply appear below,
    with the same item's answers under every other prompt.
    """)
    return


@app.cell
def _(mo):
    e0_mode = mo.ui.dropdown(["direct", "reasoning"], value="direct", label="mode")
    e0_mode
    return (e0_mode,)


@app.cell
def _(OUTCOMES, PROMPT_ORDER, TASK_ORDER, e0, e0_mode, mo):
    _rows = e0[e0["mode"] == e0_mode.value]
    _tasks = [t for t in TASK_ORDER if t in set(_rows["task"])]
    e0_task = mo.ui.dropdown(_tasks, value=_tasks[0], label="task")
    e0_prompt = mo.ui.dropdown([p for p in PROMPT_ORDER if p in set(_rows["prompt_kind"])], value="orig.",
                               label="prompt")
    e0_outcomes = mo.ui.multiselect(OUTCOMES, value=OUTCOMES, label="outcome")
    mo.hstack([e0_task, e0_prompt, e0_outcomes], justify="start")
    return e0_outcomes, e0_prompt, e0_task


@app.cell
def _(e0, e0_mode, e0_outcomes, e0_prompt, e0_task, mo):
    e0_shown = e0[(e0["mode"] == e0_mode.value) & (e0["task"] == e0_task.value)
                  & (e0["prompt_kind"] == e0_prompt.value) & e0["outcome"].isin(e0_outcomes.value)]
    e0_table = mo.ui.table(
        e0_shown.assign(reply_start=e0_shown["reply"].str.slice(0, 100))[
            ["item", "expected", "answer", "outcome", "reply_tokens", "reply_start"]],
        selection="single", page_size=8, show_column_summaries=False, label=f"{len(e0_shown)} replies")
    e0_table
    return e0_shown, e0_table


@app.cell
def _(e0, e0_shown, e0_table, html, mo):
    def text_panel(title, text):
        return mo.vstack([mo.md(f"**{title}**"), mo.Html(
            f"<pre style='white-space: pre-wrap; max-height: 28rem; overflow-y: auto; margin: 0; "
            f"font-size: 0.8rem'>{html.escape(text)}</pre>")])

    if e0_shown.empty:
        _view = mo.md("_No reply matches._")
    else:
        _selected = e0_table.value
        _item = _selected.iloc[0]["item"] if len(_selected) else e0_shown.iloc[0]["item"]
        _row = e0_shown[e0_shown["item"] == _item].iloc[0]
        _same = e0[(e0["mode"] == _row["mode"]) & (e0["task"] == _row["task"]) & (e0["item"] == _row["item"])]
        _view = mo.vstack([
            mo.md(f"**Item {_row['item']}**: expected `{_row['expected']}`, answer `{_row['answer']}` "
                  f"({_row['outcome']})"),
            mo.hstack([text_panel("Prompt", _row["prompt"]), text_panel("Reply", _row["reply"])], widths="equal"),
            mo.md("**The same item under every prompt**"),
            mo.ui.table(_same.assign(reply_start=_same["reply"].str.slice(0, 100))[
                ["prompt_kind", "answer", "outcome", "reply_tokens", "reply_start"]],
                selection=None, show_column_summaries=False, pagination=False),
        ])
    _view
    return (text_panel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## E2 causal check: what the patched model answers

    The patched prompts are the prompts without task information of the items answered
    correctly under the original rules prompt (100 per task in direct mode, the first 10 in
    reasoning mode). Each head of a set is set, at the last token (or in reasoning mode,
    optionally at the pre-answer position), to α times its mean output over the task's prompts
    of one style (*task mean*) or over the control task's (*control mean*). *Unpatched*: the
    same prompts without patching.

    The answers are read from `results/answers/`, so causal checks still running appear when
    they finish (rerun this cell).
    """)
    return


@app.cell
def _(RESULTS, fu, given_answer, gzip, json, outcome, pd, task_name):
    PATCH_FAMILIES = {"complex": "direct", "reasoning": "reasoning, last prompt token",
                      "reasoning-pre": "reasoning, pre-answer position"}
    HEAD_SETS = {"logithead": "Logit-head", "jhead": "J-head", "jhead_only": "J only"}
    SOURCES = {"task": "task mean", "control": "control mean"}

    def _rows():
        for path in sorted((RESULTS / "answers").glob("patching_*.jsonl.gz")):
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if r["family"] not in PATCH_FAMILIES:
                        continue
                    answer = given_answer(fu.TASK_LEVELS[r["task"]].task, r["predicted"])
                    yield {"family": PATCH_FAMILIES[r["family"]], "task": task_name(r["task"]),
                           "style": r["style"], "heads": HEAD_SETS.get(r["heads"], ""),
                           "source": SOURCES.get(r["source"], ""),
                           "alpha": "unpatched" if r["source"] == "none" else str(r["alpha"]),
                           "item": str(r["item"]), "expected": str(r["expected"]), "answer": answer,
                           "outcome": outcome(r["correct"], answer), "reply": r["reply"]}

    patched = pd.DataFrame(_rows())
    return HEAD_SETS, PATCH_FAMILIES, SOURCES, patched


@app.cell
def _(PATCH_FAMILIES, mo, patched):
    _families = [f for f in PATCH_FAMILIES.values() if f in set(patched["family"])]
    patch_family = mo.ui.dropdown(_families, value=_families[0], label="mode")
    patch_family
    return (patch_family,)


@app.cell
def _(HEAD_SETS, SOURCES, TASK_ORDER, mo, patch_family, patched):
    _rows = patched[(patched["family"] == patch_family.value) & (patched["alpha"] != "unpatched")]
    _tasks = [t for t in TASK_ORDER if t in set(_rows["task"])]
    _styles = [s for s in ("rule", "example", "combined") if s in set(_rows["style"])]
    patch_task = mo.ui.dropdown(_tasks, value=_tasks[0], label="task")
    patch_style = mo.ui.dropdown(_styles, value=_styles[0], label="style")
    patch_heads = mo.ui.dropdown(list(HEAD_SETS.values()), value="J-head", label="head set")
    patch_source = mo.ui.dropdown(list(SOURCES.values()), value="task mean", label="mean")
    mo.hstack([patch_task, patch_style, patch_heads, patch_source], justify="start")
    return patch_heads, patch_source, patch_style, patch_task


@app.cell
def _(patch_family, patch_heads, patch_source, patch_style, patch_task, patched):
    _task = patched[(patched["family"] == patch_family.value) & (patched["task"] == patch_task.value)]
    _condition = ((_task["style"] == patch_style.value) & (_task["heads"] == patch_heads.value)
                  & (_task["source"] == patch_source.value))
    patch_shown = _task[_condition | (_task["alpha"] == "unpatched")]
    patch_alphas = ["unpatched", *sorted(set(patch_shown["alpha"]) - {"unpatched"}, key=float)]
    return patch_alphas, patch_shown


@app.cell
def _(LABEL_COLORS, LABEL_TASKS, NO_ANSWER_COLOR, OUTCOMES, OUTCOME_COLORS, TASK_NAMES, alt, mo,
      patch_alphas, patch_shown, patch_task):
    def _shares(data, column, domain):
        counts = data.groupby(["alpha", column]).size().rename("n").reset_index()
        counts["share"] = counts["n"] / counts.groupby("alpha")["n"].transform("sum")
        counts["rank"] = counts[column].map(domain.index)
        return counts

    def _bars(data, column, domain, colors, title):
        return alt.Chart(_shares(data, column, domain), title=title).mark_bar(stroke="white", strokeWidth=1).encode(
            x=alt.X("alpha:N", sort=patch_alphas, title="α", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("share:Q", stack="zero", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", tickCount=5), title="share of patched items"),
            color=alt.Color(f"{column}:N", scale=alt.Scale(domain=domain, range=colors),
                            legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("rank:Q"),
            tooltip=[alt.Tooltip("alpha", title="α"), column, "n", alt.Tooltip("share:Q", format=".0%")],
        ).properties(width=360, height=220)

    _charts = [_bars(patch_shown, "outcome", OUTCOMES, OUTCOME_COLORS, "Outcome")]
    _labels = next((l for t, l in LABEL_TASKS.items() if TASK_NAMES[t] == patch_task.value), None)
    if _labels:
        _given = patch_shown.assign(label=patch_shown["answer"].fillna("no answer"))
        _charts.append(_bars(_given, "label", [*_labels, "no answer"],
                             [*LABEL_COLORS[:len(_labels)], NO_ANSWER_COLOR], "Answer given"))
    _gold = patch_shown[patch_shown["alpha"] == "unpatched"]["expected"].value_counts()
    mo.vstack([
        alt.hconcat(*_charts).resolve_scale(color="independent"),
        mo.md(f"Gold labels of the {_gold.sum()} patched items: "
              + ", ".join(f"{label} {n}" for label, n in _gold.items()) if _labels else ""),
    ])
    return


@app.cell
def _(mo, patch_alphas):
    patch_alpha = mo.ui.dropdown(patch_alphas, value="unpatched" if "1" not in patch_alphas else "1", label="α")
    patch_alpha
    return (patch_alpha,)


@app.cell
def _(mo, patch_alpha, patch_shown):
    patch_rows = patch_shown[patch_shown["alpha"] == patch_alpha.value]
    patch_table = mo.ui.table(
        patch_rows.assign(reply_start=patch_rows["reply"].str.slice(0, 100))[
            ["item", "expected", "answer", "outcome", "reply_start"]],
        selection="single", page_size=8, show_column_summaries=False, label=f"{len(patch_rows)} replies")
    patch_table
    return patch_rows, patch_table


@app.cell
def _(e0, mo, patch_family, patch_rows, patch_table, text_panel):
    if patch_rows.empty:
        _view = mo.md("_No reply._")
    else:
        _item = patch_table.value.iloc[0]["item"] if len(patch_table.value) else patch_rows.iloc[0]["item"]
        _row = patch_rows[patch_rows["item"] == _item].iloc[0]
        _mode = "direct" if patch_family.value == "direct" else "reasoning"
        _prompt = e0[(e0["mode"] == _mode) & (e0["task"] == _row["task"]) & (e0["prompt_kind"] == "no task info")
                     & (e0["item"] == _row["item"])]["prompt"]
        _view = mo.vstack([
            mo.md(f"**Item {_row['item']}**, α = {_row['alpha']}: expected `{_row['expected']}`, "
                  f"answer `{_row['answer']}` ({_row['outcome']})"),
            mo.hstack([text_panel("Prompt (without task information)", _prompt.iloc[0] if len(_prompt) else ""),
                       text_panel("Reply", _row["reply"])], widths="equal"),
        ])
    _view
    return


if __name__ == "__main__":
    app.run()
