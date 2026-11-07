"""E5 (paper/proposal-llm-heads.tex): GPT-5.5 scores every head of the tables of ``llm_heads_tables.py``.

Requests, per task, choosing half (A: fold 1, B: fold 2) and readout:
- batched: 8 requests, the prompts ``*_batch0`` ... ``*_batch7`` (128 heads each);
- all: 1 request, the prompt ``*_all`` (1,024 heads);
- with ``--repeat``: the ``all`` request a second time, same prompt (stability; ``repeat`` 1).
In this order, so that ``--limit 1`` sends one batch request. Reasoning effort medium.

API: the Responses API with a detailed reasoning summary, so that the summary is kept. If the
resource refuses it (HTTP 400 or 404) on the first request, chat completions for that request
and every later one; the refusal is in the log. ``--api`` fixes the choice.

Every output is recorded in ``data/llm_heads/calls.jsonl.gz`` (gzip members appended, one JSON
line each): an ``http`` line for every HTTP attempt (the request's key, attempt, API, settings,
prompt file and SHA-256, the retry note if any, status, the whole response body), and a
``check`` line for every reply (valid or not, the error, token usage, cost, the scores read).
The endpoint and key are never written. A request whose key and prompt SHA-256 already have a
valid ``check`` line is not sent again.

Check of a reply (prompt version 2): JSON (a code fence is allowed), ``scores`` {id: score} with
a whole number from 0 to 100 for every head of the table, ``answer`` and ``input`` lists of ids
of the table, ``notes`` {id: {tokens, why}} for every score of 50 or more. A reply that fails is
asked again at most twice, with the error appended to the prompt. A head scored 50 or more
without its note is a warning (logged), not a reason to ask again; ``--recheck`` applies this to
replies already in the log (user, 2026-11-06), sending nothing. Version 1 asked for one object
per head; its all-heads replies listed only some 30 heads, and its lines stay in the log.

Writes ``data/llm_heads/scores/{task}_{half}_{readout}_{all|batched}[_r1].json`` once all the
requests of a condition are valid: per head id, layer, head, score, flags, tokens, why, and the
answer and input shares of the table.

    . ~/.azure_openai_key && .venv/bin/python scripts/llm_heads_select.py --tasks english-german --limit 1
    . ~/.azure_openai_key && .venv/bin/python scripts/llm_heads_select.py --tasks english-german --repeat
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, READOUTS, azure  # noqa: E402

OUT = DATA / "llm_heads"
LOG = OUT / "calls.jsonl.gz"
SCORES = OUT / "scores"
EFFORT, MAX_OUTPUT, RETRIES, WORKERS, N_BATCHES, BATCH = "medium", 128_000, 2, 4, 8, 128
PROMPT_VERSION = 2  # 1: one object per head (the all-heads replies listed only some heads)
PRICE = {"input": 5.0, "cached": 0.5, "output": 30.0}  # $ per million tokens (user, 2026-10-30)
LOCK = threading.Lock()


@dataclass(frozen=True)
class Request:
    task: str
    half: str
    readout: str
    condition: str  # "all" or "batched"
    batch: int | None
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.task}|{self.half}|{self.readout}|{self.condition}|{self.batch}|{self.repeat}"

    @property
    def prompt_file(self) -> Path:
        name = "all" if self.condition == "all" else f"batch{self.batch}"
        return OUT / "prompts" / f"{self.task}_{self.half}_{self.readout}_{name}.txt.gz"

    def ids(self, table: dict) -> list[str]:
        ids = list(table["rows"])
        return ids if self.batch is None else ids[self.batch * BATCH:(self.batch + 1) * BATCH]


def append_log(entry: dict) -> None:
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with LOCK, gzip.open(LOG, "at", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log() -> list[dict]:
    if not LOG.exists():
        return []
    with gzip.open(LOG, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def body(api: str, prompt: str) -> dict:
    if api == "responses":
        return {"model": azure.DEPLOYMENT, "input": prompt, "max_output_tokens": MAX_OUTPUT,
                "reasoning": {"effort": EFFORT, "summary": "detailed"}}
    return {"messages": [{"role": "user", "content": prompt}], "reasoning_effort": EFFORT,
            "max_completion_tokens": MAX_OUTPUT}


def extract(api: str, data: dict) -> tuple[str, list[str], dict, str | None]:
    """(reply text, reasoning summary, usage, why the reply is incomplete or None)."""
    if api == "responses":
        text, summary = [], []
        for item in data.get("output", []):
            if item.get("type") == "message":
                text += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
            elif item.get("type") == "reasoning":
                summary += [s.get("text", "") for s in item.get("summary", [])]
        u = data.get("usage", {})
        usage = {"input": u.get("input_tokens"), "cached": (u.get("input_tokens_details") or {}).get("cached_tokens"),
                 "output": u.get("output_tokens"),
                 "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens")}
        incomplete = None if data.get("status") == "completed" else str(data.get("incomplete_details") or data.get("status"))
        return "".join(text), summary, usage, incomplete
    choice = data["choices"][0]
    u = data.get("usage", {})
    usage = {"input": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
             "output": u.get("completion_tokens"),
             "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
    incomplete = None if choice.get("finish_reason") == "stop" else f"finish_reason {choice.get('finish_reason')}"
    return choice["message"].get("content") or "", [], usage, incomplete


def cost(usage: dict) -> float:
    cached = usage.get("cached") or 0
    return ((usage.get("input", 0) - cached) * PRICE["input"] + cached * PRICE["cached"]
            + usage.get("output", 0) * PRICE["output"]) / 1e6


def check(text: str, ids: list[str]) -> tuple[dict | None, str | None, list[str]]:
    """(scores by head id ``{"id", "score", "answer", "input", "tokens", "why"}`` or None, the
    reason the reply cannot be used or None, warnings). A head scored 50 or more without its note
    is a warning, not a reason to reject the reply: the sets use only the scores. Reply format of prompt version 2: ``scores`` {id: score},
    ``answer`` and ``input`` lists of ids, ``notes`` {id: {tokens, why}}."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    try:
        value = json.loads(fenced.group(1) if fenced else text)
    except json.JSONDecodeError as error:
        return None, f"The reply is not valid JSON ({error}).", []
    if not isinstance(value, dict) or not isinstance(value.get("scores"), dict):
        return None, 'The reply is not a JSON object with an object "scores".', []
    wanted, problems, warnings = set(ids), [], []
    given = value["scores"]
    extra = [h for h in given if h not in wanted]
    if extra:
        problems.append(f"{len(extra)} ids in scores are not heads of the table (first: {', '.join(extra[:5])})")
    for hid, score in given.items():
        if hid in wanted and (isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100):
            problems.append(f"{hid}: score {score!r} is not a whole number from 0 to 100")
    marks = {}
    for name in ("answer", "input"):
        listed = value.get(name, [])
        if not isinstance(listed, list) or not all(isinstance(h, str) for h in listed):
            problems.append(f'"{name}" is not a list of head ids')
            listed = []
        bad = [h for h in listed if h not in wanted]
        if bad:
            problems.append(f'{len(bad)} ids in "{name}" are not heads of the table (first: {", ".join(bad[:5])})')
        marks[name] = set(listed)
    notes = value.get("notes", {})
    if not isinstance(notes, dict):
        problems.append('"notes" is not an object')
        notes = {}
    for hid, score in given.items():
        if hid in wanted and isinstance(score, int) and score >= 50:
            note = notes.get(hid)
            if not (isinstance(note, dict) and isinstance(note.get("tokens"), list)
                    and all(isinstance(t, str) for t in note["tokens"])
                    and isinstance(note.get("why"), str) and note["why"].strip()):
                warnings.append(f"{hid}: score {score} without a note (tokens, why)")
    missing = [h for h in ids if h not in given]
    if missing:
        problems.append(f"{len(missing)} heads of the table have no score (first: {', '.join(missing[:10])})")
    if problems:
        return None, "Problems: " + "; ".join(problems[:20]) + (" ..." if len(problems) > 20 else "") + ".", warnings
    return {h: {"id": h, "score": given[h], "answer": h in marks["answer"], "input": h in marks["input"],
                "tokens": (notes.get(h) or {}).get("tokens"), "why": (notes.get(h) or {}).get("why")}
            for h in ids}, None, warnings


class Runner:
    def __init__(self, api: str | None, dry: bool):
        self.api, self.dry = api, dry
        self.client, self.chat_url = azure.client_and_url() if not dry else (None, None)
        if self.client is not None:
            self.client.timeout = 3600

    def url(self, api: str) -> str:
        return azure.responses_url() if api == "responses" else self.chat_url

    def run(self, req: Request, table: dict) -> dict | None:
        base = gzip.open(req.prompt_file, "rt", encoding="utf-8").read()
        sha = hashlib.sha256(base.encode()).hexdigest()
        common = {"key": req.key, "task": req.task, "half": req.half, "fold": 1 if req.half == "A" else 2,
                  "readout": req.readout, "condition": req.condition, "batch": req.batch, "repeat": req.repeat,
                  "prompt_file": str(req.prompt_file.relative_to(DATA.parent)), "prompt_sha256": sha,
                  "prompt_version": PROMPT_VERSION}
        ids, note, attempt = req.ids(table), "", 0
        while attempt <= RETRIES:
            api = self.api or "responses"
            settings = {k: v for k, v in body(api, "").items() if k not in ("input", "messages")}
            start = time.time()

            def log_http(entry, api=api, attempt=attempt, settings=settings):
                append_log({"event": "http", **common, "attempt": attempt, "api": api,
                            "api_version": azure.RESPONSES_API_VERSION if api == "responses" else azure.API_VERSION,
                            "deployment": azure.DEPLOYMENT, "settings": settings, "retry_note": note, **entry})

            try:
                data = azure.post_logged(self.client, self.url(api), body(api, base + note), log_http)
            except azure.RequestError as error:
                if self.api is None and api == "responses" and error.status in (400, 404):
                    print(f"{req.key}: Responses API refused ({error}); chat completions from now on", flush=True)
                    append_log({"event": "api", "key": req.key, "choice": "chat", "reason": str(error)})
                    self.api = "chat"
                    continue
                append_log({"event": "failed", **common, "attempt": attempt, "api": api, "error": str(error)})
                print(f"{req.key}: FAILED ({error})", flush=True)
                return None
            except Exception as error:  # network errors after every retry
                append_log({"event": "failed", **common, "attempt": attempt, "api": api,
                            "error": f"{type(error).__name__}: {error}"})
                print(f"{req.key}: FAILED ({type(error).__name__}: {error})", flush=True)
                return None
            self.api = api
            text, summary, usage, incomplete = extract(api, data)
            scores, error, warnings = ((None, f"The reply was cut off ({incomplete}).", []) if incomplete
                                       else check(text, ids))
            spent = cost(usage)
            append_log({"event": "check", **common, "attempt": attempt, "api": api, "valid": error is None,
                        "error": error, "warnings": warnings, "usage": usage, "cost_usd": round(spent, 4),
                        "seconds": round(time.time() - start, 1), "model": data.get("model"),
                        "response_id": data.get("id"), "reasoning_summary": summary, "scores": scores})
            print(f"{req.key} attempt {attempt}: {'valid' if error is None else 'INVALID'}; tokens in "
                  f"{usage.get('input')}, out {usage.get('output')} (reasoning {usage.get('reasoning')}); "
                  f"${spent:.2f}; {time.time() - start:.0f} s; summary {len(summary)} parts"
                  + (f"; {error[:200]}" if error else ""), flush=True)
            if error is None:
                return scores
            note = (f"\n\n# Your previous reply could not be used\n\n{error}\n"
                    "Reply again with the whole answer, in the answer format above.")
            attempt += 1
        return None


def recheck(requests: list[Request], tables: dict) -> None:
    """Replies already in the log, checked again: for a request with no valid reply, the first
    logged reply to its current prompt that passes ``check`` gets a new ``check`` line
    (``recheck_of_attempt``; cost 0, since its cost is on the original line). Sends nothing."""
    lines = read_log()
    valid = {(l["key"], l["prompt_sha256"]) for l in lines if l.get("event") == "check" and l.get("valid")}
    for req in requests:
        base = gzip.open(req.prompt_file, "rt", encoding="utf-8").read()
        sha = hashlib.sha256(base.encode()).hexdigest()
        if (req.key, sha) in valid:
            continue
        replies = [l for l in lines if l.get("event") == "http" and l["key"] == req.key
                   and l["prompt_sha256"] == sha and l.get("status") == 200]
        for l in sorted(replies, key=lambda l: l["attempt"]):
            data = json.loads(l["body"])
            text, summary, usage, incomplete = extract(l["api"], data)
            if incomplete:
                continue
            scores, error, warnings = check(text, req.ids(tables[(req.task, req.half, req.readout)]))
            if error is None:
                append_log({"event": "check", "key": req.key, "task": req.task, "half": req.half,
                            "fold": 1 if req.half == "A" else 2, "readout": req.readout,
                            "condition": req.condition, "batch": req.batch, "repeat": req.repeat,
                            "prompt_file": l["prompt_file"], "prompt_sha256": sha,
                            "prompt_version": PROMPT_VERSION, "attempt": l["attempt"], "api": l["api"],
                            "valid": True, "error": None, "warnings": warnings, "recheck_of_attempt": l["attempt"],
                            "usage": usage, "cost_usd": 0.0, "model": data.get("model"),
                            "response_id": data.get("id"), "reasoning_summary": summary, "scores": scores})
                print(f"{req.key}: logged reply of attempt {l['attempt']} valid under the relaxed check; "
                      f"warnings: {warnings}", flush=True)
                break
        else:
            print(f"{req.key}: no logged reply passes the check", flush=True)


def write_scores(task: str, half: str, readout: str, condition: str, repeat: int, table: dict,
                 parts: list[dict]) -> None:
    merged = {k: v for part in parts for k, v in part.items()}
    assert list(merged) == list(table["rows"]) or set(merged) == set(table["rows"]), "not every head scored"
    heads = []
    for hid, row in table["rows"].items():
        s = merged[hid]
        heads.append({"id": hid, "layer": row["layer"], "head": row["head"], "score": s["score"],
                      "answer": s["answer"], "input": s["input"], "tokens": s.get("tokens"), "why": s.get("why"),
                      "answer_share": row["answer_share"], "input_share": row["input_share"]})
    SCORES.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_{half}_{readout}_{condition}" + (f"_r{repeat}" if repeat else "")
    (SCORES / f"{stem}.json").write_text(json.dumps({"task": task, "half": half, "readout": readout,
                                                       "condition": condition, "repeat": repeat, "heads": heads},
                                                      ensure_ascii=False, indent=0))
    print(f"wrote scores/{stem}.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--halves", nargs="+", default=["A", "B"])
    parser.add_argument("--readouts", nargs="+", default=list(READOUTS))
    parser.add_argument("--conditions", nargs="+", default=["batched", "all"])
    parser.add_argument("--repeat", action="store_true", help="a second request of every all prompt")
    parser.add_argument("--limit", type=int, default=None, help="send at most this many new requests")
    parser.add_argument("--api", choices=("responses", "chat"), default=None)
    parser.add_argument("--dry-run", action="store_true", help="list and check the requests, send nothing")
    parser.add_argument("--recheck", action="store_true",
                        help="check the logged replies again (missing notes are warnings), write scores, send nothing")
    args = parser.parse_args()

    requests = []
    for task in args.tasks:
        for half in args.halves:
            for readout in args.readouts:
                if "batched" in args.conditions:
                    requests += [Request(task, half, readout, "batched", b, 0) for b in range(N_BATCHES)]
                if "all" in args.conditions:
                    requests += [Request(task, half, readout, "all", None, r) for r in range(1 + args.repeat)]
    tables = {}
    for req in requests:
        tkey = (req.task, req.half, req.readout)
        if tkey not in tables:
            tables[tkey] = json.loads((OUT / "tables" / f"{req.task}_{req.half}_{req.readout}.json").read_text())
        assert req.prompt_file.exists(), req.prompt_file

    if args.recheck:
        recheck(requests, tables)
    done = {}
    for line in read_log():
        if line.get("event") == "check" and line.get("valid"):
            done[(line["key"], line["prompt_sha256"])] = line["scores"]

    def sha(req):
        return hashlib.sha256(gzip.open(req.prompt_file, "rt", encoding="utf-8").read().encode()).hexdigest()

    todo = [r for r in requests if (r.key, sha(r)) not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    if args.recheck:
        todo = []
    print(f"{len(requests)} requests, {len(requests) - len([r for r in requests if (r.key, sha(r)) not in done])} "
          f"already valid, sending {len(todo)}", flush=True)
    if args.dry_run:
        for r in todo:
            print("  would send", r.key, r.prompt_file.name)
        return

    runner = Runner(args.api, dry=False)
    results = {}
    if todo:  # the first request alone, so that the API is settled before the others
        results[todo[0]] = runner.run(todo[0], tables[(todo[0].task, todo[0].half, todo[0].readout)])
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {r: pool.submit(runner.run, r, tables[(r.task, r.half, r.readout)]) for r in todo[1:]}
        results.update({r: f.result() for r, f in futures.items()})
    for (key, _), scores in done.items():
        for r in requests:
            if r.key == key:
                results.setdefault(r, scores)

    for task in args.tasks:
        for half in args.halves:
            for readout in args.readouts:
                table = tables[(task, half, readout)]
                groups = {}
                for r in requests:
                    if (r.task, r.half, r.readout) == (task, half, readout):
                        groups.setdefault((r.condition, r.repeat), []).append(r)
                for (condition, repeat), reqs in groups.items():
                    parts = [results.get(r) for r in reqs]
                    if all(p is not None for p in parts):
                        write_scores(task, half, readout, condition, repeat, table, parts)


if __name__ == "__main__":
    main()
