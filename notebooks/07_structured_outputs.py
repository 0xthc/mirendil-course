import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 07 · Structured outputs — getting JSON your code can trust

        A language model loves to *talk*. But your program doesn't want a paragraph —
        it wants a **record**: a `name`, an `age`, a list of `tags`, in exactly that
        shape, every time. This notebook builds the bridge between "model says words"
        and "my code gets a typed object it can rely on."

        **What you'll build, step by step:**

        1. A target **schema** as a `dataclass`, plus a `validate(obj)` function that
           checks types and required fields.
        2. The **naive** approach — "just ask for JSON and `json.loads` it" — and watch
           it fail two different ways: *malformed JSON* and *schema violations*.
        3. The **robust** approach — a parse → validate → **corrective retry** loop that
           feeds the error back to the model and tries again, up to `N` attempts.
        4. A note on the **production-grade** techniques (tool/function calling and
           JSON-schema-constrained decoding) that make this airtight.
        5. **Charts**: success rate vs. allowed retries, and a breakdown of which
           failure types the loop caught.
        6. An **interactive** retry slider that re-simulates many runs and redraws the
           success-rate curve live.

        You are a full-stack engineer, not a data scientist. So we treat the model as a
        flaky API that returns strings, and we focus on the **plumbing** that turns
        those strings into trustworthy data.
        """
    )
    return


@app.cell
def _():
    import json
    import re
    from dataclasses import dataclass, fields

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    # A consistent palette across every chart in this notebook.
    BLUE = "#57b6f5"
    ORANGE = "#f0986b"
    GREEN = "#5fd38a"
    PURPLE = "#c099f0"
    return BLUE, GREEN, ORANGE, PURPLE, dataclass, fields, json, mo, np, plt, re


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** Imagine asking a brilliant but chatty intern to "tell me about
            this customer." They might write you a lovely three-sentence story. Lovely
            — and useless to a computer. What you actually wanted was a **form**:

            > Name: ▢▢▢▢   Age: ▢▢   Tags: ▢▢▢, ▢▢▢

            Structured output is the art of getting the intern to **fill out the form**
            instead of writing prose — and then *checking* that every box is filled in
            correctly before you let the rest of your program trust it.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1 · The target: a schema your code can trust

        Our running task: read a sentence about a person and extract a clean record.

        > "Maya is 31 and loves hiking, climbing, and coffee."

        should become

        ```json
        {"name": "Maya", "age": 31, "tags": ["hiking", "climbing", "coffee"]}
        ```

        We describe that shape **once**, as a `dataclass`. A dataclass is just a typed
        struct — it documents the fields and their types in one place, which is exactly
        what we'll validate against.
        """
    )
    return


@app.cell
def _(dataclass):
    @dataclass
    class Person:
        name: str          # required, must be a string
        age: int           # required, must be a whole number
        tags: list         # required, must be a list of strings

    # The fields we require, with the python type each must be.
    SCHEMA = {"name": str, "age": int, "tags": list}
    return Person, SCHEMA


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The validator

        Parsing JSON only tells you the text was *valid JSON*. It says nothing about
        whether the object has the **right fields and types**. `{"naem": "Maya"}` is
        perfectly good JSON and completely wrong for us.

        So `validate(obj)` returns `(ok, errors)` — a boolean plus a list of
        human-readable problems. We check, in order:

        - the value is a JSON **object** (a `dict`), not a list/number/string,
        - every **required field is present**,
        - every field has the **right type** (with one classic gotcha: in Python
          `bool` is a subclass of `int`, so we reject `True` as an age on purpose),
        - `tags` is a list whose elements are all strings.
        """
    )
    return


@app.cell
def _(SCHEMA):
    def validate(obj):
        "Return (ok, errors). errors is a list of human-readable strings."
        errors = []

        if not isinstance(obj, dict):
            return False, [f"top-level value is {type(obj).__name__}, expected object"]

        for field_name, expected in SCHEMA.items():
            if field_name not in obj:
                errors.append(f"missing required field '{field_name}'")
                continue

            value = obj[field_name]
            # bool is a subclass of int in Python — reject it for numeric fields.
            if expected is int and isinstance(value, bool):
                errors.append(f"field '{field_name}' is a bool, expected int")
            elif not isinstance(value, expected):
                errors.append(
                    f"field '{field_name}' is {type(value).__name__}, "
                    f"expected {expected.__name__}"
                )

        # extra structural check: tags must be a list of strings
        tags = obj.get("tags")
        if isinstance(tags, list) and not all(isinstance(t, str) for t in tags):
            errors.append("field 'tags' must contain only strings")

        return (len(errors) == 0), errors
    return (validate,)


@app.cell
def _(mo, validate):
    # Quick sanity check of the validator on a few hand-made objects.
    _cases = {
        "good": {"name": "Maya", "age": 31, "tags": ["hiking"]},
        "missing age": {"name": "Maya", "tags": ["hiking"]},
        "age as string": {"name": "Maya", "age": "31", "tags": ["hiking"]},
        "tags not a list": {"name": "Maya", "age": 31, "tags": "hiking"},
        "age is a bool": {"name": "Maya", "age": True, "tags": ["hiking"]},
    }
    _rows = []
    for _label, _obj in _cases.items():
        _ok, _errs = validate(_obj)
        _mark = "✅" if _ok else "❌"
        _rows.append(f"| {_label} | {_mark} | {', '.join(_errs) or '—'} |")

    mo.md(
        "**Validator self-test**\n\n"
        "| case | ok? | errors |\n|---|---|---|\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2 · A stand-in model (no network, fully offline)

        We can't call a real API in this notebook, so `mock_llm` is a **stand-in**: a
        plain function that returns a JSON *string*, just like a real chat endpoint
        would hand you `response.choices[0].message.content`.

        To teach the failure modes, the mock is deliberately unreliable on its **early
        attempts** and is *rule-based and deterministic* (no random seeds to wobble):

        - **attempt 0** → returns JSON with a **trailing comma** → `json.loads` throws.
        - **attempt 1** → returns valid JSON but `age` is the **string** `"31"` → a
          schema violation.
        - **attempt 2+** → returns the clean, correct record.

        This mimics what real models actually do: a plain prompt fails sometimes, but
        when you hand the error back and ask again, the next attempt usually fixes it.
        """
    )
    return


@app.cell
def _():
    def mock_llm(prompt, attempt, feedback=None):
        """A deterministic stand-in for a chat model. Returns a JSON *string*.

        `attempt` is the 0-based retry index. The mock degrades on early attempts so
        we can exercise each failure path, then 'corrects' once it has been nudged.
        This is NOT a real model — it ignores `prompt`/`feedback` content and is
        purely a teaching fixture.
        """
        if attempt == 0:
            # Malformed: trailing comma after the last element -> JSONDecodeError.
            return '{"name": "Maya", "age": 31, "tags": ["hiking", "climbing", "coffee"],}'
        if attempt == 1:
            # Valid JSON, but age is a string -> schema violation.
            return '{"name": "Maya", "age": "31", "tags": ["hiking", "climbing", "coffee"]}'
        # Corrected, clean record.
        return '{"name": "Maya", "age": 31, "tags": ["hiking", "climbing", "coffee"]}'
    return (mock_llm,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3 · The naive approach — and why it bites you

        The tempting one-liner: *"ask for JSON, then `json.loads` it."* Let's run it
        against the mock's first two attempts and watch both failure modes happen.

        Notice we have to guard **two separate gates**:

        1. `json.loads` can raise `JSONDecodeError` — the text isn't even JSON.
        2. Even if it parses, `validate` can reject it — the JSON is the wrong *shape*.
        """
    )
    return


@app.cell
def _(json, mock_llm, validate):
    def naive_extract(sentence, attempt):
        "Parse once, validate once, no recovery. Returns a result dict for display."
        raw = mock_llm(sentence, attempt)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            return {"raw": raw, "outcome": "malformed", "detail": str(e), "obj": None}
        ok, errors = validate(obj)
        if not ok:
            return {"raw": raw, "outcome": "schema", "detail": "; ".join(errors), "obj": obj}
        return {"raw": raw, "outcome": "ok", "detail": "valid!", "obj": obj}

    _sentence = "Maya is 31 and loves hiking, climbing, and coffee."
    naive_results = [naive_extract(_sentence, a) for a in (0, 1)]
    return (naive_results,)


@app.cell
def _(mo, naive_results):
    _labels = {
        "malformed": "❌ malformed JSON (json.loads raised)",
        "schema": "❌ schema violation (parsed, but wrong shape)",
        "ok": "✅ valid",
    }
    _blocks = []
    for _i, _r in enumerate(naive_results):
        _blocks.append(
            f"**Naive attempt {_i}** — {_labels[_r['outcome']]}\n\n"
            f"model returned:\n\n```json\n{_r['raw']}\n```\n\n"
            f"problem: `{_r['detail']}`"
        )
    mo.md(
        "### What the naive approach gets\n\n"
        + "\n\n---\n\n".join(_blocks)
        + "\n\n> Two calls, two different failures, **zero usable records**. "
        "A naive parser would have crashed or silently passed bad data downstream."
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4 · The robust approach — parse, validate, correct, retry

        The fix is a loop. On each attempt we:

        1. **call** the model,
        2. **parse** — on `JSONDecodeError`, record the error and retry,
        3. **validate** — on schema errors, record them and retry,
        4. on success, **return** the clean object,
        5. and crucially, we **feed the error back** into the next call (`feedback=`),
           so the model knows what to fix — a *corrective* retry, not a blind one.

        If we exhaust the budget, we **give up gracefully** — returning failure plus the
        full transcript — instead of throwing deep inside your request handler.
        """
    )
    return


@app.cell
def _(json, validate):
    def robust_extract(sentence, max_retries, responder):
        """Loop: call -> parse -> validate -> corrective retry. Total attempts =
        max_retries + 1. Returns (success, obj, transcript)."""
        transcript = []
        feedback = None
        for attempt in range(max_retries + 1):
            raw = responder(sentence, attempt, feedback)
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                feedback = f"Your output was not valid JSON ({e}). Return ONLY valid JSON."
                transcript.append({"attempt": attempt, "raw": raw,
                                    "outcome": "malformed", "detail": str(e)})
                continue
            ok, errors = validate(obj)
            if ok:
                transcript.append({"attempt": attempt, "raw": raw,
                                   "outcome": "ok", "detail": "valid"})
                return True, obj, transcript
            feedback = "Your JSON had schema errors: " + "; ".join(errors) + ". Fix them."
            transcript.append({"attempt": attempt, "raw": raw,
                               "outcome": "schema", "detail": "; ".join(errors)})
        return False, None, transcript
    return (robust_extract,)


@app.cell
def _(mock_llm, robust_extract):
    _sentence = "Maya is 31 and loves hiking, climbing, and coffee."
    demo_success, demo_obj, demo_transcript = robust_extract(_sentence, 3, mock_llm)
    return demo_obj, demo_success, demo_transcript


@app.cell
def _(demo_obj, demo_success, demo_transcript, mo):
    _icon = {"malformed": "❌", "schema": "❌", "ok": "✅"}
    _rows = []
    for _t in demo_transcript:
        _rows.append(
            f"| {_t['attempt']} | {_icon[_t['outcome']]} {_t['outcome']} "
            f"| `{_t['detail'][:60]}` |"
        )
    _verdict = (
        f"✅ **Succeeded** after {len(demo_transcript)} attempt(s). "
        f"Final object: `{demo_obj}`"
        if demo_success
        else "❌ **Gave up** after exhausting the retry budget."
    )
    mo.callout(
        mo.md(
            "### Transcript of attempts\n\n"
            "| attempt | outcome | detail |\n|---|---|---|\n"
            + "\n".join(_rows)
            + "\n\n"
            + _verdict
        ),
        kind="success" if demo_success else "danger",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **The production-grade way (do this when you can).** The retry loop is the
            reliable *fallback*, but modern model APIs let you stop most failures
            before they happen:

            - **Tool / function calling** — you hand the model a JSON Schema for the
              fields you want; the API returns the arguments already shaped to it.
            - **Constrained / structured decoding** ("JSON mode", grammar-constrained
              sampling) — the decoder is *physically prevented* from emitting tokens
              that would break the schema, so malformed JSON becomes impossible.

            Even with these, keep `validate()` and a retry or two: schemas constrain
            *shape*, not *meaning* (a constrained model can still put the age in the
            name field). Belt **and** suspenders.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5 · How well does retrying actually work?

        To chart this we need many runs, not one. So we simulate a fleet of extraction
        jobs where each job needs a different number of corrective attempts before it
        succeeds — some get it right immediately, some are stubborn. The distribution
        below is fixed with a **seeded RNG**, so the numbers are stable every run.

        `n_failures = k` means that job emits `k` bad responses (alternating malformed
        / schema) and then a good one. A run *succeeds* when its `n_failures` fits
        inside the retry budget.
        """
    )
    return


@app.cell
def _(np):
    # Deterministic fleet of 2000 jobs. Each value = how many bad responses that job
    # emits before producing a valid one. Seeded so every export is identical.
    _rng = np.random.default_rng(7)
    N_JOBS = 2000
    failure_counts = _rng.choice(
        [0, 1, 2, 3, 4],
        size=N_JOBS,
        p=[0.45, 0.30, 0.15, 0.07, 0.03],  # most jobs are easy; a few are stubborn
    )

    def simulate(counts, max_retries):
        """Given the fleet and a retry budget, return (success_rate, n_malformed,
        n_schema) where the counts are failures *caught* by the loop."""
        total_attempts_allowed = max_retries + 1
        successes = 0
        malformed = 0
        schema = 0
        for f in counts:
            caught = f if f < total_attempts_allowed else total_attempts_allowed
            if f < total_attempts_allowed:
                successes += 1
            for a in range(caught):
                # mock alternates: even attempts malformed, odd attempts schema
                if a % 2 == 0:
                    malformed += 1
                else:
                    schema += 1
        return successes / len(counts), malformed, schema
    return N_JOBS, failure_counts, simulate


@app.cell
def _(BLUE, GREEN, N_JOBS, failure_counts, plt, simulate):
    _retries = list(range(0, 7))
    _rates = [simulate(failure_counts, r)[0] * 100 for r in _retries]

    fig_rate, ax_rate = plt.subplots(figsize=(7, 4))
    _bars = ax_rate.bar(_retries, _rates, color=BLUE, edgecolor="white")
    ax_rate.plot(_retries, _rates, color=GREEN, marker="o", linewidth=2, zorder=3)
    for _x, _y in zip(_retries, _rates):
        ax_rate.text(_x, _y + 1.5, f"{_y:.0f}%", ha="center", fontsize=9)
    ax_rate.set_title(f"Success rate vs. allowed retries (fleet of {N_JOBS} jobs)")
    ax_rate.set_xlabel("max_retries (extra attempts after the first call)")
    ax_rate.set_ylabel("jobs that produced a valid record (%)")
    ax_rate.set_ylim(0, 108)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)
    ax_rate
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The curve climbs fast and then flattens: the first couple of retries rescue the
        bulk of jobs, after which you hit diminishing returns (and each extra attempt
        costs latency and tokens). That knee is exactly the tradeoff you tune in prod.

        Now the **second chart**: of all the failures the loop *caught* (at a generous
        budget), how many were malformed JSON vs. schema violations?
        """
    )
    return


@app.cell
def _(ORANGE, PURPLE, failure_counts, plt, simulate):
    _, _malformed, _schema = simulate(failure_counts, 6)

    fig_break, ax_break = plt.subplots(figsize=(6, 4))
    _cats = ["malformed JSON\n(json.loads raised)", "schema violation\n(wrong shape)"]
    _vals = [_malformed, _schema]
    _bars = ax_break.bar(_cats, _vals, color=[ORANGE, PURPLE], edgecolor="white")
    for _b, _v in zip(_bars, _vals):
        ax_break.text(_b.get_x() + _b.get_width() / 2, _v + max(_vals) * 0.01,
                      str(_v), ha="center", fontsize=10)
    ax_break.set_title("Failure types caught by the retry loop")
    ax_break.set_ylabel("count across the fleet")
    ax_break.spines["top"].set_visible(False)
    ax_break.spines["right"].set_visible(False)
    ax_break
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6 · Interactive: tune the retry budget

        Drag the slider to set `max_retries`. The chart re-simulates the whole fleet
        and redraws the success-rate curve, highlighting your chosen budget. Watch how
        quickly the gains flatten out.
        """
    )
    return


@app.cell
def _(mo):
    retry_slider = mo.ui.slider(0, 6, value=2, label="max_retries", show_value=True)
    retry_slider
    return (retry_slider,)


@app.cell
def _(BLUE, GREEN, ORANGE, failure_counts, mo, plt, retry_slider, simulate):
    _chosen = retry_slider.value
    _retries = list(range(0, 7))
    _rates = [simulate(failure_counts, r)[0] * 100 for r in _retries]
    _colors = [ORANGE if r == _chosen else BLUE for r in _retries]

    fig_live, ax_live = plt.subplots(figsize=(7, 4))
    ax_live.bar(_retries, _rates, color=_colors, edgecolor="white")
    ax_live.plot(_retries, _rates, color=GREEN, marker="o", linewidth=2, zorder=3)
    _here = _rates[_chosen]
    ax_live.text(_chosen, _here + 2, f"{_here:.0f}%", ha="center",
                 fontsize=11, fontweight="bold")
    ax_live.set_title(f"max_retries = {_chosen}  →  {_here:.1f}% of jobs succeed")
    ax_live.set_xlabel("max_retries")
    ax_live.set_ylabel("success rate (%)")
    ax_live.set_ylim(0, 108)
    ax_live.spines["top"].set_visible(False)
    ax_live.spines["right"].set_visible(False)

    mo.vstack([
        mo.md(f"**Chosen budget:** {_chosen} retries "
              f"({_chosen + 1} total attempts per job) → **{_here:.1f}%** success."),
        ax_live,
    ])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Q1 · Why isn't `json.loads` succeeding enough on its own?": mo.md(
                "Because valid JSON and a *valid record* are two different things. "
                "`json.loads` only guarantees the text parses — `{\"naem\": \"Maya\"}` "
                "parses fine but is the wrong shape. You need a **second gate**, "
                "`validate()`, to check required fields and types before your code "
                "trusts the object. Parsing checks syntax; validation checks meaning."
            ),
            "Q2 · Why feed the error back into the retry instead of just calling again?": mo.md(
                "A blind retry re-rolls the same dice and often reproduces the same "
                "mistake. A **corrective** retry includes the specific failure (\"age "
                "must be an int, you sent a string\") in the next prompt, which steers "
                "the model toward the fix. It also converges faster, so you can keep "
                "`max_retries` small — and small budgets matter because every extra "
                "attempt adds latency and token cost."
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What you learned

        - **Two gates, not one.** Parsing (`json.loads`) catches malformed text;
          `validate()` catches the wrong *shape*. You need both before trusting data.
        - A **dataclass + a validator** turn "the model should return JSON" into a
          precise, checkable contract.
        - The **robust loop** — parse → validate → *corrective* retry → graceful
          give-up — converts a flaky text generator into a dependable data source.
        - **Retries have a knee.** The first one or two rescue most jobs; beyond that
          you pay latency and tokens for shrinking gains. Tune the budget to the curve.
        - In production, prefer **tool/function calling** or **constrained decoding** to
          prevent failures up front — but keep validation and a couple of retries as
          your safety net.

        **Keep going:** the course chapter
        [`../course/p2-talking-to-models.html`](../course/p2-talking-to-models.html)
        covers prompting and structured outputs in the broader context of talking to
        models.
        """
    )
    return


if __name__ == "__main__":
    app.run()
