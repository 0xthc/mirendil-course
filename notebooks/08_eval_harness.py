import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 08 · Building an eval harness

        Before you ship a prompt tweak, a new model, or a "tiny" refactor of your
        AI feature, you want to answer one question with a straight face:
        **did this change make the product better or worse?** "It looked good when
        I tried it twice" is not an answer. An **eval harness** is.

        In this notebook you build a complete, runnable eval harness from scratch —
        the same shape you'd put in CI in front of a real LLM:

        1. An **offline dataset** of `{input, expected, rubric}` cases (a graded test).
        2. Two **scorers**: a strict *exact/structural* check, and a *mock LLM-as-judge*
           that grades free-form answers — and we'll be honest about the judge's pitfalls.
        3. The **harness**: run a model over the dataset, score every case, aggregate.
        4. A **v1 vs v2 comparison** where v2 is better overall but secretly
           **regresses on one case** — exactly the bug an eval is built to catch.
        5. A **regression gate**: a function that returns PASS/FAIL so a deploy can
           be blocked automatically.
        6. **Charts** that make the regression impossible to miss, and an
           **interactive threshold slider** so you can feel how the gate behaves.

        Everything here is **100% offline** — no network, no API keys, no torch. The
        model and the judge are deterministic Python stand-ins so the notebook runs
        identically every time. In production you'd swap `mock_model_*` for a real API
        call and `mock_judge` for a real LLM grader; the *harness around them is the
        part that matters*, and that part is real.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import re

    return mo, np, plt, re


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** An **eval** is a graded test you give your model — the *same*
            test, the *same* way, every time. You write down a handful of questions
            (`input`) together with the answer you'd accept (`expected`), let the
            model take the test, and grade it. Because the test never changes, the
            **score** becomes a number you can trust: if it goes up after a change,
            the change helped; if it goes down, it hurt. No vibes, no "seems fine."
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1 · The dataset — a graded test for the model

        Our example product is a **customer-support assistant**: given a question, it
        should return a short, correct answer. Our eval set is a list of dicts. Each
        case carries three things:

        | field | what it is | used by |
        |---|---|---|
        | `input` | the question we send the model | the model |
        | `expected` | a gold answer we'd be happy with | the exact scorer + judge |
        | `rubric` | keywords/facts the answer **must** contain | the judge |

        Keep the dataset small, hand-written, and version-controlled. 8–12 sharp cases
        you actually understand beat 10,000 noisy ones. This *is* your test suite — treat
        it like code.
        """
    )
    return


@app.cell
def _():
    # The offline eval set. 10 hand-written cases. This is the whole "test suite".
    EVAL_CASES = [
        {
            "id": "c1",
            "input": "How do I reset my password?",
            "expected": "Click 'Forgot password' on the login page and follow the email link.",
            "rubric": ["forgot password", "email"],
        },
        {
            "id": "c2",
            "input": "What is your refund window?",
            "expected": "You can request a refund within 30 days of purchase.",
            "rubric": ["30 days", "refund"],
        },
        {
            "id": "c3",
            "input": "How do I contact support?",
            "expected": "Email support@example.com or use the in-app chat.",
            "rubric": ["support@example.com", "chat"],
        },
        {
            "id": "c4",
            "input": "Where can I download my invoice?",
            "expected": "Invoices are under Billing > Invoices in your account settings.",
            "rubric": ["billing", "invoices"],
        },
        {
            "id": "c5",
            "input": "Do you offer a free trial?",
            "expected": "Yes, we offer a 14-day free trial with no credit card required.",
            "rubric": ["14-day", "free trial"],
        },
        {
            "id": "c6",
            "input": "How do I cancel my subscription?",
            "expected": "Go to Billing > Subscription and click Cancel.",
            "rubric": ["billing", "cancel"],
        },
        {
            "id": "c7",
            "input": "Do you ship internationally?",
            "expected": "Yes, we ship to over 50 countries with standard customs fees.",
            "rubric": ["yes", "ship", "countries"],
        },
        {
            "id": "c8",
            "input": "What payment methods do you accept?",
            "expected": "We accept Visa, Mastercard, Amex, and PayPal.",
            "rubric": ["paypal", "visa"],
        },
        {
            "id": "c9",
            "input": "Is my data encrypted?",
            "expected": "Yes, all data is encrypted in transit and at rest.",
            "rubric": ["encrypted", "transit", "rest"],
        },
        {
            "id": "c10",
            "input": "How do I upgrade my plan?",
            "expected": "Open Billing > Plan and choose a higher tier.",
            "rubric": ["billing", "plan"],
        },
    ]
    return (EVAL_CASES,)


@app.cell
def _(EVAL_CASES, mo):
    # Render the dataset as a readable table.
    def _render_dataset(cases):
        header = "| id | input | expected | rubric |\n|---|---|---|---|\n"
        body = "\n".join(
            f"| `{c['id']}` | {c['input']} | {c['expected']} | {', '.join(c['rubric'])} |"
            for c in cases
        )
        return header + body

    mo.md(
        f"**The eval set ({len(EVAL_CASES)} cases):**\n\n" + _render_dataset(EVAL_CASES)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2 · Two scorers, two jobs

        A scorer turns *(input, output, expected)* into a number. We build two, because
        they answer different questions:

        - **Exact / structural scorer** — *"is the output literally what we expected,
          and is it well-formed?"* Cheap, deterministic, zero ambiguity. Perfect when
          there's one right answer (a JSON shape, a status code, a canonical string).
          But it's brutally strict: a correct answer phrased differently scores **0**.

        - **LLM-as-judge** — *"would a reasonable grader call this answer good?"* It reads
          the free-form output and grades it against the expected answer + rubric. This
          is how you score open-ended text where many wordings are fine. The catch: a
          judge is itself a model, so it has **pitfalls** you must respect (below).

        Our `mock_judge` is a deterministic, rule-based stand-in: it rewards covering the
        rubric keywords and overlapping with the gold answer. A real judge would be an
        LLM call like *"Score 0–1 how well OUTPUT answers INPUT given EXPECTED."* The
        harness doesn't care which one you plug in.
        """
    )
    return


@app.cell
def _(re):
    # --- Text normalization shared by both scorers -------------------------------
    _STOPWORDS = {
        "the", "a", "an", "to", "of", "in", "and", "or", "is", "are", "you",
        "your", "we", "our", "with", "can", "for", "on", "at", "it", "by",
    }

    def normalize_text(s):
        "lowercase, collapse whitespace, drop trailing punctuation"
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s.strip(" .!?")

    def content_tokens(s):
        "meaningful word tokens (drops punctuation, short words, stopwords)"
        words = re.findall(r"[a-z0-9@.]+", s.lower())
        return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}

    return content_tokens, normalize_text


@app.cell
def _(content_tokens, normalize_text):
    # --- Scorer 1: exact / structural (deterministic, no judgement) ---------------
    def exact_match(output, expected):
        "1.0 if the normalized output equals the normalized gold answer, else 0.0"
        return 1.0 if normalize_text(output) == normalize_text(expected) else 0.0

    def format_valid(output):
        "structural sanity: non-empty, capitalized, ends with '.', a real sentence"
        o = output.strip()
        ok = (
            len(o) > 0
            and o[0].isupper()
            and o.endswith(".")
            and 3 <= len(o.split()) <= 30
        )
        return 1.0 if ok else 0.0

    # --- Scorer 2: mock LLM-as-judge (deterministic stand-in for a real LLM) -------
    def mock_judge(input_q, output, expected, rubric):
        """
        Stand-in for an LLM grader. Returns a score in [0, 1].

        Real version: one API call asking a model to rate the answer. Here we fake
        it deterministically so the notebook is reproducible and offline:
          - 60% of the score = fraction of rubric keywords present in the output
          - 40% of the score = how much of the gold answer's content it recovers
        """
        o = normalize_text(output)
        if not o:
            return 0.0
        # rubric coverage: each required fact/keyword present in the answer?
        hits = sum(1 for kw in rubric if normalize_text(kw) in o)
        coverage = hits / len(rubric) if rubric else 0.0
        # content overlap with the gold answer (recall of expected content words)
        exp_tok = content_tokens(expected)
        out_tok = content_tokens(output)
        overlap = len(exp_tok & out_tok) / len(exp_tok) if exp_tok else 0.0
        return round(0.6 * coverage + 0.4 * overlap, 2)

    return exact_match, format_valid, mock_judge


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **LLM-as-judge pitfalls — read before you trust the number.** A judge is a
            model grading a model, so it inherits model failure modes:

            - **Bias toward length & style.** Judges often score verbose, confident, or
              nicely-formatted answers higher even when they're *wrong*. (Our mock judge
              rewards keyword stuffing — a real one can be fooled the same way.)
            - **Self-preference.** A judge tends to favor outputs from the same model
              family it belongs to.
            - **Non-determinism.** A real LLM judge can give different scores to the same
              answer on different runs — pin temperature low and average if needed.
            - **It must be validated.** Before you trust a judge, check its grades against
              a human-labeled sample. An unvalidated judge is just a confident guess.

            Rule of thumb: use exact/structural scorers wherever a single right answer
            exists; reserve the judge for genuinely open-ended output, and **keep it
            honest with a rubric and spot-checks.**
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3 · Two models to compare — v1 and v2

        To demonstrate the harness we need something to grade. `mock_model_v1` and
        `mock_model_v2` are deterministic stand-ins (just lookup tables of canned
        answers) so the whole notebook is reproducible. **v2 is the "improved" model**:
        on most cases it returns the polished, gold-matching answer.

        But v2 has a **hidden regression** baked in — on the international-shipping
        question (`c7`) it confidently returns the *wrong* answer. v1 got that one
        right. This is the exact situation an eval harness exists to catch: an average
        that goes **up** while a specific, important case quietly **breaks**.

        In production these two functions would be two real API calls (e.g. an old vs.
        new prompt, or two model versions). The harness code below wouldn't change.
        """
    )
    return


@app.cell
def _():
    # Deterministic stand-in models: lookup tables keyed by the question.
    # v1 = older model: answers are partial / missing some required facts.
    _V1_OUTPUTS = {
        "How do I reset my password?": "Use the Forgot password link and check your email.",
        "What is your refund window?": "Refunds are available.",
        "How do I contact support?": "Contact support@example.com.",
        "Where can I download my invoice?": "Check your account billing section.",
        "Do you offer a free trial?": "Yes, there is a free trial.",
        "How do I cancel my subscription?": "You can cancel in your billing settings.",
        # c7: v1 gets international shipping RIGHT.
        "Do you ship internationally?": "Yes, we ship to over 50 countries with standard customs fees.",
        "What payment methods do you accept?": "We take Visa and PayPal.",
        "Is my data encrypted?": "Data is encrypted.",
        "How do I upgrade my plan?": "Upgrade in the billing area.",
    }

    # v2 = "improved" model: polished, gold-matching answers... except one regression.
    _V2_OUTPUTS = {
        "How do I reset my password?": "Click 'Forgot password' on the login page and follow the email link.",
        "What is your refund window?": "You can request a refund within 30 days of purchase.",
        "How do I contact support?": "Email support@example.com or use the in-app chat.",
        "Where can I download my invoice?": "Invoices are under Billing > Invoices in your account settings.",
        "Do you offer a free trial?": "Yes, we offer a 14-day free trial with no credit card required.",
        "How do I cancel my subscription?": "Go to Billing > Subscription and click Cancel.",
        # c7: REGRESSION — v2 confidently gives the wrong answer.
        "Do you ship internationally?": "No, we only ship within the United States.",
        "What payment methods do you accept?": "We accept Visa, Mastercard, Amex, and PayPal.",
        "Is my data encrypted?": "Yes, all data is encrypted in transit and at rest.",
        "How do I upgrade my plan?": "Open Billing > Plan and choose a higher tier.",
    }

    def mock_model_v1(input_q):
        "STAND-IN for a real model/API call (older version)."
        return _V1_OUTPUTS.get(input_q, "")

    def mock_model_v2(input_q):
        "STAND-IN for a real model/API call (new version under test)."
        return _V2_OUTPUTS.get(input_q, "")

    return mock_model_v1, mock_model_v2


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4 · The harness — run, score, aggregate

        This is the heart of the whole notebook, and it's refreshingly small. Give it
        a model function and the dataset, and it:

        1. runs the model over **every** case,
        2. applies **every** scorer to each output,
        3. records a per-case row, and
        4. **aggregates** into headline metrics (exact-match accuracy, mean judge score,
           format-valid rate, and pass rate at a threshold).

        `run_eval` is pure and reusable — we'll call it for v1 and again for v2 and just
        compare the results.
        """
    )
    return


@app.cell
def _(exact_match, format_valid, mock_judge):
    def run_eval(model_fn, cases, threshold=0.6):
        "Run a model over the dataset, score each case, and aggregate."
        rows = []
        for c in cases:
            out = model_fn(c["input"])
            judge = mock_judge(c["input"], out, c["expected"], c["rubric"])
            rows.append(
                {
                    "id": c["id"],
                    "input": c["input"],
                    "output": out,
                    "expected": c["expected"],
                    "judge": judge,
                    "exact": exact_match(out, c["expected"]),
                    "format_valid": format_valid(out),
                    "pass": judge >= threshold,
                }
            )
        n = len(rows)
        agg = {
            "mean_judge": sum(r["judge"] for r in rows) / n,
            "exact_acc": sum(r["exact"] for r in rows) / n,
            "format_rate": sum(r["format_valid"] for r in rows) / n,
            "pass_rate": sum(1 for r in rows if r["pass"]) / n,
        }
        return {"rows": rows, "agg": agg}

    return (run_eval,)


@app.cell
def _(EVAL_CASES, mock_model_v1, mock_model_v2, run_eval):
    DEFAULT_THRESHOLD = 0.6
    results_v1 = run_eval(mock_model_v1, EVAL_CASES, DEFAULT_THRESHOLD)
    results_v2 = run_eval(mock_model_v2, EVAL_CASES, DEFAULT_THRESHOLD)
    return DEFAULT_THRESHOLD, results_v1, results_v2


@app.cell
def _(DEFAULT_THRESHOLD, mo, results_v2):
    # Per-case results table for the model under test (v2).
    def _render_results(res, threshold):
        header = (
            "| id | input | output | judge | exact | format | pass |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        lines = []
        for r in res["rows"]:
            verdict = "✅" if r["pass"] else "❌"
            lines.append(
                f"| `{r['id']}` | {r['input']} | {r['output']} | "
                f"{r['judge']:.2f} | {r['exact']:.0f} | {r['format_valid']:.0f} | {verdict} |"
            )
        agg = res["agg"]
        foot = (
            f"\n\n**Aggregates (v2):** mean judge `{agg['mean_judge']:.2f}` · "
            f"exact-match `{agg['exact_acc']:.0%}` · format-valid `{agg['format_rate']:.0%}` · "
            f"pass rate `{agg['pass_rate']:.0%}` (threshold {threshold:.2f})"
        )
        return header + "\n".join(lines) + foot

    mo.md(
        "**Per-case results for v2 (the model under test):**\n\n"
        + _render_results(results_v2, DEFAULT_THRESHOLD)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5 · Compare v1 vs v2 — and catch the regression

        Now the payoff. We line up the two runs and look at **per-case deltas** in the
        judge score. If you only looked at the headline mean you'd happily ship v2 — it's
        better on average. The per-case view tells the real story: one case went *down*,
        and it went down hard.
        """
    )
    return


@app.cell
def _(mo, results_v1, results_v2):
    # Per-case comparison table with deltas; flag any regression.
    def _render_compare(base, cand):
        base_by = {r["id"]: r for r in base["rows"]}
        header = (
            "| id | input | v1 judge | v2 judge | Δ | note |\n"
            "|---|---|---|---|---|---|\n"
        )
        lines = []
        for r in cand["rows"]:
            b = base_by[r["id"]]["judge"]
            d = r["judge"] - b
            if d < -1e-9:
                note = "🔴 **REGRESSION**"
            elif d > 1e-9:
                note = "🟢 improved"
            else:
                note = "—"
            lines.append(
                f"| `{r['id']}` | {r['input']} | {b:.2f} | {r['judge']:.2f} | "
                f"{d:+.2f} | {note} |"
            )
        m1 = base["agg"]["mean_judge"]
        m2 = cand["agg"]["mean_judge"]
        foot = (
            f"\n\n**Overall mean judge:** v1 `{m1:.2f}` → v2 `{m2:.2f}` "
            f"({m2 - m1:+.2f}). The average went **up** — but look at `c7`."
        )
        return header + "\n".join(lines) + foot

    mo.md("**v1 vs v2, case by case:**\n\n" + _render_compare(results_v1, results_v2))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The regression gate

        A comparison table is nice for a human, but CI needs a **yes/no**. The gate is a
        pure function that returns `(passed, reasons)`. It **fails the deploy** when
        either:

        - **any case** scores below the absolute `threshold` (a hard quality floor), or
        - the **overall** mean judge score **drops** versus the baseline.

        It also reports *which* cases regressed (dropped vs. baseline) so the on-call
        engineer knows where to look. Wire `passed` to your CI exit code and a bad change
        can't reach production.
        """
    )
    return


@app.cell
def _():
    def regression_gate(base, cand, threshold, overall_tol=0.0):
        """
        Decide whether `cand` (new model) may ship vs `base` (current).
        Returns (passed: bool, reasons: list[str], failing_ids, regressed_ids).
        """
        reasons = []
        # Hard floor: no case may score below the threshold.
        failing_ids = [r["id"] for r in cand["rows"] if r["judge"] < threshold]
        if failing_ids:
            reasons.append(
                f"{len(failing_ids)} case(s) below threshold {threshold:.2f}: {failing_ids}"
            )
        # No overall regression allowed.
        m_base = base["agg"]["mean_judge"]
        m_cand = cand["agg"]["mean_judge"]
        if m_cand < m_base - overall_tol:
            reasons.append(
                f"overall mean judge dropped {m_base:.2f} → {m_cand:.2f}"
            )
        # Informational: per-case drops vs baseline.
        base_by = {r["id"]: r["judge"] for r in base["rows"]}
        regressed_ids = [
            r["id"] for r in cand["rows"] if r["judge"] < base_by[r["id"]] - 1e-9
        ]
        passed = len(reasons) == 0
        return passed, reasons, failing_ids, regressed_ids

    return (regression_gate,)


@app.cell
def _(DEFAULT_THRESHOLD, mo, regression_gate, results_v1, results_v2):
    _passed, _reasons, _failing, _regressed = regression_gate(
        results_v1, results_v2, DEFAULT_THRESHOLD
    )
    _verdict = "PASS ✅ — safe to deploy" if _passed else "FAIL ❌ — block the deploy"
    _detail = (
        "\n".join(f"- {r}" for r in _reasons) if _reasons else "- no problems found"
    )
    _reg = (
        f"\n\nCases that regressed vs v1: **{_regressed}**" if _regressed else ""
    )
    mo.callout(
        mo.md(
            f"### Regression gate @ threshold {DEFAULT_THRESHOLD:.2f}\n\n"
            f"**{_verdict}**\n\n{_detail}{_reg}"
        ),
        kind="success" if _passed else "danger",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6 · Charts — make the regression impossible to miss

        Three views of the same eval run:

        1. **Headline metrics** — v1 vs v2 on the three aggregate scores.
        2. **Per-case judge score** — where the regression hides; the dropped case is
           drawn in **red**.
        3. **Pass / fail counts** at the default threshold.
        """
    )
    return


@app.cell
def _(np, plt, results_v1, results_v2):
    # Chart (a): grouped bar of headline metrics.
    _metrics = ["exact_acc", "mean_judge", "format_rate"]
    _labels = ["exact-match\naccuracy", "mean\njudge score", "format-valid\nrate"]
    _v1 = [results_v1["agg"][m] for m in _metrics]
    _v2 = [results_v2["agg"][m] for m in _metrics]

    _x = np.arange(len(_metrics))
    _w = 0.38
    fig_a, ax_a = plt.subplots(figsize=(7, 4))
    _b1 = ax_a.bar(_x - _w / 2, _v1, _w, label="v1", color="#f0c674")
    _b2 = ax_a.bar(_x + _w / 2, _v2, _w, label="v2", color="#5fd38a")
    for _bars in (_b1, _b2):
        for _bar in _bars:
            ax_a.annotate(
                f"{_bar.get_height():.2f}",
                (_bar.get_x() + _bar.get_width() / 2, _bar.get_height()),
                ha="center", va="bottom", fontsize=9,
            )
    ax_a.set_xticks(_x)
    ax_a.set_xticklabels(_labels)
    ax_a.set_ylim(0, 1.15)
    ax_a.set_ylabel("score (0–1)")
    ax_a.set_title("Headline metrics: v1 vs v2 (higher is better)")
    ax_a.legend()
    ax_a.spines[["top", "right"]].set_visible(False)
    fig_a.tight_layout()
    ax_a
    return


@app.cell
def _(np, plt, results_v1, results_v2):
    # Chart (b): per-case judge score; highlight any case where v2 < v1 in red.
    _ids = [r["id"] for r in results_v1["rows"]]
    _s1 = [r["judge"] for r in results_v1["rows"]]
    _s2 = [r["judge"] for r in results_v2["rows"]]
    _v2_colors = [
        "#e0685f" if b > a else "#5fd38a" for a, b in zip(_s2, _s1)
    ]  # red where v2 dropped below v1

    _x = np.arange(len(_ids))
    _w = 0.4
    fig_b, ax_b = plt.subplots(figsize=(9, 4))
    ax_b.bar(_x - _w / 2, _s1, _w, label="v1", color="#f0c674")
    ax_b.bar(_x + _w / 2, _s2, _w, label="v2", color=_v2_colors)
    ax_b.axhline(0.6, ls="--", lw=1, color="#888", label="threshold 0.60")
    ax_b.set_xticks(_x)
    ax_b.set_xticklabels(_ids)
    ax_b.set_ylim(0, 1.1)
    ax_b.set_xlabel("case")
    ax_b.set_ylabel("judge score")
    ax_b.set_title("Per-case judge score — red = v2 regressed below v1")
    ax_b.legend(loc="lower left")
    ax_b.spines[["top", "right"]].set_visible(False)
    fig_b.tight_layout()
    ax_b
    return


@app.cell
def _(np, plt, results_v1, results_v2):
    # Chart (c): pass / fail counts at the default threshold.
    def _counts(res):
        p = sum(1 for r in res["rows"] if r["pass"])
        return p, len(res["rows"]) - p

    _p1, _f1 = _counts(results_v1)
    _p2, _f2 = _counts(results_v2)

    _x = np.arange(2)
    _w = 0.5
    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    ax_c.bar(_x, [_p1, _p2], _w, label="pass", color="#5fd38a")
    ax_c.bar(_x, [_f1, _f2], _w, bottom=[_p1, _p2], label="fail", color="#e0685f")
    for _i, (_p, _f) in enumerate([(_p1, _f1), (_p2, _f2)]):
        ax_c.annotate(f"{_p} pass", (_i, _p / 2), ha="center", va="center", fontsize=9)
        if _f:
            ax_c.annotate(
                f"{_f} fail", (_i, _p + _f / 2), ha="center", va="center", fontsize=9
            )
    ax_c.set_xticks(_x)
    ax_c.set_xticklabels(["v1", "v2"])
    ax_c.set_ylabel("number of cases")
    ax_c.set_title("Pass / fail counts at threshold 0.60")
    ax_c.legend()
    ax_c.spines[["top", "right"]].set_visible(False)
    fig_c.tight_layout()
    ax_c
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7 · Interactive — feel the threshold

        The gate's behavior hinges on **where you set the bar**. Drag the slider and
        watch the verdict, the failing cases, and the pass rate recompute live. Notice:

        - lower it below the regressed case's score and the hard-floor check stops
          firing — but the gate can *still* fail on the overall-drop rule;
        - raise it and you start tripping borderline cases too.

        Choosing a threshold is a product decision: how good is *good enough* to ship?
        """
    )
    return


@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.6, label="regression-gate threshold"
    )
    threshold_slider
    return (threshold_slider,)


@app.cell
def _(mo, regression_gate, results_v1, results_v2, threshold_slider):
    _t = threshold_slider.value
    _passed, _reasons, _failing, _regressed = regression_gate(
        results_v1, results_v2, _t
    )
    _pass_rate = sum(1 for r in results_v2["rows"] if r["judge"] >= _t) / len(
        results_v2["rows"]
    )
    _verdict = "PASS ✅ — safe to deploy" if _passed else "FAIL ❌ — block the deploy"
    _detail = "\n".join(f"- {r}" for r in _reasons) if _reasons else "- no problems found"
    mo.callout(
        mo.md(
            f"### Gate @ threshold {_t:.2f}\n\n"
            f"**{_verdict}**\n\n"
            f"v2 pass rate: **{_pass_rate:.0%}** · "
            f"cases below threshold: **{_failing or 'none'}** · "
            f"cases regressed vs v1: **{_regressed or 'none'}**\n\n"
            f"{_detail}"
        ),
        kind="success" if _passed else "danger",
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Q1 · Why run an eval *before* shipping a prompt or model change?": mo.md(
                r"""
                Because manual spot-checking is biased and forgetful. You try the two
                or three inputs you happen to think of, they look fine, and you ship —
                meanwhile a case you didn't think to retry has silently broken. An eval
                is the *same* graded test every time, so it turns "seems fine" into a
                **number you can compare across versions**. In this notebook v2 had a
                higher average yet regressed on `c7`; only a fixed, per-case eval +
                gate caught it. Run it in CI and a regression can't merge.
                """
            ),
            "Q2 · What are the pitfalls of LLM-as-judge, and how do you manage them?": mo.md(
                r"""
                A judge is a model grading a model, so it carries model failure modes:
                **length/style bias** (verbose or confident answers score too high),
                **self-preference** (favors its own model family), and
                **non-determinism** (different score on re-run). Manage them by:
                (1) using **exact/structural scorers** wherever a single right answer
                exists and reserving the judge for open-ended text; (2) giving the judge
                a concrete **rubric** instead of a vague "is this good?"; (3) **validating**
                the judge against a human-labeled sample before trusting it; and
                (4) pinning low temperature / averaging multiple grades. A judge is a
                useful instrument, not an oracle.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What you learned

        - An **eval** is a graded test: a fixed dataset of `{input, expected, rubric}`
          cases so a score becomes a number you can trust across versions.
        - **Two scorer types**: cheap **exact/structural** checks for single-answer
          tasks, and an **LLM-as-judge** for open-ended output — with real pitfalls
          (bias, non-determinism) that demand a rubric and human validation.
        - The **harness** is small and reusable: run a model over the dataset, score
          every case, aggregate to headline metrics + a per-case table.
        - **Comparing v1 vs v2 per case** catches the regression an average hides — and
          a **regression gate** turns that into an automatic PASS/FAIL you can wire to
          a CI exit code.
        - The **threshold** is a product decision; the slider let you feel how it moves
          the verdict.

        Swap `mock_model_*` for real API calls and `mock_judge` for a real LLM grader,
        and this is a production eval harness. The harness around the models — the part
        you built here — is the durable, valuable piece.

        **Next:** read the matching course chapter → [`../course/p5-evals.html`](../course/p5-evals.html)
        """
    )
    return


if __name__ == "__main__":
    app.run()
