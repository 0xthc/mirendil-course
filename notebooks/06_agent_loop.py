import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 06 · Build an agent loop (no API key required)

        A single model call answers a question. An **agent** chases a *goal*: it calls a
        tool, looks at what came back, decides what to do next, and keeps going until the
        job is done. That loop is most of the value — and most of the danger.

        In this notebook you'll build the whole thing on tiny, readable Python, with
        **no network and no API key**. We replace the real LLM with a deterministic
        `mock_llm(...)` so every run is reproducible and instant.

        **What you'll build, step by step:**

        1. A tiny **tool registry** — 3 real Python tools (`calculator`, `search`,
           `word_count`) plus the JSON-Schema declarations the model would actually see.
        2. A **mock LLM** — a rule-based stand-in that returns either a *tool call* or a
           *final answer*, so the loop is fully deterministic.
        3. The **agent loop** itself — observe → think → act → get result → repeat — with a
           **max-steps guard** and a per-step **token/cost counter**.
        4. **Transcripts** of real runs (calculation, lookup, multi-step).
        5. **Failure handling** — a tool raises, the loop catches it, feeds the error back,
           and recovers gracefully.
        6. **Charts** — cost growing every step, and steps-to-completion across goals
           (including a runaway loop being *capped* by the guard).
        7. An **interactive** panel: drag a `max_steps` slider and pick a goal; watch the
           transcript react.

        You're a full-stack engineer, not a data scientist — so we trace **control flow and
        data**, not model internals. If you can read a `for` loop, you can read this.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    import ast
    import json
    import operator

    return mo, np, plt, re, ast, json, operator


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** A plain model call is like asking a smart friend one question and
            getting one answer. An **agent** is like handing that friend a calculator, a
            search box, and a notepad and saying *"figure this out."* Now they can look
            things up, try things, and react to what they find — **think → use a tool →
            look at the result → repeat → until done.**

            But you also want a *"stop after N tries"* rule and a spending limit, because a
            confused agent will happily keep dialing forever. That stop rule is the
            **loop guard**, and building it is half the job.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1 · The tool registry — what the model is allowed to do

        The model can only call tools you have **declared**. A declaration is a *name*, a
        *description*, and a *JSON Schema* for its arguments. The model picks a tool by
        **reading those words**, so the description is the contract — it's prompt
        engineering, not an afterthought.

        Here are three real Python functions. Note `calculator` uses a tiny safe-eval
        (an AST walk) instead of raw `eval`, because you should never hand `eval` a string
        a model produced.
        """
    )
    return


@app.cell
def _(ast, operator):
    # ---- Three real tools -----------------------------------------------------

    _ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _safe_eval(node):
        "Recursively evaluate a parsed arithmetic expression — numbers and +-*/ only."
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
        raise ValueError("unsupported expression")

    def calculator(expr: str):
        "Evaluate an arithmetic expression like '23 * 17 + 100'. Returns a number."
        tree = ast.parse(expr, mode="eval")
        return _safe_eval(tree.body)

    # A canned, offline "search index" so we need no network.
    _SEARCH_INDEX = {
        "capital of france": "Paris is the capital of France.",
        "otters": "Otters hold hands while sleeping so they don't drift apart.",
        "speed of light": "The speed of light is about 299,792 km per second.",
    }

    def search(query: str):
        "Look up a short fact for a query. Returns a one-sentence string (offline index)."
        q = query.lower()
        for key, fact in _SEARCH_INDEX.items():
            if key in q:
                return fact
        return f"No results found for {query!r}."

    def word_count(text: str):
        "Count the words in a piece of text. Returns an integer."
        return len(text.split())

    # Dispatch table: tool name -> callable. This is what the loop calls.
    TOOL_FNS = {"calculator": calculator, "search": search, "word_count": word_count}

    return calculator, search, word_count, TOOL_FNS


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The schemas the model actually sees

        A Python function has types and a docstring, but the model is shown a JSON Schema.
        Below is the `TOOLS` registry: the exact structured description we'd send to a real
        API. **Few, sharp tools beat many fuzzy ones** — three well-named tools go a long way.
        """
    )
    return


@app.cell
def _():
    # ---- The tool registry: declarations the model reads -----------------------
    TOOLS = [
        {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression. Use for any math the user asks for.",
            "input_schema": {
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "e.g. '23 * 17 + 100'"}},
                "required": ["expr"],
            },
        },
        {
            "name": "search",
            "description": "Look up a short fact for a query. Use when you need a fact you don't know.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "word_count",
            "description": "Count the words in a piece of text. Use after you have some text.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    ]
    return (TOOLS,)


@app.cell
def _(mo, json, TOOLS):
    _schema_md = "\n\n".join(
        f"**`{t['name']}`** — {t['description']}\n\n```json\n{json.dumps(t['input_schema'], indent=2)}\n```"
        for t in TOOLS
    )
    mo.md("The model sees these three declarations:\n\n" + _schema_md)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2 · The mock LLM — a deterministic stand-in

        > **This replaces a real LLM API.** A real model reads the running transcript and
        > *generates* its next move. Our `mock_llm` does the same job with simple rules, so
        > the notebook runs offline and identically every time.

        Its contract is exactly a real model's: given `messages` (the running transcript)
        and `tools` (the registry), return **either**

        - `{"thought": ..., "tool": name, "args": {...}}` — *"please run this tool"*, or
        - `{"thought": ..., "final": "..."}` — *"I'm done, here's the answer."*

        The rules read the goal and what tool results already exist, then pick the next
        move. Notice the **recovery rule first**: if the last tool call errored, it stops
        gracefully instead of charging ahead.
        """
    )
    return


@app.cell
def _(re):
    def mock_llm(messages, tools):
        "A rule-based stand-in for a real LLM. Returns a tool call or a final answer."
        goal = next(m["content"] for m in messages if m["role"] == "user")
        g = goal.lower()
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        done = [m["tool"] for m in tool_msgs]

        def result_of(name):
            return next((m["content"] for m in reversed(tool_msgs)
                         if m["tool"] == name and not m.get("error")), None)

        # 0) RECOVERY: if the most recent tool call failed, acknowledge and stop.
        if tool_msgs and tool_msgs[-1].get("error"):
            return {"thought": "The last tool errored — I can't complete this. Stopping cleanly.",
                    "final": f"I couldn't finish: the tool failed with \"{tool_msgs[-1]['content']}\"."}

        # 1) RUNAWAY: a goal with no end condition — keeps asking forever (needs the guard).
        if "forever" in g or "runaway" in g:
            return {"thought": "Still going… this goal has no stop condition.",
                    "tool": "search", "args": {"query": "more"}}

        # 2) MULTI-STEP: search, then count the words in what came back.
        if "count" in g and "word" in g:
            if "search" not in done:
                return {"thought": "I need a fact first, then I'll count its words.",
                        "tool": "search", "args": {"query": goal}}
            if "word_count" not in done:
                return {"thought": "Got the text; now count the words in it.",
                        "tool": "word_count", "args": {"text": result_of("search")}}
            return {"thought": "I have the fact and its word count — done.",
                    "final": f"Found: \"{result_of('search')}\" — that snippet has {result_of('word_count')} words."}

        # 3) MATH: extract an arithmetic expression, compute it, then report.
        if re.search(r"\d\s*[-+*/]\s*\d", goal):
            if "calculator" not in done:
                m = re.search(r"[-+]?\d[\d\s+\-*/().]*\d", goal)
                return {"thought": "This needs arithmetic — call the calculator.",
                        "tool": "calculator", "args": {"expr": (m.group(0).strip() if m else goal)}}
            return {"thought": "The calculator returned a value — report it.",
                    "final": f"The answer is {result_of('calculator')}."}

        # 4) LOOKUP: a single fact search, then answer.
        if any(k in g for k in ("search", "capital", "find", "look up", "speed of light", "fact")):
            if "search" not in done:
                return {"thought": "I don't know this offhand — search for it.",
                        "tool": "search", "args": {"query": goal}}
            return {"thought": "Search returned a fact — answer with it.",
                    "final": result_of("search")}

        # 5) FALLBACK: nothing matched.
        return {"thought": "No tool fits this goal.",
                "final": "I'm not sure how to help with that one."}

    return (mock_llm,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3 · The agent loop

        Now the engine. The control bounces between the **model** (decides) and **your
        code** (runs the tool, feeds the result back):

        ```
        observe ──▶ think ──▶ act ──▶ get result ──▶ (repeat)
        ```

        Three things every production loop needs, all here:

        - **A max-steps guard.** Never trust the loop to end on its own. If we hit the cap,
          we stop and say so. (Real loops add repeat-detection and dollar budgets too — see
          the notes at the end.)
        - **A cost counter.** Each turn re-sends the *entire growing transcript*, so cost
          grows every step. We estimate tokens per turn and accumulate.
        - **Error handling.** Tool calls run inside `try/except`; a failure becomes a normal
          observation fed back to the model, not a crash.

        `run_agent` returns a structured record (every step, the final answer, and *why* it
        stopped) so we can render and chart it.
        """
    )
    return


@app.cell
def _(mock_llm, TOOLS, TOOL_FNS):
    def count_tokens(messages):
        "Rough token estimate: ~4 chars/token, plus a little per-message overhead."
        return sum(len(str(m.get("content", ""))) for m in messages) // 4 + 3 * len(messages)

    def run_agent(goal, max_steps=6, price_per_1k=0.003,
                  llm=mock_llm, registry=TOOLS, fns=TOOL_FNS):
        "Drive the observe→think→act loop until the model says done or we hit max_steps."
        messages = [
            {"role": "system", "content": "You are a helpful agent. Use tools, then answer."},
            {"role": "user", "content": goal},
        ]
        steps, cum_tokens, cum_cost = [], 0, 0.0
        final_answer, stopped = None, "max_steps"

        for _i in range(max_steps):
            tokens = count_tokens(messages)            # cost of THIS turn's transcript
            cost = tokens / 1000 * price_per_1k
            cum_tokens += tokens
            cum_cost += cost

            reply = llm(messages, registry)            # think
            base = {"thought": reply.get("thought", ""), "tokens": tokens,
                    "cum_tokens": cum_tokens, "cost": cost, "cum_cost": cum_cost}

            if "final" in reply:                       # model says it's done
                steps.append({**base, "kind": "final", "final": reply["final"]})
                final_answer, stopped = reply["final"], "final"
                break

            tool, args = reply["tool"], reply["args"]  # act
            try:
                result, error = fns[tool](**args), False
            except Exception as exc:                   # observe a failure, don't crash
                result, error = f"{type(exc).__name__}: {exc}", True

            steps.append({**base, "kind": "tool", "tool": tool, "args": args,
                          "result": result, "error": error})
            messages.append({"role": "assistant", "content": f"call {tool}({args})",
                             "tool": tool, "args": args})
            messages.append({"role": "tool", "tool": tool, "content": str(result), "error": error})

        return {"goal": goal, "steps": steps, "final": final_answer,
                "stopped": stopped, "max_steps": max_steps, "cum_cost": cum_cost,
                "cum_tokens": sum(s["tokens"] for s in steps)}

    return run_agent, count_tokens


@app.cell
def _(mo):
    def render_transcript(rec):
        "Turn a run record into a readable mo.md transcript."
        lines = [f"**Goal:** {rec['goal']}", ""]
        for n, s in enumerate(rec["steps"], 1):
            if s["kind"] == "final":
                lines.append(f"**Step {n} · answer** &nbsp; *{s['thought']}*")
                lines.append(f"> {s['final']}")
            else:
                flag = " ⚠️ error" if s.get("error") else ""
                lines.append(f"**Step {n} · tool**{flag} &nbsp; *{s['thought']}*")
                lines.append(f"- call: `{s['tool']}({s['args']})`")
                lines.append(f"- result: `{s['result']}`")
            lines.append("")
        badge = "✅ finished" if rec["stopped"] == "final" else "⛔ hit max_steps guard"
        lines.append(f"**Stopped:** {badge} after {len(rec['steps'])} step(s) · "
                     f"~{rec['cum_tokens']} tokens · ~${rec['cum_cost']:.5f}")
        return mo.md("\n".join(lines))

    return (render_transcript,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4 · Run it on real goals

        Three goals exercising the three paths the mock can take. Watch control bounce
        between *think* (the model) and *result* (your code), and watch the token count
        climb each step.
        """
    )
    return


@app.cell
def _(run_agent, render_transcript):
    rec_math = run_agent("What is 23 * 17 + 100?")
    render_transcript(rec_math)
    return (rec_math,)


@app.cell
def _(run_agent, render_transcript):
    rec_lookup = run_agent("Search for the capital of France and tell me.")
    render_transcript(rec_lookup)
    return (rec_lookup,)


@app.cell
def _(run_agent, render_transcript):
    # A genuine multi-step goal: search, THEN count words in the result.
    rec_multi = run_agent("Search for a fun fact about otters, then count the words in it.")
    render_transcript(rec_multi)
    return (rec_multi,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5 · Failure handling — when a tool blows up

        Tools fail: bad input, a flaky API, a timeout. The loop must treat a failure as
        **just another observation**. Below, the agent is asked to divide by zero. The
        calculator raises `ZeroDivisionError`; `run_agent` catches it, feeds the error
        text back, and the mock LLM's **recovery rule** stops cleanly with an explanation
        instead of crashing or looping.
        """
    )
    return


@app.cell
def _(run_agent, render_transcript):
    rec_fail = run_agent("What is 10 / 0?")
    render_transcript(rec_fail)
    return (rec_fail,)


@app.cell
def _(mo, rec_fail):
    _err_step = next((s for s in rec_fail["steps"] if s.get("error")), None)
    mo.callout(
        mo.md(
            f"""
            The tool raised **`{_err_step['result']}`**, but the program didn't crash —
            the error became an observation, the model saw it, and the run ended with a
            graceful answer. **A failing tool is data, not a crash.**
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6 · Charts

        ### (a) Cost grows every single step

        Each turn re-sends the whole transcript, which keeps getting longer — so per-step
        token cost *rises* and cumulative cost curves *upward*. This is the headline cost
        property of agents. (Bars = tokens spent on that turn; line = running total.)
        """
    )
    return


@app.cell
def _(plt, np, rec_multi):
    _steps = rec_multi["steps"]
    _x = np.arange(1, len(_steps) + 1)
    _per = [s["tokens"] for s in _steps]
    _cum = [s["cum_tokens"] for s in _steps]

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.bar(_x, _per, color="#57b6f5", label="tokens this turn")
    _ax.set_xlabel("loop step")
    _ax.set_ylabel("tokens this turn", color="#57b6f5")
    _ax.set_xticks(_x)
    for _xi, _v in zip(_x, _per):
        _ax.text(_xi, _v + 0.5, str(_v), ha="center", va="bottom", fontsize=8, color="#3a7fb0")

    _ax2 = _ax.twinx()
    _ax2.plot(_x, _cum, color="#f0986b", marker="o", linewidth=2, label="cumulative tokens")
    _ax2.set_ylabel("cumulative tokens", color="#f0986b")

    _ax.set_title("Per-step vs. cumulative tokens (cost climbs as the transcript grows)")
    _l1, _lab1 = _ax.get_legend_handles_labels()
    _l2, _lab2 = _ax2.get_legend_handles_labels()
    _ax.legend(_l1 + _l2, _lab1 + _lab2, loc="upper left", fontsize=8)
    _fig.tight_layout()
    _ax
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### (b) Steps to completion — and the runaway loop the guard catches

        Different goals take different numbers of steps. The **runaway** goal has no stop
        condition: without a guard it would run *forever*. With `max_steps`, it's capped —
        the orange bar slams into the ceiling and the loop bails. **You** make it stop, not
        the model.
        """
    )
    return


@app.cell
def _(plt, np, run_agent):
    _cap = 6
    _runs = [
        ("math", run_agent("What is 23 * 17 + 100?", max_steps=_cap)),
        ("lookup", run_agent("Search for the capital of France.", max_steps=_cap)),
        ("multi-step", run_agent("Search for a fact about otters, then count the words.", max_steps=_cap)),
        ("runaway", run_agent("Keep searching forever.", max_steps=_cap)),
    ]
    _labels = [name for name, _ in _runs]
    _heights = [len(r["steps"]) for _, r in _runs]
    _colors = ["#5fd38a" if r["stopped"] == "final" else "#f0986b" for _, r in _runs]

    _fig2, _bx = plt.subplots(figsize=(7, 4))
    _bars = _bx.bar(_labels, _heights, color=_colors)
    _bx.axhline(_cap, color="#888", linestyle="--", linewidth=1)
    _bx.text(len(_labels) - 0.5, _cap + 0.05, f"max_steps = {_cap}", ha="right", va="bottom",
             fontsize=8, color="#555")
    for _b, (_, _r) in zip(_bars, _runs):
        _tag = "done" if _r["stopped"] == "final" else "capped"
        _bx.text(_b.get_x() + _b.get_width() / 2, _b.get_height() + 0.05, _tag,
                 ha="center", va="bottom", fontsize=9)
    _bx.set_ylabel("loop steps taken")
    _bx.set_ylim(0, _cap + 1.2)
    _bx.set_title("Steps to completion (green = finished · orange = stopped by the guard)")
    _fig2.tight_layout()
    _bx
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7 · Interactive — your hand on the guard

        Pick a goal and drag the `max_steps` guard. The **runaway** goal never finishes, so
        the transcript length tracks the cap exactly — proof the guard, not the model, is
        what stops it. The other goals finish before the cap, so raising it changes nothing.
        """
    )
    return


@app.cell
def _(mo):
    goal_picker = mo.ui.dropdown(
        options={
            "math: 23 * 17 + 100": "What is 23 * 17 + 100?",
            "lookup: capital of France": "Search for the capital of France and tell me.",
            "multi-step: otters + word count": "Search for a fun fact about otters, then count the words in it.",
            "failure: 10 / 0": "What is 10 / 0?",
            "runaway: never finishes": "Keep searching forever.",
        },
        value="runaway: never finishes",
        label="goal",
    )
    max_steps_slider = mo.ui.slider(1, 12, value=4, label="max_steps guard")
    mo.vstack([goal_picker, max_steps_slider])
    return goal_picker, max_steps_slider


@app.cell
def _(mo, goal_picker, max_steps_slider, run_agent, render_transcript):
    rec_live = run_agent(goal_picker.value, max_steps=max_steps_slider.value)
    mo.vstack([
        mo.md(f"Running with **max_steps = {max_steps_slider.value}** → "
              f"**{len(rec_live['steps'])} step(s)**, stopped on **{rec_live['stopped']}**."),
        render_transcript(rec_live),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8 · Check your understanding
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Why does an agent get more expensive on every step?": mo.md(
                "Each turn re-sends the **entire running transcript** to the model — every "
                "previous thought, tool call, and result. The transcript only grows, so the "
                "per-turn token count (and dollar cost) rises step after step. That's why "
                "tools should return **compact** results: a tool that dumps 20 KB inflates "
                "the cost of *every later step*, not just its own."
            ),
            "If the model decides when it's done, why do we still need a max-steps guard?": mo.md(
                "Because the model **sometimes never decides**. A flaky tool, an impossible "
                "goal, or a confused plan can loop forever (our `runaway` goal does exactly "
                "this). The guard is *your* code's hard stop — you never trust the loop to "
                "end on its own. Production loops layer several independent guards: max steps, "
                "repeat-detection (same tool + same args twice), and a dollar/token budget."
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What you learned**

            - An agent is a **loop where the model chooses the next action and the world
              answers back**: observe → think → act → get result → repeat.
            - The model never touches your systems directly — it can only *request* a tool
              call as structured data. **Your code** validates, runs it, and feeds the result
              back. That gap is where every safety control lives.
            - The **tool registry** (name + description + JSON Schema) is the entire interface
              the model sees. Few, sharp tools beat many fuzzy ones.
            - **Cost grows every step** because the whole transcript is re-sent each turn.
            - **Guards are non-negotiable**: a max-steps cap (plus repeat-detection and
              budgets) is what makes the loop *finish*.
            - **Failures are observations**: catch the error, feed it back, recover or bail —
              don't crash.

            Next: read the matching course chapter, **`../course/p3-agents.html`**, which
            walks the same loop, registry, and guard ideas with production framing.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
