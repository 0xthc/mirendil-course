import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # A latency & cost model for LLM inference

        You don't need a GPU or a single model call to reason about what an LLM
        feature will **cost** and how **slow** it will feel. You need a small
        spreadsheet's worth of arithmetic — and that's all this notebook is.

        **What you'll build:** two pure functions, `cost(...)` and `latency(...)`,
        that turn *(input tokens, output tokens, model tier, caching)* into dollars
        and milliseconds. Then you'll chart the levers, wire up sliders so you can
        feel them, and end with a ranked cheat-sheet of how to make a feature
        cheaper and faster.

        Everything here is plain Python + numpy + matplotlib. No torch, no network.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** Think of the model as a person reading your prompt out loud and
            then writing an answer by hand.

            - **Reading the prompt (prefill)** is fast: the eyes skim the whole page
              in one glance. Lots of input tokens are read almost *in parallel*.
            - **Writing the answer (decode)** is slow: each word of the reply is
              written **one at a time**, and the model must look back at everything
              it already wrote before adding the next word. This is sequential and
              cannot be skipped.

            So the **output tokens are the expensive part** — in both time and money.
            The prompt is read in one fast pass; the answer is produced letter by
            letter. Keep that picture in your head for the whole notebook.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _():
    # ---- Shared palette (kept consistent across every chart) -----------------
    COL_SMALL = "#5fd38a"   # small tier  (green)
    COL_MID = "#57b6f5"     # mid tier    (blue)
    COL_LARGE = "#c099f0"   # large tier  (purple)
    COL_OUT = "#f0986b"     # output / decode (orange) — the expensive part
    COL_NEUTRAL = "#9aa0a6" # gridlines / input / prefill (grey)
    return COL_LARGE, COL_MID, COL_NEUTRAL, COL_OUT, COL_SMALL


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. The price & speed sheet (illustrative!)

        Below are three made-up model **tiers**. These numbers are **illustrative
        teaching values**, *not* a real price sheet — don't quote them at anyone.
        They're chosen to have realistic *shapes*: bigger models cost more per token
        and emit tokens more slowly, and output always costs several times more than
        input.

        Each tier carries five numbers:

        | field | meaning |
        |---|---|
        | `in_per_1m`  | $ per **1M input** tokens (reading the prompt) |
        | `out_per_1m` | $ per **1M output** tokens (writing the answer) — ~5× input |
        | `decode_ms`  | ms to emit **one output token** (sequential) |
        | `prefill_ms` | ms to read **one input token** (cheap, near-parallel) |
        | `base_ms`    | fixed overhead per request (queue + network) |

        Note `prefill_ms ≪ decode_ms`: reading a token is roughly two orders of
        magnitude faster than writing one. That single inequality drives most of
        what follows.
        """
    )
    return


@app.cell
def _():
    # Three illustrative tiers. NOT a real price list — teaching shapes only.
    TIERS = {
        "small": dict(in_per_1m=0.25, out_per_1m=1.25, decode_ms=5.0,  prefill_ms=0.03, base_ms=150.0),
        "mid":   dict(in_per_1m=1.00, out_per_1m=5.00, decode_ms=12.0, prefill_ms=0.06, base_ms=200.0),
        "large": dict(in_per_1m=5.00, out_per_1m=25.0, decode_ms=30.0, prefill_ms=0.12, base_ms=300.0),
    }

    # When prompt caching hits, cached input tokens are billed at ~0.1x price.
    CACHE_DISCOUNT = 0.1
    return CACHE_DISCOUNT, TIERS


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. The model: two functions

        ### `cost(input_tok, output_tok, tier, cached_fraction)`

        Cost has two terms, and pricing is **per token, but not per token equally**:

        ```
        input_cost  = input_tok  * effective_input_price
        output_cost = output_tok * output_price        # output_price ≈ 5x input
        cost        = input_cost + output_cost
        ```

        **Prompt caching** changes only the *input* side. If a `cached_fraction` of
        your prompt is a prefix the provider has already processed (a fixed system
        prompt, a long document you keep re-sending), those tokens are billed at
        `CACHE_DISCOUNT` (~0.1×) instead of full price:

        ```
        effective_input_price =
            (1 - cached_fraction) * input_price          # fresh tokens, full price
          +      cached_fraction  * input_price * 0.1     # cached tokens, 1/10th
        ```

        Caching does **nothing** for output — you still have to generate every
        answer token from scratch.
        """
    )
    return


@app.cell
def _(CACHE_DISCOUNT, TIERS):
    def cost(input_tok, output_tok, tier="mid", cached_fraction=0.0):
        """Dollars for one request. Prices are per-1M-token; we divide by 1e6.

        cached_fraction in [0,1]: share of INPUT tokens served from the prompt
        cache at CACHE_DISCOUNT (~0.1x) price. Output is never discounted.
        """
        t = TIERS[tier]
        in_price = t["in_per_1m"] / 1e6
        out_price = t["out_per_1m"] / 1e6

        eff_in_price = (
            (1.0 - cached_fraction) * in_price
            + cached_fraction * in_price * CACHE_DISCOUNT
        )
        input_cost = input_tok * eff_in_price
        output_cost = output_tok * out_price
        return input_cost + output_cost

    return (cost,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### `latency(input_tok, output_tok, tier)`

        Latency is **end-to-end wall-clock** for one request, in milliseconds:

        ```
        prefill  = input_tok  * prefill_ms     # read the prompt (fast, ~parallel)
        decode   = output_tok * decode_ms      # write the answer (slow, SEQUENTIAL)
        latency  = base_ms + prefill + decode
        ```

        - `base_ms` is fixed overhead (queueing, network round-trip).
        - **prefill** scales with the prompt but each token is cheap, so even a huge
          prompt adds relatively little.
        - **decode** is the killer: every output token waits for the previous one,
          so this term grows linearly *and* with a steep slope. This is why a 2,000-
          token answer feels so much slower than a 200-token one.

        (`time-to-first-token` ≈ `base_ms + prefill`; everything after that is the
        decode stream trickling out token by token.)
        """
    )
    return


@app.cell
def _(TIERS):
    def latency(input_tok, output_tok, tier="mid"):
        """Wall-clock milliseconds for one request, split into three terms."""
        t = TIERS[tier]
        prefill = input_tok * t["prefill_ms"]
        decode = output_tok * t["decode_ms"]
        return t["base_ms"] + prefill + decode

    def latency_parts(input_tok, output_tok, tier="mid"):
        """Same model, but return the three components for plotting."""
        t = TIERS[tier]
        return dict(
            base=t["base_ms"],
            prefill=input_tok * t["prefill_ms"],
            decode=output_tok * t["decode_ms"],
        )

    return latency, latency_parts


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Worked example: one realistic request

        Let's price and time a single, fairly typical RAG-style call: a **2,000-token
        prompt** (system prompt + a retrieved document + the user question) producing
        a **400-token answer**, on the **mid** tier, with **no caching** yet.
        """
    )
    return


@app.cell
def _(cost, latency_parts, mo):
    ex_in, ex_out, ex_tier = 2000, 400, "mid"

    ex_input_cost = cost(ex_in, 0, ex_tier, 0.0)             # input-only piece
    ex_output_cost = cost(0, ex_out, ex_tier, 0.0)           # output-only piece
    ex_total_cost = ex_input_cost + ex_output_cost

    ex_parts = latency_parts(ex_in, ex_out, ex_tier)
    ex_total_ms = sum(ex_parts.values())

    mo.md(
        f"""
        **Request:** {ex_in:,} input tokens → {ex_out:,} output tokens on `{ex_tier}`.

        **Cost breakdown**

        | piece | tokens | $ | share |
        |---|---:|---:|---:|
        | input  | {ex_in:,} | ${ex_input_cost:.6f} | {100*ex_input_cost/ex_total_cost:.0f}% |
        | output | {ex_out:,} | ${ex_output_cost:.6f} | {100*ex_output_cost/ex_total_cost:.0f}% |
        | **total** | | **${ex_total_cost:.6f}** | 100% |

        Notice the output is only **{ex_out:,}** tokens — one fifth of the input —
        yet it's the larger slice of the bill, because each output token costs ~5×.

        **Latency breakdown**

        | piece | ms | share |
        |---|---:|---:|
        | base (overhead) | {ex_parts['base']:.0f} | {100*ex_parts['base']/ex_total_ms:.0f}% |
        | prefill (read {ex_in:,} in) | {ex_parts['prefill']:.0f} | {100*ex_parts['prefill']/ex_total_ms:.0f}% |
        | decode (write {ex_out:,} out) | {ex_parts['decode']:.0f} | {100*ex_parts['decode']/ex_total_ms:.0f}% |
        | **total** | **{ex_total_ms:.0f} ms** | 100% |

        The prompt is **5× longer** than the answer, but reading it (prefill) is a
        rounding error next to **writing** the answer (decode). Output dominates both
        columns.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. The charts

        Four pictures, one per lever. Read them in order — each isolates one variable
        while holding the rest fixed.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### (a) Cost vs. **output** tokens — output dominates

        Hold the input fixed at a modest 500 tokens and grow only the **answer
        length**. Every tier's bill climbs steeply, because output is the pricey
        side. The takeaway: *how long you let the model talk* is your single biggest
        cost knob.
        """
    )
    return


@app.cell
def _(COL_LARGE, COL_MID, COL_SMALL, cost, np, plt):
    out_axis = np.arange(0, 4001, 50)
    fixed_input_a = 500

    fig_a, ax_a = plt.subplots(figsize=(9, 5))
    for _tier, _col in (("small", COL_SMALL), ("mid", COL_MID), ("large", COL_LARGE)):
        _y = [cost(fixed_input_a, int(o), _tier, 0.0) * 1000 for o in out_axis]
        ax_a.plot(out_axis, _y, color=_col, lw=3, label=f"{_tier} tier")

    ax_a.set_title(
        f"Cost grows with OUTPUT tokens (input fixed at {fixed_input_a})",
        fontweight="bold",
    )
    ax_a.set_xlabel("output tokens (length of the answer)")
    ax_a.set_ylabel("cost per request (× $0.001, i.e. tenths of a cent)")
    ax_a.legend(loc="upper left")
    ax_a.grid(True, alpha=0.25)
    ax_a
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### (b) Cost vs. **input** tokens — caching crushes long prompts

        Now hold the output fixed and grow the **prompt**. We draw each tier twice:
        solid = **no caching**, dashed = **90% of the prompt cached** (a big fixed
        system prompt / reused document). With caching on, the input term nearly
        flattens — long prompts stop hurting once their stable prefix is cached.
        """
    )
    return


@app.cell
def _(COL_LARGE, COL_MID, COL_SMALL, cost, np, plt):
    in_axis = np.arange(0, 32001, 250)
    fixed_output_b = 300
    cached_frac_b = 0.9

    fig_b, ax_b = plt.subplots(figsize=(9, 5))
    for _tier, _col in (("small", COL_SMALL), ("mid", COL_MID), ("large", COL_LARGE)):
        _y_off = [cost(int(i), fixed_output_b, _tier, 0.0) * 1000 for i in in_axis]
        _y_on = [cost(int(i), fixed_output_b, _tier, cached_frac_b) * 1000 for i in in_axis]
        ax_b.plot(in_axis, _y_off, color=_col, lw=3, label=f"{_tier} — cache OFF")
        ax_b.plot(in_axis, _y_on, color=_col, lw=2, ls="--", label=f"{_tier} — cache 90%")

    ax_b.set_title(
        f"Prompt caching flattens INPUT cost (output fixed at {fixed_output_b})",
        fontweight="bold",
    )
    ax_b.set_xlabel("input tokens (length of the prompt)")
    ax_b.set_ylabel("cost per request (× $0.001)")
    ax_b.legend(loc="upper left", fontsize=8, ncol=2)
    ax_b.grid(True, alpha=0.25)
    ax_b
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### (c) Latency: prefill vs. decode as the answer grows

        A stacked area chart on the **mid** tier, with a fixed 2,000-token prompt.
        The grey/blue band (base + prefill) is constant and thin; the orange
        **decode** band swallows the chart as the answer lengthens. Decode is the
        wall-clock you actually feel.
        """
    )
    return


@app.cell
def _(COL_MID, COL_NEUTRAL, COL_OUT, latency_parts, np, plt):
    out_axis_c = np.arange(0, 2001, 25)
    fixed_input_c = 2000
    tier_c = "mid"

    base_band = np.array([latency_parts(fixed_input_c, int(o), tier_c)["base"] for o in out_axis_c])
    prefill_band = np.array([latency_parts(fixed_input_c, int(o), tier_c)["prefill"] for o in out_axis_c])
    decode_band = np.array([latency_parts(fixed_input_c, int(o), tier_c)["decode"] for o in out_axis_c])

    fig_c, ax_c = plt.subplots(figsize=(9, 5))
    ax_c.stackplot(
        out_axis_c,
        base_band / 1000.0,
        prefill_band / 1000.0,
        decode_band / 1000.0,
        labels=["base (overhead)", "prefill (read prompt)", "decode (write answer)"],
        colors=[COL_NEUTRAL, COL_MID, COL_OUT],
    )
    ax_c.set_title(
        f"Decode dominates latency as the answer grows ({tier_c} tier, {fixed_input_c}-tok prompt)",
        fontweight="bold",
    )
    ax_c.set_xlabel("output tokens (length of the answer)")
    ax_c.set_ylabel("latency (seconds)")
    ax_c.legend(loc="upper left")
    ax_c.grid(True, alpha=0.2)
    ax_c
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### (d) Daily cost at scale

        One request's cost looks like nothing — fractions of a cent. Multiply by
        **requests/day** and the tiers (and caching) separate dramatically. Same
        per-request workload (2,000 in → 400 out), 100,000 requests/day. Caching the
        prompt prefix turns into real money at volume.
        """
    )
    return


@app.cell
def _(COL_LARGE, COL_MID, COL_NEUTRAL, COL_OUT, COL_SMALL, cost, np, plt):
    reqs_per_day_d = 100_000
    in_d, out_d = 2000, 400

    tiers_d = ["small", "mid", "large"]
    base_colors_d = [COL_SMALL, COL_MID, COL_LARGE]
    daily_off = [cost(in_d, out_d, t, 0.0) * reqs_per_day_d for t in tiers_d]
    daily_on = [cost(in_d, out_d, t, 0.9) * reqs_per_day_d for t in tiers_d]

    x_d = np.arange(len(tiers_d))
    w_d = 0.38

    fig_d, ax_d = plt.subplots(figsize=(9, 5))
    bars_off = ax_d.bar(x_d - w_d / 2, daily_off, w_d, color=base_colors_d, label="cache OFF")
    bars_on = ax_d.bar(
        x_d + w_d / 2, daily_on, w_d, color=COL_OUT, alpha=0.85, label="cache 90%"
    )
    for _b in list(bars_off) + list(bars_on):
        ax_d.annotate(
            f"${_b.get_height():,.0f}",
            xy=(_b.get_x() + _b.get_width() / 2, _b.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=8,
        )
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([t + " tier" for t in tiers_d])
    ax_d.set_title(
        f"Daily cost = per-request × {reqs_per_day_d:,} req/day  ({in_d} in → {out_d} out)",
        fontweight="bold",
    )
    ax_d.set_ylabel("cost per day ($)")
    ax_d.legend(loc="upper left")
    ax_d.grid(True, axis="y", alpha=0.25)
    _ = COL_NEUTRAL
    ax_d
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Interactive: feel the levers

        Drag the controls and watch the numbers and the little breakdown chart
        update live. Try the moves that matter: push **output tokens** up vs.
        **input tokens** up (output hurts more), flip **prompt caching** on with a
        long prompt, and switch **tier**.
        """
    )
    return


@app.cell
def _(mo):
    in_slider = mo.ui.slider(0, 32000, step=250, value=2000, label="input tokens", show_value=True)
    out_slider = mo.ui.slider(0, 4000, step=50, value=400, label="output tokens", show_value=True)
    reqs_slider = mo.ui.slider(1000, 1_000_000, step=1000, value=100_000, label="requests / day", show_value=True)
    tier_dropdown = mo.ui.dropdown(options=["small", "mid", "large"], value="mid", label="model tier")
    cache_switch = mo.ui.switch(value=False, label="prompt caching (90% of prompt cached)")

    mo.vstack([in_slider, out_slider, reqs_slider, tier_dropdown, cache_switch])
    return cache_switch, in_slider, out_slider, reqs_slider, tier_dropdown


@app.cell
def _(
    COL_MID,
    COL_NEUTRAL,
    COL_OUT,
    cache_switch,
    cost,
    in_slider,
    latency_parts,
    mo,
    out_slider,
    plt,
    reqs_slider,
    tier_dropdown,
):
    live_in = in_slider.value
    live_out = out_slider.value
    live_tier = tier_dropdown.value
    live_cached = 0.9 if cache_switch.value else 0.0
    live_reqs = reqs_slider.value

    live_cost = cost(live_in, live_out, live_tier, live_cached)
    live_in_cost = cost(live_in, 0, live_tier, live_cached)
    live_out_cost = cost(0, live_out, live_tier, 0.0)
    live_day = live_cost * live_reqs

    live_parts = latency_parts(live_in, live_out, live_tier)
    live_ms = sum(live_parts.values())

    # Small live breakdown: cost split (left) and latency split (right).
    fig_live, (ax_cost, ax_lat) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax_cost.bar(["input", "output"], [live_in_cost * 1000, live_out_cost * 1000],
                color=[COL_NEUTRAL, COL_OUT])
    ax_cost.set_title("cost split (× $0.001)", fontsize=10)
    ax_cost.set_ylabel("per request")
    ax_lat.bar(
        ["base", "prefill", "decode"],
        [live_parts["base"], live_parts["prefill"], live_parts["decode"]],
        color=[COL_NEUTRAL, COL_MID, COL_OUT],
    )
    ax_lat.set_title("latency split (ms)", fontsize=10)
    fig_live.tight_layout()

    mo.vstack([
        mo.md(
            f"""
            **cost / request:** ${live_cost:.6f}  •  **cost / day:**
            ${live_day:,.2f}  •  **latency:** {live_ms/1000:.2f} s
            ({live_ms:.0f} ms)

            tier `{live_tier}`, caching **{'ON' if cache_switch.value else 'OFF'}**,
            {live_in:,} in → {live_out:,} out, {live_reqs:,} req/day
            """
        ),
        fig_live,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. The levers, ranked by impact

        When a feature is too slow or too expensive, reach for these roughly in
        order:

        1. **Shorten the output.** It's the dominant term in *both* cost and latency.
           Ask for less (tighter prompts, `max_tokens`, structured/short answers,
           stop sequences). Cutting a 1,000-token answer to 300 is a ~70% win on the
           expensive side.
        2. **Cache the prompt prefix.** If you re-send a fixed system prompt or the
           same document, caching bills the stable part at ~0.1×. Huge for long,
           repetitive prompts; does nothing for output.
        3. **Route to a cheaper tier.** Many requests don't need the big model. Send
           the easy ones to `small`/`mid` and reserve `large` for the hard ones —
           often a 5–20× unit-cost swing.
        4. **Batch / consolidate.** Combine many small jobs into fewer requests to
           amortize the fixed `base_ms` overhead and reuse cached prefixes. Helps
           throughput and latency-per-item, but doesn't shrink the per-token bill.

        Output length first, caching second, routing third, batching fourth. Do the
        top one before the bottom one — it's almost always the bigger lever.
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Q1 — Why do output tokens dominate cost and latency, not input?": mo.md(
                r"""
                Two compounding reasons:

                - **Price.** Output is billed at roughly **5× the input rate**, so even
                  a short answer can outweigh a much longer prompt on the bill.
                - **Mechanics.** Input is read in one near-parallel **prefill** pass
                  (cheap per token), but output is generated **one token at a time** —
                  each new token attends to everything already produced. That's the
                  slow, sequential **decode** loop, and `decode_ms ≫ prefill_ms`.

                So a 2,000-in / 400-out request spends most of its money *and* most of
                its wall-clock on those 400 output tokens.
                """
            ),
            "Q2 — When does prompt caching help the most (and when not at all)?": mo.md(
                r"""
                Caching helps most when a **large, stable prefix** is sent **over and
                over**: a long fixed system prompt, few-shot examples, or a document
                you re-query many times. Then most input tokens bill at ~0.1×, and the
                input term nearly flattens (chart b).

                It helps **little or none** when:

                - the prompt is short (not much to discount),
                - every request's prompt is unique (nothing to reuse), or
                - your cost is **output-bound** — caching never touches output, so a
                  chatty, long-answer feature barely benefits. Shorten the output
                  instead.
                """
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

            - A request's cost = **input** term + **output** term, where output is
              priced ~5× input — and its latency = **base + prefill + decode**, where
              **decode** (one token at a time) dominates.
            - **Output length** is the master lever for both money and speed.
            - **Prompt caching** flattens long-prompt input cost (~0.1× on the cached
              prefix) but does nothing for output.
            - Per-request fractions of a cent become real budgets once multiplied by
              **requests/day** — so model the daily number, not the single call.
            - Ranked levers: shorten output → cache prefix → route to a cheaper tier
              → batch.

            All numbers here were **illustrative**. Swap your provider's real
            per-1M-token prices and measured per-token latency into `TIERS` and the
            same two functions become your actual planning model.

            Next: the matching course chapter — **`../course/p1-inference.html`**.
            """
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
