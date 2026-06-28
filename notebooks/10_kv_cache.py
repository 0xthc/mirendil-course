import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # The KV cache & prompt caching

        Two caches sit between your code and a fast, cheap LLM feature, and they
        are easy to confuse:

        - The **KV cache** lives *inside one generation*. It's the trick that lets a
          model write a 500-token answer without re-reading its own draft 500 times.
          It turns generation from **O(n²)** work into **O(n)** — and it's the thing
          that eats your GPU memory.
        - **Prompt caching** lives *across requests*. It lets request #2 reuse the
          prefill (the KV cache!) that request #1 already paid for, billing the
          shared prefix at ~0.1× instead of full price.

        **What you'll build:**

        1. A toy single-head causal attention in pure **numpy**, then *generation*
           done two ways — naïve (no cache) vs. KV-cached — proving they give the
           **same** output while the cache does far less work.
        2. The work-vs-length and memory-vs-length curves (with a slider), so you can
           see the quadratic-vs-linear gap and why long contexts cost VRAM.
        3. A **prompt-caching cost model** with the real billing multipliers, charted
           to its break-even point, with sliders to feel when it pays off.

        Everything runs **offline** — numpy + matplotlib + stdlib, tiny tensors, no
        torch, no network, no API keys.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** Imagine writing a story one word at a time. To choose each next
            word, you glance back at everything written so far.

            - **Without a KV cache**, every time you add a word you re-read the *whole
              story from the beginning*. Word 100 means re-reading 100 words; word 500
              means re-reading 500. The reading piles up fast.
            - **With a KV cache**, you keep a little **scratchpad** summarising each
              word the moment you write it. To add the next word you only process that
              *one new word* and glance at the scratchpad. No re-reading.

            The KV cache is that scratchpad. It changes nothing about *what* the model
            writes — it just stops it from re-reading the story to write each word.
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
    # ---- Shared palette (consistent across every chart) ----------------------
    COL_NAIVE = "#57b6f5"   # naïve / no-cache (blue)
    COL_NAIVE2 = "#f0986b"  # naïve secondary (orange)
    COL_CACHE = "#5fd38a"   # KV cache / cached path (green)
    COL_GQA = "#c099f0"     # GQA config (purple)
    COL_HILITE = "#f0c674"  # break-even / annotations (amber)
    COL_GREY = "#9aa0a6"    # gridlines / neutral
    return COL_CACHE, COL_GQA, COL_GREY, COL_HILITE, COL_NAIVE, COL_NAIVE2


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — The KV cache (the mechanism)

        ## 1.1 Toy causal attention, in numpy

        Attention, stripped to its bones, is four lines. Given a sequence of token
        vectors `X` with shape `[S, d]` (S tokens, each a d-dim vector):

        1. Project each token into a **query**, **key**, and **value**:
           `Q = X @ Wq`, `K = X @ Wk`, `V = X @ Wv`.
        2. Score every query against every key: `scores = Q @ K.T / sqrt(d)`.
        3. **Causal mask**: token *i* may only look at tokens `≤ i` (no peeking at the
           future), so we set the upper triangle of `scores` to `-inf`.
        4. `softmax` each row into weights, then mix the values: `out = softmax @ V`.

        The two quantities we'll *cache* are **K** and **V** — one key vector and one
        value vector per token. (Q is never cached: each new token brings its own
        query and then is done with it.)
        """
    )
    return


@app.cell
def _(np):
    def softmax_rows(scores):
        """Row-wise softmax, numerically stable."""
        shifted = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    def make_weights(d, seed=0):
        """Fixed random Q/K/V projection matrices [d, d]."""
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d)
        Wq = rng.standard_normal((d, d)) * scale
        Wk = rng.standard_normal((d, d)) * scale
        Wv = rng.standard_normal((d, d)) * scale
        return Wq, Wk, Wv

    def causal_attention(X, Wq, Wk, Wv):
        """Full single-head causal self-attention over X [S, d] -> out [S, d]."""
        S, d = X.shape
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        scores = Q @ K.T / np.sqrt(d)
        future = np.triu(np.ones((S, S), dtype=bool), k=1)  # strict upper triangle
        scores = np.where(future, -np.inf, scores)
        A = softmax_rows(scores)
        return A @ V

    return causal_attention, make_weights, softmax_rows


@app.cell
def _(causal_attention, make_weights, mo, np):
    # A tiny, deterministic sequence to play with.
    _rng = np.random.default_rng(42)
    d_demo = 8           # head dimension
    S_demo = 6           # sequence length
    X_demo = _rng.standard_normal((S_demo, d_demo))
    Wq_demo, Wk_demo, Wv_demo = make_weights(d_demo, seed=0)

    out_demo = causal_attention(X_demo, Wq_demo, Wk_demo, Wv_demo)

    mo.md(
        f"""
        We built a sequence of **{S_demo} tokens**, each a **{d_demo}-dim** vector,
        and ran one full causal-attention pass:

        ```
        X   shape = {tuple(X_demo.shape)}   (the input tokens)
        out shape = {tuple(out_demo.shape)}   (one context vector per token)
        ```

        `out[i]` is token *i*'s view of the sequence — a value-mixture over tokens
        `0..i` only. This whole-sequence pass is exactly what happens during
        **prefill** (reading the prompt). Generation is what comes next, and it's
        where the cache earns its keep.
        """
    )
    return S_demo, Wk_demo, Wq_demo, Wv_demo, X_demo, d_demo


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1.2 Generation, two ways

        Generation is **autoregressive**: produce one token, append it, produce the
        next, and so on. The output for the brand-new token at position *t* is just
        attention of *its* query over the keys/values of tokens `0..t`.

        We'll simulate this over a known sequence, revealing it one token at a time,
        and compute each new token's output **two ways** — counting the work as the
        number of **K/V vectors we run through the projections** (the canonical thing
        the cache saves):

        - **(a) Naïve / no cache** — at every step, re-project **K, V for *all*
          tokens `0..t`** from scratch. Step *t* recomputes `t+1` key/value pairs.
        - **(b) KV cache** — keep the past K, V around. At step *t*, project **only
          the one new token**, append it to the cache, and attend. Step *t*
          recomputes `1` pair.

        Then we assert the two paths produce the **same outputs** — so the cache is a
        pure optimisation, never a change in behaviour.
        """
    )
    return


@app.cell
def _(np, softmax_rows):
    def generate_naive(X, Wq, Wk, Wv):
        """No cache: re-project K,V for the WHOLE prefix at every step."""
        S, d = X.shape
        outs = np.zeros((S, d))
        work_per_step = np.zeros(S, dtype=np.int64)
        for t in range(S):
            prefix = X[: t + 1]                 # tokens 0..t
            Q = prefix @ Wq
            K = prefix @ Wk                     # <- recomputed every step
            V = prefix @ Wv                     # <- recomputed every step
            scores = Q @ K.T / np.sqrt(d)
            future = np.triu(np.ones((t + 1, t + 1), dtype=bool), k=1)
            scores = np.where(future, -np.inf, scores)
            full_out = softmax_rows(scores) @ V
            outs[t] = full_out[t]               # keep only the new token's row
            work_per_step[t] = t + 1            # projected K,V for t+1 tokens
        return outs, work_per_step

    def generate_cached(X, Wq, Wk, Wv):
        """KV cache: project K,V for only the NEW token, reuse the rest."""
        S, d = X.shape
        outs = np.zeros((S, d))
        work_per_step = np.zeros(S, dtype=np.int64)
        K_cache = np.zeros((0, d))
        V_cache = np.zeros((0, d))
        for t in range(S):
            new_tok = X[t : t + 1]              # just the one new token [1, d]
            q = new_tok @ Wq
            k = new_tok @ Wk                    # <- one projection only
            v = new_tok @ Wv                    # <- one projection only
            K_cache = np.vstack([K_cache, k])   # append to the scratchpad
            V_cache = np.vstack([V_cache, v])
            scores = q @ K_cache.T / np.sqrt(d)  # [1, t+1] — no mask needed
            outs[t] = (softmax_rows(scores) @ V_cache)[0]
            work_per_step[t] = 1                 # projected K,V for 1 token
        return outs, work_per_step

    return generate_cached, generate_naive


@app.cell
def _(
    Wk_demo,
    Wq_demo,
    Wv_demo,
    X_demo,
    generate_cached,
    generate_naive,
    np,
):
    out_naive, work_naive_demo = generate_naive(X_demo, Wq_demo, Wk_demo, Wv_demo)
    out_cached, work_cached_demo = generate_cached(X_demo, Wq_demo, Wk_demo, Wv_demo)

    same_outputs = np.allclose(out_naive, out_cached, atol=1e-12)
    assert same_outputs, "KV cache changed the output — that must never happen!"

    max_abs_diff = float(np.max(np.abs(out_naive - out_cached)))
    return max_abs_diff, out_cached, out_naive, same_outputs


@app.cell
def _(max_abs_diff, mo, same_outputs, work_cached_total_str, work_naive_total_str):
    mo.callout(
        mo.md(
            f"""
            ✅ **Verified: the KV cache is a pure optimisation.**

            Naïve generation and KV-cached generation produced **identical** outputs
            (`np.allclose` passed; max abs difference = `{max_abs_diff:.2e}` — just
            floating-point dust). Over this 6-token run the naïve path re-projected
            **{work_naive_total_str}** key/value vectors; the cache re-projected
            **{work_cached_total_str}**. Same answer, a fraction of the work.
            """
        ),
        kind="success" if same_outputs else "danger",
    )
    return


@app.cell
def _(work_cached_demo, work_naive_demo):
    # Stringify totals for the success callout above (defined here so the callout
    # cell stays a single mo.callout expression).
    work_naive_total_str = f"{int(work_naive_demo.sum())}"
    work_cached_total_str = f"{int(work_cached_demo.sum())}"
    return work_cached_total_str, work_naive_total_str


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1.3 The cost curve: O(n²) vs. O(n)

        Step *t* costs `t+1` for the naïve path and `1` for the cache. Summed over a
        sequence of length *n*:

        - **Naïve:** `1 + 2 + 3 + … + n = n(n+1)/2` → grows like **n²** (quadratic).
        - **Cached:** `1 + 1 + … + 1 = n` → grows like **n** (linear).

        That gap *is* the reason long-context generation is feasible at all. The left
        chart shows cumulative work; the right chart shows per-step work (naïve climbs
        with every token; the cache stays flat).
        """
    )
    return


@app.cell
def _(COL_CACHE, COL_NAIVE, COL_NAIVE2, np, plt):
    n_axis = np.arange(1, 257)                       # sequence lengths 1..256
    cum_naive = np.cumsum(n_axis)                    # n(n+1)/2  -> O(n^2)
    cum_cached = np.cumsum(np.ones_like(n_axis))     # n         -> O(n)
    per_step_naive = n_axis                          # t+1 at each step
    per_step_cached = np.ones_like(n_axis)           # 1 at each step

    fig_work, (ax_cum, ax_step) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_cum.plot(n_axis, cum_naive, color=COL_NAIVE, lw=3, label="naïve  O(n²)")
    ax_cum.plot(n_axis, cum_cached, color=COL_CACHE, lw=3, label="KV cache  O(n)")
    ax_cum.fill_between(n_axis, cum_cached, cum_naive, color=COL_NAIVE2, alpha=0.15)
    ax_cum.set_title("Cumulative K/V projections", fontweight="bold")
    ax_cum.set_xlabel("sequence length n (tokens generated)")
    ax_cum.set_ylabel("total K/V vectors projected")
    ax_cum.legend(loc="upper left")
    ax_cum.grid(True, alpha=0.25)
    ax_cum.annotate(
        "quadratic blow-up\n(the work you avoid)",
        xy=(200, cum_naive[199]), xytext=(40, cum_naive[199] * 0.9),
        fontsize=9, color=COL_NAIVE2,
    )

    ax_step.plot(n_axis, per_step_naive, color=COL_NAIVE, lw=3, label="naïve: t+1 / step")
    ax_step.plot(n_axis, per_step_cached, color=COL_CACHE, lw=3, label="cache: 1 / step")
    ax_step.set_title("Per-step work", fontweight="bold")
    ax_step.set_xlabel("step (token position t)")
    ax_step.set_ylabel("K/V vectors projected this step")
    ax_step.legend(loc="upper left")
    ax_step.grid(True, alpha=0.25)

    fig_work.tight_layout()
    ax_cum
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1.4 The catch: the cache costs **memory**

        The cache isn't free — you trade compute for **VRAM**. Every token keeps a
        key *and* a value, for **every layer** and **every KV head**. The size is a
        single multiply:

        ```
        kv_bytes = 2 · n_layers · n_kv_heads · head_dim · seq · bytes_per_elem
                   ▲
                   └─ the 2 is for K *and* V
        ```

        This is why **Grouped-Query Attention (GQA)** exists: instead of one KV head
        per query head (Multi-Head Attention, MHA), several query heads *share* one
        KV head. Fewer `n_kv_heads` → a proportionally smaller cache, for nearly the
        same quality. Below we compare a 32-head MHA model against an 8-KV-head GQA
        model (same 32 layers, 128 head-dim, fp16).
        """
    )
    return


@app.cell
def _(COL_GQA, COL_GREY, COL_NAIVE, np, plt):
    def kv_bytes(n_layers, n_kv_heads, head_dim, seq, bytes_per_elem=2):
        """Bytes held in the KV cache. The leading 2 covers K and V."""
        return 2 * n_layers * n_kv_heads * head_dim * seq * bytes_per_elem

    N_LAYERS = 32
    HEAD_DIM = 128
    seq_axis = np.arange(0, 131_072 + 1, 2048)        # 0 .. 128K tokens

    gb_mha = np.array([kv_bytes(N_LAYERS, 32, HEAD_DIM, s) for s in seq_axis]) / 1e9
    gb_gqa = np.array([kv_bytes(N_LAYERS, 8, HEAD_DIM, s) for s in seq_axis]) / 1e9

    fig_mem, ax_mem = plt.subplots(figsize=(9, 4.6))
    ax_mem.plot(seq_axis / 1024, gb_mha, color=COL_NAIVE, lw=3, label="MHA — 32 KV heads")
    ax_mem.plot(seq_axis / 1024, gb_gqa, color=COL_GQA, lw=3, label="GQA — 8 KV heads")
    ax_mem.fill_between(seq_axis / 1024, gb_gqa, gb_mha, color=COL_GREY, alpha=0.12)
    ax_mem.set_title(
        "KV cache size grows linearly with context (fp16, 32 layers, head_dim 128)",
        fontweight="bold",
    )
    ax_mem.set_xlabel("sequence length (K tokens)")
    ax_mem.set_ylabel("KV cache size (GB)")
    ax_mem.legend(loc="upper left")
    ax_mem.grid(True, alpha=0.25)
    _k = 64  # annotate the 64K point
    _i = np.argmin(np.abs(seq_axis - _k * 1024))
    ax_mem.annotate(
        f"at 64K: {gb_mha[_i]:.1f} GB (MHA)\nvs {gb_gqa[_i]:.1f} GB (GQA) — 4× smaller",
        xy=(64, gb_mha[_i]), xytext=(8, gb_mha[_i] * 0.78),
        fontsize=9, color=COL_GQA,
        arrowprops=dict(arrowstyle="->", color=COL_GREY),
    )
    ax_mem
    return (kv_bytes,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Why this matters for throughput.** The KV cache shares VRAM with the
            model weights. Whatever's left after loading weights is divided among the
            KV caches of all *concurrent* requests — so a bigger per-request cache
            means a **smaller batch size** and **lower throughput**. GQA (and tricks
            like paged attention / quantised KV) exist precisely to fit more, longer
            requests into the same card. Cache size *is* a capacity-planning number.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1.5 Interactive: feel both curves

        Drag **max sequence length** to rescale the work-vs-length chart, and drag
        **KV heads** to resize the memory curve (fewer heads = GQA = a smaller cache).
        Both charts redraw live.
        """
    )
    return


@app.cell
def _(mo):
    seqlen_slider = mo.ui.slider(
        16, 4096, step=16, value=512, label="max sequence length (work chart)",
        show_value=True,
    )
    kvheads_slider = mo.ui.slider(
        1, 32, step=1, value=8, label="KV heads (memory chart: 32=MHA, fewer=GQA)",
        show_value=True,
    )
    mo.vstack([seqlen_slider, kvheads_slider])
    return kvheads_slider, seqlen_slider


@app.cell
def _(
    COL_CACHE,
    COL_GQA,
    COL_NAIVE,
    COL_NAIVE2,
    kv_bytes,
    kvheads_slider,
    mo,
    np,
    plt,
    seqlen_slider,
):
    live_n = seqlen_slider.value
    live_heads = kvheads_slider.value

    n_ax = np.arange(1, live_n + 1)
    cum_naive_live = np.cumsum(n_ax)
    cum_cached_live = np.cumsum(np.ones_like(n_ax))
    speedup = cum_naive_live[-1] / cum_cached_live[-1]

    seq_ax = np.arange(0, 131_072 + 1, 2048)
    gb_live = np.array([kv_bytes(32, live_heads, 128, s) for s in seq_ax]) / 1e9
    gb_mha_ref = np.array([kv_bytes(32, 32, 128, s) for s in seq_ax]) / 1e9

    fig_live1, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    axL.plot(n_ax, cum_naive_live, color=COL_NAIVE, lw=3, label="naïve O(n²)")
    axL.plot(n_ax, cum_cached_live, color=COL_CACHE, lw=3, label="cache O(n)")
    axL.fill_between(n_ax, cum_cached_live, cum_naive_live, color=COL_NAIVE2, alpha=0.15)
    axL.set_title(f"Cumulative work up to n={live_n}", fontweight="bold")
    axL.set_xlabel("sequence length")
    axL.set_ylabel("K/V vectors projected")
    axL.legend(loc="upper left")
    axL.grid(True, alpha=0.25)

    axR.plot(seq_ax / 1024, gb_mha_ref, color="#cfd3d6", lw=2, ls="--", label="MHA (32) ref")
    axR.plot(seq_ax / 1024, gb_live, color=COL_GQA, lw=3, label=f"{live_heads} KV heads")
    axR.set_title("KV cache size vs context", fontweight="bold")
    axR.set_xlabel("sequence length (K tokens)")
    axR.set_ylabel("KV cache (GB)")
    axR.legend(loc="upper left")
    axR.grid(True, alpha=0.25)
    fig_live1.tight_layout()

    mo.vstack([
        mo.md(
            f"""
            At **n = {live_n}**, naïve does **{int(cum_naive_live[-1]):,}** K/V
            projections vs the cache's **{int(cum_cached_live[-1]):,}** — a
            **{speedup:.0f}×** reduction. With **{live_heads} KV heads**, a 128K
            context needs **{gb_live[-1]:.1f} GB** of cache (MHA-32 would need
            **{gb_mha_ref[-1]:.1f} GB**).
            """
        ),
        fig_live1,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — Prompt caching (the cost model)

        ## 2.1 What it is

        The KV cache from Part 1 is built fresh during **prefill** — the model reads
        your prompt and fills the cache before it writes a single token. **Prompt
        caching** says: if the *next* request starts with the **same prefix** (a fixed
        system prompt, tool definitions, a long document, few-shot examples), don't
        re-prefill it — **reuse the KV cache from last time**.

        You pay a small premium the first time to **write** the prefix into the cache,
        and then a deep discount to **read** it on every later request.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** It's like a barista who memorises your "usual." The first time
            you order, explaining it takes a moment longer (the **write** premium).
            Every time after, you just say "the usual" and it's instant and cheap (the
            **read** discount). If you change your order — even slightly — they have to
            learn it again from that point on.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2.2 The billing model

        Costs are quoted as multipliers of the **base input price** (per token):

        | line item | multiplier | when |
        |---|---|---|
        | uncached input | **1.0×** | normal input, no caching |
        | cache **write** | **1.25×** | writing the prefix, 5-minute TTL |
        | cache **write** | **2.0×** | writing the prefix, 1-hour TTL |
        | cache **read** | **0.1×** | reading a cached prefix (a 10× discount) |

        So for a request whose prompt = **prefix** (stable) + **suffix** (the variable
        bit, e.g. the user's question):

        - **Request #1** pays `write_mult × prefix + 1.0 × suffix` — it fills the cache.
        - **Requests #2…N** pay `0.1 × prefix + 1.0 × suffix` — they hit the cache.
        - **Uncached baseline** always pays `1.0 × (prefix + suffix)`.

        (Each later request also *refreshes* the TTL, so a steady stream of traffic
        keeps the entry warm. The suffix is never cached — it changes every time.)
        """
    )
    return


@app.cell
def _(np):
    WRITE_MULT = {"5-min": 1.25, "1-hour": 2.0}
    READ_MULT = 0.1
    BASE_PRICE = 3.0 / 1e6   # illustrative $/input-token; ratios are what matter

    def cost_over_requests(n_requests, prefix_tokens, suffix_tokens, ttl="5-min"):
        """Cumulative $ for cached vs uncached over a run of identical-prefix calls."""
        write_mult = WRITE_MULT[ttl]
        cached = np.empty(n_requests)
        uncached = np.empty(n_requests)
        for i in range(n_requests):
            if i == 0:
                per_cached = write_mult * prefix_tokens + 1.0 * suffix_tokens
            else:
                per_cached = READ_MULT * prefix_tokens + 1.0 * suffix_tokens
            cached[i] = per_cached
            uncached[i] = 1.0 * (prefix_tokens + suffix_tokens)
        return np.cumsum(cached) * BASE_PRICE, np.cumsum(uncached) * BASE_PRICE

    def break_even_request(cum_cached, cum_uncached):
        """1-based index of the first request where cached total < uncached total."""
        hits = np.where(cum_cached < cum_uncached)[0]
        return int(hits[0] + 1) if len(hits) else None

    return BASE_PRICE, cost_over_requests, break_even_request


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2.3 The break-even chart

        With a big reused prefix, caching loses on request #1 (you paid the write
        premium) but pulls ahead almost immediately. Here: a **4,000-token prefix**
        (system prompt + document) and a **200-token suffix** (the question), over 12
        requests. We plot cumulative cost for the uncached baseline and both TTLs, and
        mark where each crosses below the baseline.
        """
    )
    return


@app.cell
def _(
    COL_CACHE,
    COL_GQA,
    COL_HILITE,
    COL_NAIVE2,
    break_even_request,
    cost_over_requests,
    np,
    plt,
):
    n_req_demo = 12
    prefix_demo = 4000
    suffix_demo = 200

    cum_unc, _ = cost_over_requests(n_req_demo, prefix_demo, suffix_demo, "5-min")
    cum_5min, _u1 = cost_over_requests(n_req_demo, prefix_demo, suffix_demo, "5-min")
    cum_1hr, _u2 = cost_over_requests(n_req_demo, prefix_demo, suffix_demo, "1-hour")
    # uncached baseline is identical for both calls; recompute cleanly:
    _, cum_baseline = cost_over_requests(n_req_demo, prefix_demo, suffix_demo, "5-min")

    be_5min = break_even_request(cum_5min, cum_baseline)
    be_1hr = break_even_request(cum_1hr, cum_baseline)

    req_x = np.arange(1, n_req_demo + 1)
    fig_be, ax_be = plt.subplots(figsize=(9, 5))
    ax_be.plot(req_x, cum_baseline, color=COL_NAIVE2, lw=3, marker="o", label="uncached (1.0× always)")
    ax_be.plot(req_x, cum_5min, color=COL_CACHE, lw=3, marker="o", label="cached, 5-min TTL (write 1.25×)")
    ax_be.plot(req_x, cum_1hr, color=COL_GQA, lw=3, marker="o", label="cached, 1-hour TTL (write 2.0×)")

    for _be, _series, _col, _lab in (
        (be_5min, cum_5min, COL_CACHE, "5-min"),
        (be_1hr, cum_1hr, COL_GQA, "1-hour"),
    ):
        if _be is not None:
            ax_be.axvline(_be, color=_col, ls=":", alpha=0.6)
            ax_be.annotate(
                f"{_lab} break-even\n@ request {_be}",
                xy=(_be, _series[_be - 1]),
                xytext=(_be + 0.3, _series[_be - 1] + cum_baseline[-1] * 0.06),
                fontsize=8, color=_col,
            )

    ax_be.set_title(
        f"Prompt caching break-even (prefix={prefix_demo:,}, suffix={suffix_demo})",
        fontweight="bold",
    )
    ax_be.set_xlabel("number of requests (same prefix)")
    ax_be.set_ylabel("cumulative cost ($)")
    ax_be.legend(loc="upper left")
    ax_be.grid(True, alpha=0.25)
    _ = COL_HILITE
    ax_be
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Note the crossovers: the **5-minute** TTL (cheaper write) breaks even by about
        the **2nd** request, the **1-hour** TTL (pricier write, but survives longer
        gaps between calls) by about the **3rd**. After that, both pull steadily away
        from the uncached line — the slope is gentler because the prefix now bills at
        0.1× on every request.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2.4 Interactive: when does it pay off?

        Set the **prefix** and **suffix** sizes, the **number of requests**, and the
        **TTL**. The chart redraws and the readout reports total spend, dollars saved,
        and the percentage saved. Try a *tiny* prefix (caching barely helps) vs a
        *huge* one (caching wins almost immediately); and a *long suffix* (output-ish
        cost the cache can't touch).
        """
    )
    return


@app.cell
def _(mo):
    prefix_slider = mo.ui.slider(0, 16000, step=250, value=4000, label="prefix tokens (stable, cacheable)", show_value=True)
    suffix_slider = mo.ui.slider(0, 4000, step=50, value=200, label="suffix tokens (varies per request)", show_value=True)
    nreq_slider = mo.ui.slider(1, 50, step=1, value=10, label="number of requests", show_value=True)
    ttl_dropdown = mo.ui.dropdown(options=["5-min", "1-hour"], value="5-min", label="cache TTL")
    mo.vstack([prefix_slider, suffix_slider, nreq_slider, ttl_dropdown])
    return nreq_slider, prefix_slider, suffix_slider, ttl_dropdown


@app.cell
def _(
    COL_CACHE,
    COL_NAIVE2,
    cost_over_requests,
    mo,
    nreq_slider,
    np,
    plt,
    prefix_slider,
    suffix_slider,
    ttl_dropdown,
):
    p_val = prefix_slider.value
    s_val = suffix_slider.value
    n_val = nreq_slider.value
    ttl_val = ttl_dropdown.value

    cum_c, cum_u = cost_over_requests(n_val, p_val, s_val, ttl_val)
    total_c = float(cum_c[-1])
    total_u = float(cum_u[-1])
    saved = total_u - total_c
    pct = (saved / total_u * 100) if total_u > 0 else 0.0

    req_axis = np.arange(1, n_val + 1)
    fig_int, ax_int = plt.subplots(figsize=(9, 4.6))
    ax_int.plot(req_axis, cum_u, color=COL_NAIVE2, lw=3, marker="o", label="uncached")
    ax_int.plot(req_axis, cum_c, color=COL_CACHE, lw=3, marker="o", label=f"cached ({ttl_val})")
    ax_int.fill_between(req_axis, cum_c, cum_u, color=COL_CACHE, alpha=0.12)
    ax_int.set_title("Cumulative cost: cached vs uncached", fontweight="bold")
    ax_int.set_xlabel("number of requests")
    ax_int.set_ylabel("cumulative cost ($)")
    ax_int.legend(loc="upper left")
    ax_int.grid(True, alpha=0.25)

    _verdict = "saves money 🎉" if saved > 0 else "does **not** pay off here"
    mo.vstack([
        mo.md(
            f"""
            **prefix {p_val:,} · suffix {s_val:,} · {n_val} requests · {ttl_val} TTL**

            - uncached total: **${total_u:.4f}**
            - cached total: **${total_c:.4f}**
            - saved: **${saved:.4f}**  →  **{pct:.1f}%** — caching {_verdict}
            """
        ),
        fig_int,
    ])
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Gotchas that quietly kill your hit rate.**

            - **Prefix-exact matching.** The cache keys on the *literal* token prefix.
              A timestamp, a UUID, a "Today is …" line, or **unsorted JSON** (key order
              wobbles) near the *start* of the prompt busts the match for everything
              after it. Put volatile content **last**, keep the stable prefix
              byte-for-byte identical.
            - **Minimum cacheable length.** Providers only cache a prefix above some
              floor (often ~1K–4K tokens, model-dependent). Below that, nothing caches
              and you just pay 1.0×.
            - **TTL expiry.** The entry lives ~5 min (or ~1 hour) since last *use*. A
              burst of traffic stays warm; sparse traffic lets it lapse and you re-pay
              the write premium.
            - **Verify, don't assume.** Check the response usage —
              `cache_read_input_tokens` (and `cache_creation_input_tokens`) — to
              confirm you're actually hitting the cache, not silently paying full price.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Q1 — Why is naïve generation O(n²) and the KV cache O(n)?": mo.md(
                r"""
                Generating token *t* needs the keys/values of all tokens `0..t`.
                **Without** a cache you re-project K and V for the *entire* prefix at
                every step, so step *t* costs `t+1` and the run costs
                `1+2+…+n = n(n+1)/2` → **O(n²)**. **With** a cache you keep the past
                K/V and only project the **one new token** each step — `1` per step,
                `n` total → **O(n)**. Same outputs (we asserted it with
                `np.allclose`); the cache just refuses to redo work it already did.
                """
            ),
            "Q2 — When does prompt caching pay off, and what invalidates it?": mo.md(
                r"""
                It pays off when a **large, stable prefix** is reused across **many**
                requests: you eat a one-time write premium (1.25× or 2.0×) and then
                bill that prefix at **0.1×** forever after — break-even lands around
                request **2** (5-min) or **3** (1-hour). It **doesn't** help when the
                prefix is tiny (below the cacheable floor), unique per request, or when
                your cost is dominated by the **suffix/output** (never cached).
                Invalidated by: any change to the prefix bytes (timestamps, UUIDs,
                unsorted-JSON key order), and by **TTL expiry** between calls.
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

            - The **KV cache** stores each token's key & value so generation processes
              only the *new* token per step: **O(n²) → O(n)** work, with **identical**
              outputs (verified). It's a pure speed/compute optimisation.
            - It costs **memory**: `2 · n_layers · n_kv_heads · head_dim · seq · bytes`.
              That VRAM competes with the weights and caps batch size / throughput —
              which is why **GQA** (fewer KV heads) and paged/quantised KV exist.
            - **Prompt caching** reuses a prior request's prefill across requests:
              **write** the prefix once (1.25× / 2.0×), then **read** it at **0.1×**.
              Break-even is ~2–3 requests for a big reused prefix.
            - Caching only helps a **stable, exact, long-enough** prefix sent
              **repeatedly within the TTL** — verify with `cache_read_input_tokens`.

            Next, the matching course chapters:
            **`../course/p1b-kv-cache.html`** and **`../course/p1-inference.html`**.
            """
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
