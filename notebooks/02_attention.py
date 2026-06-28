import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 02 · Attention, built from scratch

        This notebook builds **scaled dot-product attention** — the engine inside every
        transformer — one tensor at a time, on tiny CPU tensors you can actually read.
        We do multi-head attention *and* the causal mask, then prove our hand-rolled
        version matches PyTorch's optimized `scaled_dot_product_attention`.

        **What you'll build, step by step:**

        1. The `Q`, `K`, `V` projections and the `[B, heads, S, head_dim]` layout.
        2. Raw attention **scores** = `Q @ Kᵀ / √head_dim` (with a heatmap).
        3. The **causal mask** (no peeking at the future) — and what it does to the scores.
        4. **Softmax** → attention weights (each row sums to 1).
        5. The weighted sum of `V`, concatenating heads, and the output projection `W_o`.
        6. A **multi-head grid** so you can see heads attend differently.
        7. An **interactive** explorer: pick a query token, toggle the mask, watch it react.
        8. A correctness check against `F.scaled_dot_product_attention(..., is_causal=True)`.

        You are a full-stack engineer, not a data scientist — so we trace **shapes and
        data flow**, not the theory of why attention works. If you can follow a tensor
        changing shape, you can follow this whole notebook.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    return mo, np, torch, F, plt


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** Each word in a sentence raises its hand and asks the room:
            *"who here is relevant to me?"* Every other word answers, and the asking
            word listens **more** to the relevant answers and **less** to the rest —
            then updates its own understanding. Do that for every word at once and you
            have attention.

            **Causal** just means you may only ask words that came **before** you —
            no peeking at the future. (That's the rule that lets a model *generate*
            text left to right.)
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 0 · The cast of shapes

        Almost every confusion in attention is really a *shape* confusion. Five letters
        carry the whole story:

        | symbol | meaning | our tiny value |
        |---|---|---|
        | `B` | batch size — how many sequences at once | **1** |
        | `S` | sequence length — tokens per sequence | **8** |
        | `H` | hidden size — width of each token's vector | **16** |
        | `num_heads` | how many parallel attention "heads" | **4** |
        | `head_dim` | width each head works in, always `H / num_heads` | **4** |

        Our input `X` has shape `[B, S, H]`: a batch of `B` sequences, each `S` tokens
        long, each token a vector of `H` numbers. That single tensor flows through the
        whole block and comes back out **the same shape** — that invariance is what lets
        you stack dozens of these blocks.
        """
    )
    return


@app.cell
def _(mo, torch):
    B = 1
    S = 8
    H = 16
    num_heads = 4
    head_dim = H // num_heads  # = 4

    # Our toy input: one sequence of 8 tokens, each a 16-wide vector.
    X = torch.randn(B, S, H)

    mo.md(
        f"""
        Concrete numbers for this notebook:

        ```
        B={B}  S={S}  H={H}  num_heads={num_heads}  head_dim = H/num_heads = {head_dim}
        X.shape = {tuple(X.shape)}   # [B, S, H]
        ```

        Sanity check: `num_heads × head_dim = {num_heads} × {head_dim} = {num_heads * head_dim} = H`. ✅
        """
    )
    return B, S, H, num_heads, head_dim, X


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1 · Project X into Query, Key, Value

        Three separate weight matrices — `W_q`, `W_k`, `W_v`, each `[H, H]` — turn every
        token into three new vectors:

        - **Query** (`Q`): *"what am I looking for?"*
        - **Key** (`K`): *"what do I offer?"*
        - **Value** (`V`): *"what information do I carry?"*

        Each projection is just a matrix multiply. We use `F.linear(X, W)`, which computes
        `X @ W.T` (PyTorch stores weights **output-first**, as `[out_features, in_features]`,
        and transposes for you). With square `[H, H]` weights the shape is unchanged:
        `[B, S, H] → [B, S, H]`.

        Then we **split the width `H` into `num_heads` lanes**: reshape `[B, S, H]` into
        `[B, S, num_heads, head_dim]` and `transpose(1, 2)` to get `[B, num_heads, S, head_dim]`.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Gotcha — why `[B, heads, S, head_dim]` and not `[B, S, heads, head_dim]`?**
            Because PyTorch's `scaled_dot_product_attention` expects the **head axis right
            after the batch axis**. You naturally *build* the tensor as
            `[B, S, heads, head_dim]`, then `transpose(1, 2)` to swap `S` and `heads`.
            The head axis becomes a "batch-like" dimension the attention op processes in
            parallel — every head does its own independent attention.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(F, torch, X, B, S, H, num_heads, head_dim):
    # Weight matrices, stored [out_features, in_features] = [H, H].
    # (1/sqrt(fan_in) scaling keeps the numbers small and readable.)
    scale = H**-0.5
    W_q = torch.randn(H, H) * scale
    W_k = torch.randn(H, H) * scale
    W_v = torch.randn(H, H) * scale
    W_o = torch.randn(H, H) * scale

    def project_to_heads(projected, n_heads, d_head):
        # [B, S, H] -> [B, S, n_heads, d_head] -> [B, n_heads, S, d_head]
        b, s, _ = projected.shape
        return projected.view(b, s, n_heads, d_head).transpose(1, 2)

    # Project, then split into heads. q,k,v: [B, num_heads, S, head_dim]
    Q = project_to_heads(F.linear(X, W_q), num_heads, head_dim)
    K = project_to_heads(F.linear(X, W_k), num_heads, head_dim)
    V = project_to_heads(F.linear(X, W_v), num_heads, head_dim)
    return W_q, W_k, W_v, W_o, project_to_heads, Q, K, V


@app.cell
def _(mo, X, Q, K, V, F, W_q):
    mo.md(
        f"""
        Watch the shapes flow:

        ```
        X                      : {tuple(X.shape)}          # [B, S, H]
        F.linear(X, W_q)       : {tuple(F.linear(X, W_q).shape)}          # [B, S, H]  (a plain matmul)
        Q = split into heads   : {tuple(Q.shape)}       # [B, num_heads, S, head_dim]
        K                      : {tuple(K.shape)}       # same
        V                      : {tuple(V.shape)}       # same
        ```

        The `H=16` width got sliced into `num_heads=4` lanes of `head_dim=4`. Each head
        now holds its own little 4-dimensional view of every token.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2 · The attention scores

        For each head, every **query** token is compared against every **key** token by a
        dot product. Big dot product = "these two are relevant to each other". We divide by
        `√head_dim` to keep the numbers from blowing up as `head_dim` grows (that's the
        "scaled" in *scaled dot-product attention*):

        ```
        scores = Q @ Kᵀ / √head_dim        # [B, num_heads, S, S]
        ```

        The result is an `S × S` grid **per head**: `scores[q, k]` is how much query token
        `q` cares about key token `k`. Let's compute it and look at head 0's grid as a heatmap.
        """
    )
    return


@app.cell
def _(Q, K, head_dim, torch):
    # scores[..., q, k] = how much query q attends to key k, for each head.
    # Q,K: [B, heads, S, head_dim] -> scores: [B, heads, S, S]
    scores = (Q @ K.transpose(-2, -1)) / (head_dim**0.5)
    return (scores,)


@app.cell
def _(scores, plt):
    fig_raw, ax_raw = plt.subplots(figsize=(5, 4.2))
    im_raw = ax_raw.imshow(scores[0, 0].detach().numpy(), cmap="magma")
    ax_raw.set_title("Raw attention scores — head 0\n(before masking or softmax)")
    ax_raw.set_xlabel("key token (what I look at)")
    ax_raw.set_ylabel("query token (who is asking)")
    ax_raw.set_xticks(range(scores.shape[-1]))
    ax_raw.set_yticks(range(scores.shape[-2]))
    fig_raw.colorbar(im_raw, ax=ax_raw, label="score (pre-softmax)")
    fig_raw.tight_layout()
    ax_raw
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Notice the grid is **fully filled in** — right now every query can see every key,
        *including keys in the future* (the upper-right triangle, where `key > query`).
        For a model that generates text left-to-right, that's cheating. Time to fix it.

        ## 3 · The causal mask (no peeking at the future)

        **Causal** attention forbids a query at position `i` from looking at any key at
        position `j > i`. We build a boolean mask that is `True` for those forbidden
        future positions (the strict upper triangle), then set those scores to `-∞`.
        Why `-∞`? Because the next step is a softmax, and `softmax(-∞) = 0` — those keys
        contribute exactly nothing.
        """
    )
    return


@app.cell
def _(scores, S, torch):
    # True where key is in the *future* of the query (strict upper triangle).
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

    # Apply it: forbidden positions become -inf so softmax sends them to 0.
    scores_masked = scores.masked_fill(causal_mask, float("-inf"))
    return causal_mask, scores_masked


@app.cell
def _(scores_masked, np, plt):
    # Show head 0 after masking. -inf can't be drawn, so paint those cells as a
    # distinct "blocked" color via the colormap's "bad" value.
    grid_masked = scores_masked[0, 0].detach().numpy().copy()
    grid_masked[np.isinf(grid_masked)] = np.nan

    cmap_masked = plt.cm.magma.copy()
    cmap_masked.set_bad("#1a1a2e")  # dark = blocked (the future)

    fig_mask, ax_mask = plt.subplots(figsize=(5, 4.2))
    im_mask = ax_mask.imshow(grid_masked, cmap=cmap_masked)
    ax_mask.set_title("Causal-masked scores — head 0\n(dark upper triangle = the future, blocked)")
    ax_mask.set_xlabel("key token (what I look at)")
    ax_mask.set_ylabel("query token (who is asking)")
    ax_mask.set_xticks(range(grid_masked.shape[1]))
    ax_mask.set_yticks(range(grid_masked.shape[0]))
    fig_mask.colorbar(im_mask, ax=ax_mask, label="score (pre-softmax)")
    fig_mask.tight_layout()
    ax_mask
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The dark staircase is the future, walled off. Query token 0 can only see key 0;
        query 7 can see all of 0–7. Each row sees itself and everything to its left.

        ## 4 · Softmax → attention weights

        Raw scores aren't a "blend" yet — they're arbitrary real numbers. **Softmax** along
        the **key axis** (the last dim) turns each query's row into a probability
        distribution: all non-negative, summing to 1. Those are the actual attention
        **weights** — how much of each value vector this query will absorb.

        ```
        weights = softmax(scores_masked, dim=-1)    # over the last (key) axis
        ```

        The `-∞` future cells become exactly `0`. We'll *assert* every row sums to 1.
        """
    )
    return


@app.cell
def _(scores_masked, torch):
    # Softmax over the LAST dim (keys): each query row becomes a distribution.
    weights = torch.softmax(scores_masked, dim=-1)  # [B, heads, S, S]

    # Every query's weights must sum to 1 (it's a probability distribution).
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    return weights, row_sums


@app.cell
def _(weights, plt):
    w0 = weights[0, 0].detach().numpy()
    fig_w, ax_w = plt.subplots(figsize=(5, 4.4))
    im_w = ax_w.imshow(w0, cmap="magma", vmin=0.0, vmax=1.0)
    ax_w.set_title("Attention weights after softmax — head 0\n(each row sums to 1; future = 0)")
    ax_w.set_xlabel("key token")
    ax_w.set_ylabel("query token")
    ax_w.set_xticks(range(w0.shape[1]))
    ax_w.set_yticks(range(w0.shape[0]))
    # Annotate the lower triangle so you can read the actual probabilities.
    for qi in range(w0.shape[0]):
        for ki in range(w0.shape[1]):
            if w0[qi, ki] > 0.005:
                ax_w.text(ki, qi, f"{w0[qi, ki]:.2f}", ha="center", va="center",
                          color="#57b6f5", fontsize=7)
    fig_w.colorbar(im_w, ax=ax_w, label="attention weight")
    fig_w.tight_layout()
    ax_w
    return


@app.cell
def _(mo, row_sums):
    mo.callout(
        mo.md(
            f"""
            **Verified:** every query row sums to 1.0 (min={row_sums.min():.4f},
            max={row_sums.max():.4f}). Row 0 is a single `1.00` (token 0 can only attend to
            itself). Each later row spreads its 1.0 over itself and the tokens before it.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5 · Weighted sum of V, concat heads, output projection

        The weights tell each query *how much* of each value to take. We finish attention in
        three moves:

        ```
        per_head = weights @ V                      # [B, heads, S, S] @ [B, heads, S, head_dim]
                                                    #   -> [B, heads, S, head_dim]
        concat   = per_head.transpose(1,2).reshape(B, S, H)   # glue heads back: -> [B, S, H]
        output   = F.linear(concat, W_o)            # final mix -> [B, S, H]
        ```

        1. **Weighted sum:** multiply weights by `V` → each token gets a blended value vector.
        2. **Concat heads:** `transpose(1, 2)` puts `S` back in front, then `reshape` stitches
           the `num_heads × head_dim` lanes back into one `H`-wide vector per token.
        3. **Output projection `W_o`:** one last `[H, H]` matmul mixes the head outputs back
           into the residual stream.

        In, out, **same shape** `[B, S, H]`.
        """
    )
    return


@app.cell
def _(weights, V, W_o, F, B, S, H):
    per_head = weights @ V                                  # [B, heads, S, head_dim]
    concat = per_head.transpose(1, 2).contiguous().view(B, S, H)  # [B, S, H]
    output = F.linear(concat, W_o)                          # [B, S, H]
    return per_head, concat, output


@app.cell
def _(mo, weights, V, per_head, concat, output, X):
    mo.md(
        f"""
        The full pipeline shapes, end to end:

        ```
        weights              : {tuple(weights.shape)}    # [B, heads, S, S]
        V                    : {tuple(V.shape)}    # [B, heads, S, head_dim]
        per_head = weights@V : {tuple(per_head.shape)}    # [B, heads, S, head_dim]
        concat (heads glued) : {tuple(concat.shape)}        # [B, S, H]
        output = concat@W_oᵀ : {tuple(output.shape)}        # [B, S, H]

        X (we started here)  : {tuple(X.shape)}        # [B, S, H]  -- same shape out as in
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6 · All the heads at once

        Each head learned (well, *was randomly initialized with*) its own way of looking
        around. Here are the post-softmax causal weights for **all four heads** side by side.
        Different staircases = heads attend differently. In a trained model, one head might
        track the previous token, another might find the subject of a sentence, and so on.
        """
    )
    return


@app.cell
def _(weights, num_heads, plt):
    fig_grid, axes_grid = plt.subplots(1, num_heads, figsize=(3.1 * num_heads, 3.2))
    for h in range(num_heads):
        wh = weights[0, h].detach().numpy()
        axg = axes_grid[h]
        axg.imshow(wh, cmap="magma", vmin=0.0, vmax=1.0)
        axg.set_title(f"head {h}")
        axg.set_xlabel("key")
        if h == 0:
            axg.set_ylabel("query")
        axg.set_xticks(range(wh.shape[1]))
        axg.set_yticks(range(wh.shape[0]))
    fig_grid.suptitle("Causal attention weights, per head (each row sums to 1)")
    fig_grid.tight_layout()
    axes_grid[0]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7 · Interactive: explore one query token

        Pick a **query token** and a **head**, and toggle the **causal mask** on or off.
        The bar chart shows that query's attention distribution (how it splits its 1.0 of
        attention across the keys); the heatmap redraws the whole head so you can see the
        mask appear and disappear.
        """
    )
    return


@app.cell
def _(mo, S, num_heads):
    query_picker = mo.ui.slider(0, S - 1, value=S - 1, label="query token index")
    head_picker = mo.ui.slider(0, num_heads - 1, value=0, label="head")
    causal_toggle = mo.ui.switch(value=True, label="causal mask ON")
    mo.vstack([query_picker, head_picker, causal_toggle])
    return query_picker, head_picker, causal_toggle


@app.cell
def _(Q, K, head_dim, S, torch, query_picker, head_picker, causal_toggle):
    # Recompute scores -> weights for the chosen head, honoring the live toggle.
    qi_sel = query_picker.value
    h_sel = head_picker.value
    causal_on = causal_toggle.value

    scores_sel = (Q[0, h_sel] @ K[0, h_sel].transpose(-2, -1)) / (head_dim**0.5)  # [S, S]
    if causal_on:
        mask_sel = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        scores_sel = scores_sel.masked_fill(mask_sel, float("-inf"))
    weights_sel = torch.softmax(scores_sel, dim=-1)  # [S, S]
    row_sel = weights_sel[qi_sel].detach().numpy()
    return qi_sel, h_sel, causal_on, weights_sel, row_sel


@app.cell
def _(mo, qi_sel, h_sel, causal_on, row_sel, np):
    top_k = int(np.argmax(row_sel))
    mo.md(
        f"""
        **Query token {qi_sel}, head {h_sel}, causal mask {"ON" if causal_on else "OFF"}.**

        This query spreads its attention across {int((row_sel > 0.005).sum())} key token(s).
        It attends most strongly to **key token {top_k}** (weight {row_sel[top_k]:.2f}).
        {"With the mask on, every key after token " + str(qi_sel) + " is forced to 0 — the future is invisible." if causal_on else "With the mask OFF, this query can see *all* keys, including future ones (it's peeking)."}
        """
    )
    return


@app.cell
def _(row_sel, qi_sel, causal_on, np, plt):
    fig_bar, ax_bar = plt.subplots(figsize=(6, 3.2))
    keys_axis = np.arange(len(row_sel))
    bar_colors = ["#57b6f5" if k <= qi_sel or not causal_on else "#cccccc"
                  for k in keys_axis]
    ax_bar.bar(keys_axis, row_sel, color=bar_colors)
    ax_bar.axvline(qi_sel, color="#f0986b", linestyle="--", linewidth=1.5,
                   label=f"the query itself (token {qi_sel})")
    ax_bar.set_title(f"Where query token {qi_sel} sends its attention")
    ax_bar.set_xlabel("key token")
    ax_bar.set_ylabel("attention weight")
    ax_bar.set_xticks(keys_axis)
    ax_bar.set_ylim(0, 1)
    ax_bar.legend()
    for k, val in zip(keys_axis, row_sel):
        if val > 0.005:
            ax_bar.text(k, val + 0.02, f"{val:.2f}", ha="center", fontsize=7)
    fig_bar.tight_layout()
    ax_bar
    return


@app.cell
def _(weights_sel, h_sel, causal_on, plt):
    ws = weights_sel.detach().numpy()
    fig_hsel, ax_hsel = plt.subplots(figsize=(5, 4.2))
    im_hsel = ax_hsel.imshow(ws, cmap="magma", vmin=0.0, vmax=1.0)
    ax_hsel.set_title(f"head {h_sel} weights — causal {'ON' if causal_on else 'OFF'}")
    ax_hsel.set_xlabel("key token")
    ax_hsel.set_ylabel("query token")
    ax_hsel.set_xticks(range(ws.shape[1]))
    ax_hsel.set_yticks(range(ws.shape[0]))
    fig_hsel.colorbar(im_hsel, ax=ax_hsel, label="attention weight")
    fig_hsel.tight_layout()
    ax_hsel
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8 · Does it match PyTorch? (the proof)

        Everything above was hand-rolled so you could see each step. In real code you'd call
        the fused, optimized `F.scaled_dot_product_attention(q, k, v, is_causal=True)` — it
        does the scale, the causal mask, the softmax, and the weighted sum in one shot. If our
        from-scratch math is right, the two must agree. Let's run both and assert.
        """
    )
    return


@app.cell
def _(F, Q, K, V, W_o, B, S, H, output, torch, mo):
    # PyTorch's fused version of steps 2-5, with the causal mask built in.
    ref_per_head = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=True)
    ref_concat = ref_per_head.transpose(1, 2).contiguous().view(B, S, H)
    ref_output = F.linear(ref_concat, W_o)

    # Our hand-rolled `output` must match PyTorch's `ref_output`.
    torch.testing.assert_close(output, ref_output, rtol=1e-5, atol=1e-5)
    max_abs_diff = (output - ref_output).abs().max().item()

    mo.callout(
        mo.md(
            f"""
            ✅ **Match confirmed.** Our from-scratch attention equals
            `F.scaled_dot_product_attention(..., is_causal=True)` to floating-point
            tolerance — max absolute difference = `{max_abs_diff:.2e}`.

            Same scale, same causal mask, same softmax, same weighted sum — we just wrote it
            out longhand so every step was visible.
            """
        ),
        kind="success",
    )
    return ref_output, max_abs_diff


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Why this matters for parallelism.** Look back at the causal heatmap: a query at
            token `i` needs the **keys and values of every token up to `i`**. That single fact
            is the whole reason attention is hard to split across GPUs.

            - The **MLP** touches each token on its own — trivially shardable by token.
            - **Attention** forces tokens to consult each other. If you shard the sequence
              across GPUs (**sequence / tensor-sequence parallelism**), each GPU only holds
              *some* tokens — so before attention it must **gather** the K/V of the other
              tokens it's allowed to see. That gather is the price of splitting attention.

            Hold onto the picture of the lower-triangular staircase — it explains every
            communication step in the parallelism modules.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 9 · Check your understanding""")
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "If H = 16 and num_heads = 4, what is head_dim, and what shape is Q before attention?":
                mo.md(
                    r"""
                    `head_dim = H / num_heads = 16 / 4 = **4**`. Before the attention op, `Q`
                    is `[B, num_heads, S, head_dim] = [1, 4, 8, 4]`. The `4 × 4 = 16`
                    reconstructs `H` exactly when you concatenate the heads back together.
                    """
                ),
            "Why do we set masked positions to −∞ instead of 0 before the softmax?":
                mo.md(
                    r"""
                    Because the mask is applied **before** the softmax, and `softmax(−∞) = 0`.
                    Setting a score to `−∞` guarantees that key gets *exactly* zero weight and
                    contributes nothing to the blend. Setting it to `0` wouldn't work — `0` is
                    a perfectly ordinary score and `softmax` would still give it a positive
                    weight, letting the query peek at the future.
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

        - **`Q`, `K`, `V`** are three linear projections of the same input `X`; splitting the
          width `H` into `num_heads × head_dim` and transposing gives the
          `[B, heads, S, head_dim]` layout that the attention op expects.
        - **Scores** = `Q @ Kᵀ / √head_dim` form an `S × S` relevance grid per head.
        - The **causal mask** sets future positions to `−∞` so the softmax zeros them out —
          no peeking ahead.
        - **Softmax** over the key axis turns scores into weights that sum to 1; the
          **weighted sum of `V`**, concatenated across heads and run through `W_o`, returns a
          `[B, S, H]` tensor — same shape as the input.
        - Our hand-rolled version is **numerically identical** to
          `F.scaled_dot_product_attention(..., is_causal=True)`.
        - A query needs the K/V of all earlier tokens — the reason attention (not the MLP) is
          where GPUs are forced to talk.

        **Next:** read the companion chapter
        [`../course/03-attention-refresher.html`](../course/03-attention-refresher.html),
        then move on to the parallelism modules to see how this single block gets split
        across GPUs.
        """
    )
    return


if __name__ == "__main__":
    app.run()
