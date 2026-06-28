import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Tensor Parallelism (TP) — split the weights across GPUs

        **What you'll build:** a tiny, fully-runnable model of tensor parallelism. You'll
        simulate `D` GPUs ("ranks") inside this one notebook as a plain Python list of
        tensors, implement the three collective operations (`broadcast`, `all_reduce`,
        `all_gather`) as pure functions, and then build **TP attention** and **TP MLP**
        from scratch — checking at every step that the parallel result is *bit-for-bit*
        equal to a plain single-process reference.

        By the end you'll understand the one genuinely tricky idea in TP —
        **column-parallel vs row-parallel** — and exactly why attention ends with an
        `all_reduce`.

        /// admonition | ELI5
            type: info

        A thick novel needs proofreading, and attention already splits the job into
        independent **chapters** (the *heads*) that never reference each other. So hand
        each proofreader (GPU) a different stack of chapters. Nobody needs the whole
        book; each keeps only their stack. That's tensor parallelism — slice the *work
        and the weights* by head. The catch: at the very end you staple everyone's
        corrections back into one book, and that stapling is a quick all-to-all **sum**
        (the `all_reduce`).
        ///
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
    from matplotlib.patches import Rectangle

    torch.manual_seed(0)

    # Course palette (keep these consistent across every chart)
    C_TP = "#57b6f5"        # tensor parallelism / accent
    C_WEIGHTS = "#c099f0"   # weights (these SHRINK under TP)
    C_ACT = "#f0986b"       # activations (these stay FLAT under TP)
    C_NEUTRAL = "#9aa5b1"
    return mo, np, torch, F, plt, Rectangle, C_TP, C_WEIGHTS, C_ACT, C_NEUTRAL


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Recap the problem TP solves

        A transformer block holds big square weight matrices. Attention alone has four
        of them — `W_q`, `W_k`, `W_v`, `W_o`, each `[H, H]`. On a single GPU you must
        store *all* of them, and for a large model they simply don't fit.

        **The plan:** shard those weights across `D` GPUs so **no GPU holds a full
        weight matrix**. We shard attention **by heads**, because the heads are already
        independent — each head does its own little attention in its own `head_dim`-wide
        lane and never talks to the others. That independence is the gift TP exploits.

        /// admonition | The one limitation to keep in the back of your mind
            type: warn

        TP shrinks the **weights** (model-state memory), but it leaves the
        **activations replicated** — every GPU still holds the full `X` of shape
        `[B, S, H]`. So TP gives **zero** activation-memory savings. That gap is exactly
        what *sequence parallelism* (the next notebook) exists to close.
        ///
        """
    )
    return


@app.cell
def _(mo, torch):
    # Tiny problem dimensions — small enough to print and read by eye.
    B_tp = 1     # batch
    S_tp = 6     # sequence length (tokens)
    H_tp = 16    # hidden size
    num_heads_tp = 4
    head_dim_tp = H_tp // num_heads_tp   # = 4  (each head is a 4-wide lane)
    world_size_tp = 4                    # simulate 4 GPUs => each rank owns 1 head

    mo.md(
        rf"""
        We'll use deliberately tiny tensors so every shape is legible:

        ```
        B (batch)        = {B_tp}
        S (seq length)   = {S_tp}
        H (hidden)       = {H_tp}
        num_heads        = {num_heads_tp}
        head_dim (D_head)= {head_dim_tp}      # H / num_heads
        world_size (D)   = {world_size_tp}      # simulated GPUs  ->  1 head per rank
        ```

        With `world_size = {world_size_tp}` and `{num_heads_tp}` heads, **each rank owns
        exactly one head** — the cleanest case to reason about.
        """
    )
    return B_tp, S_tp, H_tp, num_heads_tp, head_dim_tp, world_size_tp


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Simulating `D` GPUs in one process

        We can't spin up 4 real GPUs inside a notebook, so we **simulate** them. A group
        of `D` ranks is just a **Python list of tensors** — `ranks[r]` is "the tensor
        living on GPU `r`". The three collective operations every parallel program needs
        become tiny pure functions over that list:

        - **`broadcast`** — one-to-all. Rank `src` has a tensor; everyone ends up with a
          copy. (Used to hand the same input `X` to every GPU.)
        - **`all_reduce`** — sum every rank's tensor element-wise; the **total lands on
          every rank**. (This is the "staple the corrections together" step.)
        - **`all_gather`** — concatenate each rank's *distinct* chunk along an axis;
          everyone ends up with the whole tensor.

        In real code these are `torch.distributed.broadcast / all_reduce / all_gather`
        and they move bytes over NVLink. Here they're just list comprehensions — but the
        *semantics* are identical, which is all we need to learn the algorithm.
        """
    )
    return


@app.cell
def _(torch):
    def sim_broadcast(rank_tensors, src=0):
        "one-to-all: everyone receives a copy of src's tensor"
        return [rank_tensors[src].clone() for _ in rank_tensors]

    def sim_all_reduce(rank_tensors):
        "sum across ranks; the total lands on every rank"
        total = rank_tensors[0].clone()
        for t in rank_tensors[1:]:
            total = total + t
        return [total.clone() for _ in rank_tensors]

    def sim_all_gather(rank_tensors, dim):
        "concatenate distinct chunks along `dim`; everyone gets the whole tensor"
        full = torch.cat(list(rank_tensors), dim=dim)
        return [full.clone() for _ in rank_tensors]

    return sim_broadcast, sim_all_reduce, sim_all_gather


@app.cell
def _(mo, torch, sim_broadcast, sim_all_reduce, sim_all_gather):
    # Worked examples on tiny 1-D tensors so you can watch the lists change.
    demo_ranks = [torch.tensor([float(r), float(r) + 0.5]) for r in range(4)]

    bc = sim_broadcast(demo_ranks, src=0)
    ar = sim_all_reduce(demo_ranks)
    ag = sim_all_gather(demo_ranks, dim=0)

    def _fmt(name, lst):
        rows = "\n".join(f"  rank{r}: {t.tolist()}" for r, t in enumerate(lst))
        return f"{name}:\n{rows}"

    mo.md(
        rf"""
        Watch each collective transform a 4-rank list. Start state — each rank holds a
        *different* little tensor:

        ```
        {_fmt("START", demo_ranks)}
        ```

        ```
        {_fmt("broadcast(src=0)  -> everyone copies rank0", bc)}

        {_fmt("all_reduce        -> every rank holds the elementwise SUM", ar)}

        {_fmt("all_gather(dim=0) -> every rank holds the concatenation", ag)}
        ```

        Picture it:

        ```
        broadcast      all_reduce            all_gather
        [a]            [a]   ┐                [a]
        [b]   src=0    [b]   ├─ sum ─┐        [b]   concat
        [c]   ──────>  [c]   ┘       ▼        [c]   ───────>  [a|b|c|d] on all
        [d]            [d]        Σ on all     [d]
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. The crux: column-parallel vs row-parallel

        Q/K/V and the output projection `W_o` are **all** square `[H, H]` matrices, so a
        natural question (and the one that generated the most confusion in the original
        sessions) is: *why can't I shard them the same way?* Because they sit on
        **opposite sides** of the attention operation, so they get sliced along
        **different axes**.

        Reason from input/output features. Remember the PyTorch convention:

        > `F.linear(X, W)` computes `X @ W.T`, with `W` stored as `[out_features, in_features]`.

        ### Column-parallel — split the OUTPUT features (for Q, K, V)

        The Q/K/V projections run *before* attention; their job is to produce per-head
        Q/K/V. Rank `r` only needs the Q/K/V for *its* heads, so it only needs the rows
        of `W_q` that produce those output features:

        ```
        W_q is [H, H] = [out, in].   Slice the OUT (head) rows:
            W_q_p[r] = W_q[r*chunk : (r+1)*chunk, :]     # shape [H/D, H]
            X @ W_q_p.T  ->  [B, S, H/D]                 # only this rank's heads
        ```

        Each rank produces a *different, complete* slice of the output. The input `X` is
        replicated, so **no communication is needed** — every rank just computes its
        slice.

        ### Row-parallel — split the INPUT features (for `W_o`)

        The output projection `W_o` runs *after* attention; its input is the concatenated
        head outputs. But rank `r` only *has* its own heads' outputs — a partial input.
        So it takes the columns of `W_o` that consume *those* input features:

        ```
        W_o is [H, H] = [out, in].   Slice the IN (head) cols:
            W_o_p[r] = W_o[:, r*chunk : (r+1)*chunk]     # shape [H, H/D]
            attn_p @ W_o_p.T  ->  [B, S, H]              # a PARTIAL full-width output
        ```

        Now each rank produces a `[B, S, H]` tensor that is the **partial** contribution
        of its heads. The true output is the **sum** of all ranks' partials — that's the
        `all_reduce`.

        /// admonition | The pattern that recurs forever
            type: success

        **Column-parallel (split output)** → no communication, each rank holds a distinct
        slice. **Row-parallel (split input)** → produces partial sums → needs an
        `all_reduce` to combine. Attention is *column-parallel Q/K/V* followed by
        *row-parallel W_o*. The `all_reduce` at the end is the price of splitting `W_o`
        by input.
        ///
        """
    )
    return


@app.cell
def _(torch, num_heads_tp, world_size_tp):
    # The two sharding helpers, mirroring shard_col_parallel / shard_row_parallel
    # in ../tensor_parallelism.py exactly.

    def shard_col_parallel(W, rank, world_size, num_heads):
        "split OUTPUT features (rows) of a [H, H] QKV weight -> [H/D, H]"
        H_out, H_in = W.shape
        D_head = H_out // num_heads
        p_heads = num_heads // world_size
        start = rank * p_heads * D_head
        end = start + p_heads * D_head
        return W[start:end, :].contiguous()

    def shard_row_parallel(W_o, rank, world_size, num_heads):
        "split INPUT features (cols) of a [H, H] output weight -> [H, H/D]"
        H_out, H_in = W_o.shape
        D_head = H_in // num_heads
        p_heads = num_heads // world_size
        start = rank * p_heads * D_head
        end = start + p_heads * D_head
        return W_o[:, start:end].contiguous()

    return shard_col_parallel, shard_row_parallel


@app.cell
def _(mo, torch, H_tp, num_heads_tp, world_size_tp, shard_col_parallel, shard_row_parallel):
    # Build full weights, then show what one rank gets from each helper.
    W_q_full = torch.randn(H_tp, H_tp)
    W_o_full = torch.randn(H_tp, H_tp)

    col_shard_r1 = shard_col_parallel(W_q_full, rank=1, world_size=world_size_tp, num_heads=num_heads_tp)
    row_shard_r1 = shard_row_parallel(W_o_full, rank=1, world_size=world_size_tp, num_heads=num_heads_tp)

    mo.md(
        rf"""
        Concretely, with `H={H_tp}`, `num_heads={num_heads_tp}`, `world_size={world_size_tp}`
        (so chunk = `H/D = {H_tp // world_size_tp}` features per rank):

        ```
        Full W_q : {tuple(W_q_full.shape)}   ->  column-parallel slice for rank 1 : {tuple(col_shard_r1.shape)}   (rows {1 * (H_tp // world_size_tp)}..{2 * (H_tp // world_size_tp)})
        Full W_o : {tuple(W_o_full.shape)}   ->  row-parallel    slice for rank 1 : {tuple(row_shard_r1.shape)}   (cols {1 * (H_tp // world_size_tp)}..{2 * (H_tp // world_size_tp)})
        ```

        Same input shape `[16, 16]`, **opposite axis** sliced — which is exactly why the
        code needs two different helpers.
        """
    )
    return W_q_full, W_o_full


@app.cell
def _(mo):
    mo.md(
        r"""
        ### See the cut: which axis each helper slices

        The diagram below shows the *stored* `[out, in]` weight for both projections.
        Each color is the slice one rank keeps. Column-parallel cuts **horizontal bands**
        (output rows); row-parallel cuts **vertical bands** (input columns).
        """
    )
    return


@app.cell
def _(np, plt, Rectangle, H_tp, world_size_tp, C_TP, C_WEIGHTS, C_ACT, C_NEUTRAL):
    rank_colors = [C_TP, C_WEIGHTS, C_ACT, C_NEUTRAL]
    chunk_sz = H_tp // world_size_tp

    fig_cut, (ax_col, ax_row) = plt.subplots(1, 2, figsize=(9, 4.2))

    # Column-parallel: horizontal bands over the OUTPUT (row) axis.
    ax_col.set_title("Column-parallel  W_q  [out, in]\nsplit OUTPUT rows  ->  [H/D, H]", fontsize=10)
    for _r in range(world_size_tp):
        ax_col.add_patch(Rectangle((0, _r * chunk_sz), H_tp, chunk_sz,
                                   facecolor=rank_colors[_r], edgecolor="white", lw=1.5))
        ax_col.text(H_tp / 2, _r * chunk_sz + chunk_sz / 2, f"rank {_r}",
                    ha="center", va="center", fontsize=9, weight="bold")

    # Row-parallel: vertical bands over the INPUT (col) axis.
    ax_row.set_title("Row-parallel  W_o  [out, in]\nsplit INPUT cols  ->  [H, H/D]", fontsize=10)
    for _r in range(world_size_tp):
        ax_row.add_patch(Rectangle((_r * chunk_sz, 0), chunk_sz, H_tp,
                                   facecolor=rank_colors[_r], edgecolor="white", lw=1.5))
        ax_row.text(_r * chunk_sz + chunk_sz / 2, H_tp / 2, f"rank {_r}",
                    ha="center", va="center", fontsize=9, weight="bold", rotation=90)

    for _ax in (ax_col, ax_row):
        _ax.set_xlim(0, H_tp)
        _ax.set_ylim(H_tp, 0)
        _ax.set_xlabel("in_features")
        _ax.set_ylabel("out_features")
        _ax.set_xticks([0, H_tp])
        _ax.set_yticks([0, H_tp])

    fig_cut.tight_layout()
    ax_col
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. The insight everyone trips on: head-space vs residual-space

        This was flagged as *the* main gap, so let's slow down. Before `W_o`, the
        attention output lives in **head space** — per-head features, *not yet* a
        meaningful update to the token's residual vector. `W_o` is what *translates*
        head-space features into a residual-space update.

        The key algebra: view it as a **block sum**. With 4 heads, write the attention
        output as four side-by-side blocks and `W_o` as four stacked blocks:

        ```
        attn = [ A0 | A1 | A2 | A3 ]      each Ai : [B, S, head_dim]
        W_o  = [ W0 ; W1 ; W2 ; W3 ]      each Wi : [head_dim, H]

        attn @ W_o.T  =  A0@W0 + A1@W1 + A2@W2 + A3@W3     # a SUM of per-head terms!
        ```

        Each term `Ai @ Wi` is a **full `[B, S, H]`** residual update from a single head.
        The output projection is *secretly a sum over heads*. So if rank 0 owns head 0 it
        computes `A0@W0`, rank 1 owns head 1 and computes `A1@W1`, and the `all_reduce`
        adds the partials. **TP is just rearranging this sum across GPUs.** Let's prove it
        numerically.
        """
    )
    return


@app.cell
def _(mo, torch, F, B_tp, S_tp, H_tp, num_heads_tp, head_dim_tp, W_o_full):
    # Prove: attn @ W_o.T  ==  sum over heads of (A_i @ W_o_block_i.T)
    attn_full = torch.randn(B_tp, S_tp, H_tp)   # stand-in head-space attention output

    # Reference: the whole projection at once.
    out_reference = F.linear(attn_full, W_o_full)        # [B, S, H]

    # Per-head decomposition. W_o stored [out=H, in=H]; head i consumes input cols i*dh..(i+1)*dh.
    partials = []
    for i in range(num_heads_tp):
        cols = slice(i * head_dim_tp, (i + 1) * head_dim_tp)
        A_i = attn_full[:, :, cols]            # [B, S, head_dim]   head i's features
        W_i = W_o_full[:, cols]                # [H, head_dim]      its slice of W_o
        partials.append(F.linear(A_i, W_i))    # [B, S, H]          full-width partial

    out_summed = sum(partials)
    torch.testing.assert_close(out_summed, out_reference)

    mo.md(
        rf"""
        We took a head-space `attn` of shape `{tuple(attn_full.shape)}`, computed the full
        projection, then *separately* computed each head's `Ai @ Wi.T` partial (each one a
        full `{tuple(partials[0].shape)}` tensor) and summed them.

        ```
        per-head partials : {num_heads_tp} tensors, each {tuple(partials[0].shape)}
        sum of partials   == full projection ?  ->  asserted equal
        ```

        /// admonition | Verified
            type: success

        `sum_i (A_i @ W_i.T) == attn @ W_o.T`. The output projection really is a sum over
        heads — so splitting heads across ranks and `all_reduce`-ing the partials is
        *exactly* the same computation.
        ///
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Full TP attention

        Now assemble it. We need a plain single-process **reference** attention to check
        against, then the parallel version that loops over simulated ranks:

        1. **broadcast** `X` to every rank (it's replicated),
        2. each rank does **column-parallel** Q/K/V → local attention over *its* heads,
        3. each rank does **row-parallel** `W_o` → a partial `[B, S, H]`,
        4. **`all_reduce`** the partials → the true output on every rank,
        5. assert it equals the reference.
        """
    )
    return


@app.cell
def _(torch, F):
    def project_to_heads(projected, num_heads, head_dim):
        B, S, _ = projected.shape
        return projected.view(B, S, num_heads, head_dim).transpose(1, 2)

    def attn_reference(X, W_q, W_k, W_v, W_o, num_heads):
        "plain single-process causal attention (the ground truth)"
        B, S, H = X.shape
        D_head = H // num_heads
        q = project_to_heads(F.linear(X, W_q), num_heads, D_head)
        k = project_to_heads(F.linear(X, W_k), num_heads, D_head)
        v = project_to_heads(F.linear(X, W_v), num_heads, D_head)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, S, H)
        return F.linear(a, W_o)

    return project_to_heads, attn_reference


@app.cell
def _(
    mo, torch, F,
    B_tp, S_tp, H_tp, num_heads_tp, head_dim_tp, world_size_tp,
    project_to_heads, attn_reference,
    shard_col_parallel, shard_row_parallel,
    sim_broadcast, sim_all_reduce,
):
    # Fresh full weights for the end-to-end attention check.
    torch.manual_seed(7)
    Wq_a = torch.randn(H_tp, H_tp) * H_tp ** -0.5
    Wk_a = torch.randn(H_tp, H_tp) * H_tp ** -0.5
    Wv_a = torch.randn(H_tp, H_tp) * H_tp ** -0.5
    Wo_a = torch.randn(H_tp, H_tp) * H_tp ** -0.5
    X_a = torch.randn(B_tp, S_tp, H_tp)

    # Ground truth.
    ref_attn = attn_reference(X_a, Wq_a, Wk_a, Wv_a, Wo_a, num_heads_tp)

    # --- Parallel version, looping over simulated ranks ---
    p_heads_a = num_heads_tp // world_size_tp        # heads per rank (=1 here)
    p_hidden_a = p_heads_a * head_dim_tp             # local hidden width

    # 1. broadcast X to all ranks (replicated activations).
    X_ranks = sim_broadcast([X_a for _ in range(world_size_tp)], src=0)

    # 2-3. each rank: column-parallel QKV -> local attn -> row-parallel W_o -> partial.
    partial_outs = []
    for _r in range(world_size_tp):
        Wq_p = shard_col_parallel(Wq_a, _r, world_size_tp, num_heads_tp)
        Wk_p = shard_col_parallel(Wk_a, _r, world_size_tp, num_heads_tp)
        Wv_p = shard_col_parallel(Wv_a, _r, world_size_tp, num_heads_tp)
        Wo_p = shard_row_parallel(Wo_a, _r, world_size_tp, num_heads_tp)

        Xr = X_ranks[_r]
        q_r = project_to_heads(F.linear(Xr, Wq_p), p_heads_a, head_dim_tp)
        k_r = project_to_heads(F.linear(Xr, Wk_p), p_heads_a, head_dim_tp)
        v_r = project_to_heads(F.linear(Xr, Wv_p), p_heads_a, head_dim_tp)
        a_r = F.scaled_dot_product_attention(q_r, k_r, v_r, dropout_p=0.0, is_causal=True)
        a_r = a_r.transpose(1, 2).contiguous().view(B_tp, S_tp, p_hidden_a)
        partial_outs.append(F.linear(a_r, Wo_p))     # row-parallel -> partial [B,S,H]

    # 4. all_reduce the partials.
    reduced_attn = sim_all_reduce(partial_outs)
    tp_attn_out = reduced_attn[0]

    # 5. verify.
    torch.testing.assert_close(tp_attn_out, ref_attn)

    mo.md(
        rf"""
        ```
        reference attn output : {tuple(ref_attn.shape)}
        each rank's partial   : {tuple(partial_outs[0].shape)}   ({world_size_tp} of them)
        after all_reduce(sum) : {tuple(tp_attn_out.shape)}   identical on every rank
        max |TP - reference|  : {(tp_attn_out - ref_attn).abs().max().item():.2e}
        ```

        /// admonition | TP attention verified
            type: success

        Column-parallel Q/K/V + local attention + row-parallel `W_o` + `all_reduce`
        reproduces single-process attention **exactly** — and no rank ever held a full
        weight matrix.
        ///
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. TP MLP — the same recipe, even simpler

        The MLP is `W_in` (expand `H → I`) then `W_out` (contract `I → H`). The mapping is
        perfect: **column-parallel the first matrix** (split the intermediate dimension
        `I` — each rank computes a slice of the expanded hidden), **row-parallel the
        second** (each rank consumes its slice of `I` and produces a partial output), then
        **`all_reduce`**. Expand-then-contract → column-then-row. No attention, no heads —
        just the same column/row pattern.
        """
    )
    return


@app.cell
def _(
    mo, torch, F,
    B_tp, S_tp, H_tp, world_size_tp,
    sim_broadcast, sim_all_reduce,
):
    I_tp = 32                       # MLP intermediate size
    p_I = I_tp // world_size_tp     # intermediate slice per rank

    torch.manual_seed(11)
    W_in_full = torch.randn(I_tp, H_tp) * H_tp ** -0.5      # [I, H]
    W_out_full = torch.randn(H_tp, I_tp) * I_tp ** -0.5     # [H, I]
    X_mlp = torch.randn(B_tp, S_tp, H_tp)

    # Reference MLP.
    def mlp_reference(X, W_in, W_out):
        return F.linear(F.gelu(F.linear(X, W_in)), W_out)

    ref_mlp = mlp_reference(X_mlp, W_in_full, W_out_full)

    # Parallel: broadcast X, shard, compute partials, all_reduce.
    Xmlp_ranks = sim_broadcast([X_mlp for _ in range(world_size_tp)], src=0)
    mlp_partials = []
    for _r in range(world_size_tp):
        # column-parallel W_in: split OUTPUT (intermediate) rows -> [p_I, H]
        W_in_p = W_in_full[_r * p_I:(_r + 1) * p_I, :].contiguous()
        # row-parallel W_out: split INPUT (intermediate) cols -> [H, p_I]
        W_out_p = W_out_full[:, _r * p_I:(_r + 1) * p_I].contiguous()

        hidden_p = F.gelu(F.linear(Xmlp_ranks[_r], W_in_p))   # [B, S, p_I]
        mlp_partials.append(F.linear(hidden_p, W_out_p))     # partial [B, S, H]

    tp_mlp_out = sim_all_reduce(mlp_partials)[0]
    torch.testing.assert_close(tp_mlp_out, ref_mlp)

    mo.md(
        rf"""
        ```
        W_in  {tuple(W_in_full.shape)}  -> column-parallel slice [p_I, H] = [{p_I}, {H_tp}]
        W_out {tuple(W_out_full.shape)}  -> row-parallel    slice [H, p_I] = [{H_tp}, {p_I}]
        each rank's partial  : {tuple(mlp_partials[0].shape)}   ({world_size_tp} of them)
        after all_reduce     : {tuple(tp_mlp_out.shape)}
        max |TP - reference| : {(tp_mlp_out - ref_mlp).abs().max().item():.2e}
        ```

        /// admonition | TP MLP verified
            type: success

        Column-parallel `W_in` + row-parallel `W_out` + `all_reduce` matches the
        single-process MLP exactly. Same pattern as attention — `all_reduce` is again the
        price of the row-parallel split.
        ///
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. What TP buys you (and what it doesn't) — the charts

        Three pictures make the trade-off concrete. We'll use realistic model dimensions
        for the memory math: `H = 4096`, `num_heads = 32`, `B = 1`, `S = 8192`, bf16
        (2 bytes/element).
        """
    )
    return


@app.cell
def _(np, plt, C_TP, C_NEUTRAL):
    # Chart (a): per-rank ATTENTION weight memory, single-GPU vs TP across D.
    H_big = 4096
    bytes_per = 2  # bf16
    attn_params = 4 * H_big * H_big          # W_q,W_k,W_v,W_o each [H,H]
    single_mb = attn_params * bytes_per / 1e6

    Ds = [1, 2, 4, 8, 16, 32]
    per_rank_mb = [single_mb / d for d in Ds]

    fig_a, ax_a = plt.subplots(figsize=(7, 4))
    bars_a = ax_a.bar([str(d) for d in Ds], per_rank_mb, color=C_TP, edgecolor="white")
    ax_a.axhline(single_mb, color=C_NEUTRAL, ls="--", lw=1.5,
                 label=f"single-GPU = {single_mb:.0f} MB")
    for _bar, _v in zip(bars_a, per_rank_mb):
        ax_a.text(_bar.get_x() + _bar.get_width() / 2, _v, f"{_v:.0f}",
                  ha="center", va="bottom", fontsize=8)
    ax_a.set_title("(a) Per-rank attention WEIGHT memory shrinks as 1/D")
    ax_a.set_xlabel("D  (number of GPUs / ranks)")
    ax_a.set_ylabel("MB per GPU (bf16)")
    ax_a.legend()
    fig_a.tight_layout()
    ax_a
    return


@app.cell
def _(np, plt, C_WEIGHTS, C_ACT):
    # Chart (b): model-state (weights) memory SHRINKS, activation memory FLAT, vs D.
    Hb = 4096
    Bb, Sb = 1, 8192
    bpp = 2
    weights_single_mb = 4 * Hb * Hb * bpp / 1e6      # attention weights
    act_mb = Bb * Sb * Hb * bpp / 1e6                # replicated activations X [B,S,H]

    Db = [1, 2, 4, 8, 16, 32]
    weights_mb = [weights_single_mb / d for d in Db]
    acts_mb = [act_mb for _ in Db]                   # flat — TP never shards activations

    x = np.arange(len(Db))
    w = 0.4
    fig_b, ax_b = plt.subplots(figsize=(7.5, 4))
    ax_b.bar(x - w / 2, weights_mb, w, label="model-state (weights)", color=C_WEIGHTS, edgecolor="white")
    ax_b.bar(x + w / 2, acts_mb, w, label="activations (replicated X)", color=C_ACT, edgecolor="white")
    ax_b.set_xticks(x, [str(d) for d in Db])
    ax_b.set_title("(b) TP shrinks weights but NOT activations — its core limitation")
    ax_b.set_xlabel("D  (number of GPUs / ranks)")
    ax_b.set_ylabel("MB per GPU (bf16)")
    ax_b.legend()
    fig_b.tight_layout()
    ax_b
    return


@app.cell
def _(np, plt, Rectangle, C_TP, C_WEIGHTS, C_ACT, C_NEUTRAL):
    # Chart (c): weight-matrix slicing diagram via imshow with colored col vs row slices.
    n = 16
    D_slices = 4
    chunk = n // D_slices
    palette = [C_TP, C_WEIGHTS, C_ACT, C_NEUTRAL]

    col_img = np.zeros((n, n))   # horizontal bands (column-parallel: output rows)
    row_img = np.zeros((n, n))   # vertical bands (row-parallel: input cols)
    for _r in range(D_slices):
        col_img[_r * chunk:(_r + 1) * chunk, :] = _r
        row_img[:, _r * chunk:(_r + 1) * chunk] = _r

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(palette)

    fig_c, (axc1, axc2) = plt.subplots(1, 2, figsize=(9, 4.2))
    axc1.imshow(col_img, cmap=cmap, vmin=0, vmax=D_slices - 1)
    axc1.set_title("(c) column-parallel: cut OUTPUT rows\n(Q/K/V)  -> distinct slices, no comms", fontsize=9)
    axc2.imshow(row_img, cmap=cmap, vmin=0, vmax=D_slices - 1)
    axc2.set_title("row-parallel: cut INPUT cols\n(W_o)  -> partial sums, needs all_reduce", fontsize=9)
    for _ax in (axc1, axc2):
        _ax.set_xlabel("in_features")
        _ax.set_ylabel("out_features")
        _ax.set_xticks([]); _ax.set_yticks([])
    fig_c.tight_layout()
    axc1
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8. Interactive: pick D and watch per-rank memory move

        Choose the number of ranks `D` (only **divisors of `num_heads = 32`** are valid —
        you can't split 32 heads evenly across 5 GPUs). The chart redraws the per-rank
        attention weight memory; note it always lands at `single / D`.
        """
    )
    return


@app.cell
def _(mo):
    # Define the UI element here; READ its .value in the next cell so it reacts.
    d_choice = mo.ui.dropdown(
        options={"1": 1, "2": 2, "4": 4, "8": 8, "16": 16, "32": 32},
        value="8",
        label="D (ranks) — divisors of num_heads=32",
    )
    d_choice
    return d_choice


@app.cell
def _(mo, plt, d_choice, C_TP, C_NEUTRAL):
    H_i = 4096
    attn_params_i = 4 * H_i * H_i
    single_mb_i = attn_params_i * 2 / 1e6
    D_sel = d_choice.value
    per_rank_sel = single_mb_i / D_sel

    fig_i, ax_i = plt.subplots(figsize=(6, 4))
    ax_i.bar(["single-GPU", f"TP, D={D_sel}"],
             [single_mb_i, per_rank_sel],
             color=[C_NEUTRAL, C_TP], edgecolor="white")
    for _xi, _v in enumerate([single_mb_i, per_rank_sel]):
        ax_i.text(_xi, _v, f"{_v:.0f} MB", ha="center", va="bottom", fontsize=10, weight="bold")
    ax_i.set_title(f"Per-rank attention weight memory  (1/{D_sel} of single-GPU)")
    ax_i.set_ylabel("MB per GPU (bf16)")
    ax_i.set_ylim(0, single_mb_i * 1.15)
    fig_i.tight_layout()
    ax_i
    return


@app.cell
def _(mo):
    mo.md(r"""## 9. Check your understanding""")
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Q1. Q/K/V and W_o are all [H, H]. Why can't one sharding function serve both?": mo.md(
                r"""
                Because the slice **axis** differs. Q/K/V are **column-parallel** — split
                the OUTPUT/head dimension (the rows of the stored `[out, in]` weight) —
                since their input `X` is replicated and each rank should produce a
                *distinct* slice of output, needing **no communication**. `W_o` is
                **row-parallel** — split the INPUT dimension (the columns) — since its
                input is already a per-head *partial* and each rank consumes only its
                heads, producing a partial output that must be **summed**. Same shape,
                opposite axis ⇒ two helpers (`shard_col_parallel` vs `shard_row_parallel`).
                """
            ),
            "Q2. Why does TP attention end with all_reduce, not all_gather?": mo.md(
                r"""
                Because the row-parallel `W_o` makes each rank produce a **partial sum** of
                the *same* full-width `[B, S, H]` output — the per-head contributions that
                must be **added together** (we proved `attn@W_o.T = Σ_i A_i@W_i.T`).
                `all_gather` *concatenates distinct pieces*, which would be wrong here: the
                pieces fully overlap (same batch, same positions, same hidden width) and
                represent the same output to be summed. So you want `all_reduce(SUM)`.
                Column-parallel outputs, by contrast, *would* be combined with `all_gather`
                because there each rank holds a distinct slice.
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

        - **TP shards weights, not activations.** Each rank holds `1/D` of `W_q/W_k/W_v/W_o`
          (and the MLP weights); `X` stays fully replicated at `[B, S, H]` everywhere.
        - **Heads are the natural shard.** They're already independent inside attention, so
          giving different heads to different ranks needs no extra coordination *during*
          attention.
        - **Column-parallel vs row-parallel is the whole game.** Split OUTPUT features →
          distinct slices, no comms. Split INPUT features → partial sums → `all_reduce`.
          Attention = column-parallel QKV then row-parallel `W_o`; MLP = column-parallel
          `W_in` then row-parallel `W_out`.
        - **The output projection is secretly a sum over heads** — which is precisely why
          partials `all_reduce` back to the exact single-process answer (we asserted it).
        - **The limitation:** weights shrink as `1/D`, but activation memory is flat.
          That's the opening for **sequence parallelism**.

        Companion course chapter: [`../course/05-tensor-parallelism.html`](../course/05-tensor-parallelism.html).
        Real implementation mirrored here: `../tensor_parallelism.py`
        (`shard_col_parallel`, `shard_row_parallel`, `tp_attn`, `tp_mlp`).
        """
    )
    return


if __name__ == "__main__":
    app.run()
