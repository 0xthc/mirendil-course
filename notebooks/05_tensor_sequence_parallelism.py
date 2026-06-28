import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Tensor-sequence parallelism (TSP)

        This is the paper's headline method — and the trickiest one. We're going to
        **build the whole algorithm from scratch on the CPU**, simulating a group of
        GPUs as a Python list of tensors, and prove our implementation matches an
        ordinary single-process transformer bit-for-bit.

        **What you'll build:**

        1. An intuition for how TSP differs from "TP + SP" (they are *not* the same).
        2. The reason TSP needs a **loop over weight shards** — the defining feature.
        3. A faithful re-implementation of **Algorithm 1** (TSP attention), narrated
           line by line, verified against a reference with `torch.testing.assert_close`.
        4. The TSP **MLP**, same idea, also verified.
        5. Charts using the test's *real* benchmark numbers showing TSP's memory win
           and its time cost — plus interactive controls to explore the tradeoff.

        Everything runs in seconds on tiny tensors (`B=1, S=8, H=16, 4 heads, D=4`).
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** Tensor parallelism gave each chef a few recipes. Sequence
            parallelism gave each reader a few pages. **TSP asks the *same* people to
            do both jobs at once** — each GPU holds some pages *and* some recipes.

            The clever, cheap part: you don't need extra GPUs. The price: since each
            GPU owns only *some* recipes, the recipes get passed around the table like
            **batons in a relay**, so every GPU can apply all of them to its own pages.
            Less memory, more passing-around.
            """
        ),
        kind="info",
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
    return mo, np, torch, F, plt, Rectangle


@app.cell
def _(mo):
    mo.md(
        r"""
        ## TSP vs "TP + SP" — get this distinction first

        There are **two different ways** to combine the weight-split and the
        token-split. They are not the same thing, and confusing them is the classic
        mistake.

        - **TP + SP** is a **2D grid** of GPUs. One axis does tensor parallelism, the
          other does sequence parallelism. With `T=2` and `Σ=2` you need `T·Σ = 4`
          GPUs, and *every* (head-shard, token-shard) combination physically exists on
          its own GPU. Nothing has to move during the forward pass.

        - **TSP** folds *both* roles onto the **same** `D` GPUs — the **diagonal** of
          that grid. With `D=2`, rank 0 owns (seq-shard 0, head-shard 0) and rank 1
          owns (seq-shard 1, head-shard 1). The off-diagonal cells *don't exist
          anywhere*. That's why TSP is cheaper in GPUs — and why it needs a loop.

        The diagram below makes the difference concrete.
        """
    )
    return


@app.cell
def _(plt, Rectangle):
    def _cell(ax, cx, cy, color, text, alpha=1.0, hatch=None, edge="#2b3a4a"):
        ax.add_patch(
            Rectangle(
                (cx, cy), 0.9, 0.9, facecolor=color, edgecolor=edge,
                linewidth=1.5, alpha=alpha, hatch=hatch,
            )
        )
        ax.text(cx + 0.45, cy + 0.45, text, ha="center", va="center",
                fontsize=9, color="#0f1720", weight="bold")

    def _draw_grids():
        fig_grid, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.2))
        rank_colors = ["#57b6f5", "#f0c674", "#5fd38a", "#c099f0"]

        # --- TP + SP: full 2x2 grid, every cell filled ---
        # rows = seq shards, cols = head shards. rank = row*2 + col
        for row in range(2):
            for col in range(2):
                rk = row * 2 + col
                _cell(axL, col, 1 - row, rank_colors[rk], f"rank {rk}")
        axL.set_title("TP + SP — 4 GPUs, full grid", fontsize=11, weight="bold")
        axL.text(-0.55, 0.45, "seq\nshards", ha="center", va="center",
                 fontsize=8, color="#7a8aa0", rotation=90)

        # --- TSP: same 2x2, only the DIAGONAL exists ---
        for row in range(2):
            for col in range(2):
                if row == col:
                    rk = 0 if row == 0 else 1
                    _cell(axR, col, 1 - row, rank_colors[rk], f"rank {rk}")
                else:
                    _cell(axR, col, 1 - row, "#1a2430", "missing",
                          alpha=0.5, hatch="//", edge="#3a4a5a")
        axR.set_title("TSP — 2 GPUs, diagonal only", fontsize=11, weight="bold")

        for ax in (axL, axR):
            ax.set_xlim(-0.7, 2.0)
            ax.set_ylim(-0.2, 2.2)
            ax.set_xticks([0.45, 1.45])
            ax.set_xticklabels(["head shard 0", "head shard 1"], fontsize=8, color="#7a8aa0")
            ax.set_yticks([1.45, 0.45])
            ax.set_yticklabels(["seq 0", "seq 1"], fontsize=8, color="#7a8aa0")
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)

        fig_grid.suptitle(
            "Same two jobs, two layouts: a filled grid (TP+SP) vs the diagonal (TSP)",
            fontsize=12, weight="bold",
        )
        fig_grid.tight_layout()
        return axL

    _draw_grids()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why TSP needs a loop

        Look at the diagonal layout. **Rank 0 holds sequence-shard 0 but only
        head-shard 0's weights.** To produce the *full* output for its own tokens, it
        needs *every* head shard applied to them — including head shards it doesn't
        own (which live on other ranks).

        So TSP **loops over all weight (head) shards**, circulating each shard to every
        rank in turn with a `broadcast`. On each pass, a rank applies the current
        shard's weights to its own tokens and adds the result to a running total. The
        loop is *compensating for the missing off-diagonal cells*.

        To prove our loop is correct, we'll need a multi-GPU group we can actually run.
        We simulate it.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Simulating `D` ranks in one process

        We can't spin up 4 GPUs inside a notebook, so we represent a "group of ranks"
        as a **Python list of tensors** — `[rank0_tensor, rank1_tensor, ...]` — and
        implement the three collective operations as pure functions over that list.
        This lets us run the *exact* distributed algorithm on the CPU.
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

    def project_to_heads(projected, num_heads, head_dim):
        "reshape [B, S, hidden] -> [B, num_heads, S, head_dim]"
        b, s, _ = projected.shape
        return projected.view(b, s, num_heads, head_dim).transpose(1, 2)

    return sim_broadcast, sim_all_gather, project_to_heads


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The reference: an ordinary, single-process attention

        Before sharding anything, we build the plain transformer attention we'll check
        against. Convention (matching the repo): `F.linear(X, W)` computes `X @ W.T`,
        and weights are stored `[out_features, in_features]`.

        Our tiny config: `B=1`, `S=8` tokens, `H=16` hidden, `4` heads (so head
        dim `= 4`), and `D=4` ranks. With `D=4`, each rank will own `S/D = 2` tokens
        **and** `4/4 = 1` head — exactly one cell on the diagonal.
        """
    )
    return


@app.cell
def _(torch, F, project_to_heads, mo):
    B, S, H = 1, 8, 16
    num_heads = 4
    head_dim = H // num_heads          # 4
    D = 4                              # number of ranks (world size)
    local_seq = S // D                 # 2 tokens per rank
    local_heads = num_heads // D       # 1 head per rank
    local_hidden = H // D              # 4 hidden dims per rank (one head)

    torch.manual_seed(1234)
    scale = H ** -0.5
    W_q = torch.randn(H, H) * scale
    W_k = torch.randn(H, H) * scale
    W_v = torch.randn(H, H) * scale
    W_o = torch.randn(H, H) * scale

    torch.manual_seed(5678)
    X_full = torch.randn(B, S, H)

    def attn_reference(X, Wq, Wk, Wv, Wo):
        b, s, h = X.shape
        q = project_to_heads(F.linear(X, Wq), num_heads, head_dim)
        k = project_to_heads(F.linear(X, Wk), num_heads, head_dim)
        v = project_to_heads(F.linear(X, Wv), num_heads, head_dim)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, s, h)
        return F.linear(a, Wo)

    ref_attn_out = attn_reference(X_full, W_q, W_k, W_v, W_o)

    mo.md(
        f"""
        Reference attention output computed.

        ```
        X_full        : {tuple(X_full.shape)}   (the full input)
        W_q/W_k/W_v   : {tuple(W_q.shape)}    (square QKV projections)
        W_o           : {tuple(W_o.shape)}    (output projection)
        ref_attn_out  : {tuple(ref_attn_out.shape)}   (what TSP must reproduce)
        ```
        """
    )
    return (
        B, S, H, num_heads, head_dim, D, local_seq, local_heads, local_hidden,
        W_q, W_k, W_v, W_o, X_full, attn_reference, ref_attn_out,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sharding: hand each rank its diagonal cell

        Now we split the data the way TSP does:

        - **Sequence shard** (the SP part): rank `p` gets tokens `[p·2 : (p+1)·2]`, so
          `X_p` is `[B, S/D, H]`. Activations shrink.
        - **Weight shard** (the TP part): rank `p` owns head `p`. For Q/K/V that's a
          **row slice** (column-parallel) `[H/D, H]`; for the output projection it's a
          **column slice** (row-parallel) `[H, H/D]`. Weights shrink.

        Each rank's tensors are a single entry in our simulated-group lists.
        """
    )
    return


@app.cell
def _(X_full, W_q, W_k, W_v, W_o, D, local_seq, head_dim, mo):
    def shard_col(W, r):
        "row slice -> head r's Q/K/V weights, shape [H/D, H]"
        start = r * head_dim
        return W[start : start + head_dim, :].contiguous()

    def shard_row(W, r):
        "column slice -> head r's output-proj weights, shape [H, H/D]"
        start = r * head_dim
        return W[:, start : start + head_dim].contiguous()

    # Per-rank lists. Index = rank.
    X_ranks = [X_full[:, p * local_seq : (p + 1) * local_seq, :].contiguous()
               for p in range(D)]
    Wq_ranks = [shard_col(W_q, r) for r in range(D)]
    Wk_ranks = [shard_col(W_k, r) for r in range(D)]
    Wv_ranks = [shard_col(W_v, r) for r in range(D)]
    Wo_ranks = [shard_row(W_o, r) for r in range(D)]

    mo.md(
        f"""
        Sharded across `D={D}` ranks:

        ```
        X_ranks[p]   : {tuple(X_ranks[0].shape)}   (each rank owns S/D = {local_seq} tokens)
        Wq_ranks[r]  : {tuple(Wq_ranks[0].shape)}   (each rank owns 1 head's Q proj)
        Wo_ranks[r]  : {tuple(Wo_ranks[0].shape)}   (each rank owns 1 head's O proj)
        ```

        The off-diagonal combinations — e.g. rank 0's tokens with rank 3's head — exist
        on *no* rank. The loop will reconstruct them.
        """
    )
    return X_ranks, Wq_ranks, Wk_ranks, Wv_ranks, Wo_ranks


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Algorithm 1: TSP attention, line by line

        Here is the algorithm from the paper. `r` is the weight-owner index (which head
        shard we're processing right now); `p` is a rank's sequence shard. We accumulate
        the output for our local tokens, `Y_p`, across all weight shards:

        ```text
        Y_p = 0
        for r in 0..D-1:                       # loop over every head/weight shard
            broadcast [Wq_r, Wk_r, Wv_r, Wo_r] # circulate shard r's weights to all
            Q_rp = X_p @ Wq_r.T                 # my tokens, shard r's query proj
            K_rp = X_p @ Wk_r.T
            V_rp = X_p @ Wv_r.T
            K_r, V_r = all_gather(K_rp, V_rp)   # gather K/V over SEQUENCE ranks p
            A_rp = causal_attn(Q_rp, K_r, V_r)  # my tokens attend to full seq, shard r
            Y_p += A_rp @ Wo_r.T                # accumulate shard r's contribution
        ```

        Let's run **one iteration** (`r=0`) and watch the shapes, before wiring up the
        full loop.
        """
    )
    return


@app.cell
def _(
    X_ranks, Wq_ranks, Wk_ranks, Wv_ranks, Wo_ranks,
    sim_broadcast, sim_all_gather, project_to_heads,
    F, torch, D, local_heads, head_dim, local_seq, local_hidden, B, mo,
):
    # --- one iteration of the loop, r = 0 ---
    r0 = 0
    Wq_r = sim_broadcast(Wq_ranks, src=r0)   # every rank now holds shard r0's weights
    Wk_r = sim_broadcast(Wk_ranks, src=r0)
    Wv_r = sim_broadcast(Wv_ranks, src=r0)
    Wo_r = sim_broadcast(Wo_ranks, src=r0)

    # each rank projects its OWN tokens through shard r0
    k_rp = [project_to_heads(F.linear(X_ranks[p], Wk_r[p]), local_heads, head_dim)
            for p in range(D)]
    v_rp = [project_to_heads(F.linear(X_ranks[p], Wv_r[p]), local_heads, head_dim)
            for p in range(D)]

    # all-gather K/V over the SEQUENCE ranks -> full sequence for head r0
    k_r = sim_all_gather(k_rp, dim=2)
    v_r = sim_all_gather(v_rp, dim=2)

    # rank 0 attends: its 2 tokens over the full 8-token K/V, with a causal mask
    q_r0p0 = project_to_heads(F.linear(X_ranks[0], Wq_r[0]), local_heads, head_dim)
    q_pos = torch.arange(0 * local_seq, 1 * local_seq)
    k_pos = torch.arange(local_seq * D)
    mask0 = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    a_r0p0 = F.scaled_dot_product_attention(q_r0p0, k_r[0], v_r[0], attn_mask=mask0)
    a_r0p0 = a_r0p0.transpose(1, 2).contiguous().view(B, local_seq, local_hidden)
    y_r0p0 = F.linear(a_r0p0, Wo_r[0])

    mo.md(
        f"""
        One iteration's shapes (from rank 0's point of view, head shard `r=0`):

        ```
        Wq_r[p]   broadcast : {tuple(Wq_r[0].shape)}   [H/D, H]  every rank has shard r now
        q_r0p0             : {tuple(q_r0p0.shape)}   [B, local_heads, S/D, head_dim]
        k_rp[p] (pre-gather): {tuple(k_rp[0].shape)}   each rank's own 2 tokens
        k_r[p]  (gathered) : {tuple(k_r[0].shape)}   [B, local_heads, S, head_dim]  full seq!
        a_r0p0  (attn out) : {tuple(a_r0p0.shape)}   [B, S/D, H/D]
        y_r0p0  (+= into Y): {tuple(y_r0p0.shape)}   [B, S/D, H]  contribution of head 0
        ```

        Note the asymmetry that *is* TSP: queries stay local (`S/D` tokens), but K/V are
        gathered to the **full** sequence so local tokens can attend to everything.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Correction #1 — there is NO final `all_reduce`.** You might expect one,
            like TP. There isn't. Each rank owns a *different* set of output tokens
            (it's sequence-sharded), so there's **nothing to sum across ranks**. The sum
            that *does* happen — over head shards — happens **locally**, as the
            `Y_p += …` accumulation inside the loop. The cross-rank reduction TP needed
            is replaced by a local sum over loop iterations.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Correction #2 — three typo-class bugs to avoid (all real, from the notes):**

            - **Include `W_o` in the broadcast.** It's easy to circulate only Q/K/V and
              forget the output-projection shard.
            - Use `causal_attn(Q, K, V)` — **not** `causal_attn(Q, K, K)`. The value
              argument must be `V`, not `K`.
            - For a fixed weight shard `r`, the `all_gather` is over the **sequence**
              ranks `p` — you're rebuilding the full key/value sequence, just like SP.
            """
        ),
        kind="danger",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The full TSP attention — and the proof it's correct

        Now the whole loop, across all simulated ranks. We concatenate the per-rank
        outputs back into a full `[B, S, H]` tensor and assert it equals the reference.
        This is the centerpiece.
        """
    )
    return


@app.cell
def _(
    sim_broadcast, sim_all_gather, project_to_heads, F, torch,
):
    def tsp_attn_sim(
        X_ranks_, Wq_ranks_, Wk_ranks_, Wv_ranks_, Wo_ranks_,
        world_size, local_heads_, head_dim_,
    ):
        B_, lseq, H_ = X_ranks_[0].shape
        total_seq = lseq * world_size
        lhidden = Wq_ranks_[0].shape[0]
        out_ranks = [torch.zeros_like(X_ranks_[p]) for p in range(world_size)]

        for r in range(world_size):
            # circulate shard r's weights (incl. W_o!) to every rank
            Wq_r_ = sim_broadcast(Wq_ranks_, src=r)
            Wk_r_ = sim_broadcast(Wk_ranks_, src=r)
            Wv_r_ = sim_broadcast(Wv_ranks_, src=r)
            Wo_r_ = sim_broadcast(Wo_ranks_, src=r)

            # each rank projects its own tokens through shard r
            k_rp_ = [project_to_heads(F.linear(X_ranks_[p], Wk_r_[p]), local_heads_, head_dim_)
                     for p in range(world_size)]
            v_rp_ = [project_to_heads(F.linear(X_ranks_[p], Wv_r_[p]), local_heads_, head_dim_)
                     for p in range(world_size)]

            # gather K/V over the sequence ranks -> full sequence for head r
            k_r_ = sim_all_gather(k_rp_, dim=2)
            v_r_ = sim_all_gather(v_rp_, dim=2)

            for p in range(world_size):
                q_rp_ = project_to_heads(
                    F.linear(X_ranks_[p], Wq_r_[p]), local_heads_, head_dim_
                )
                # each rank's tokens sit at absolute positions [p*lseq, (p+1)*lseq)
                q_pos_ = torch.arange(p * lseq, (p + 1) * lseq)
                k_pos_ = torch.arange(total_seq)
                mask_ = k_pos_.unsqueeze(0) <= q_pos_.unsqueeze(1)
                a_rp_ = F.scaled_dot_product_attention(
                    q_rp_, k_r_[p], v_r_[p], attn_mask=mask_
                )
                a_rp_ = a_rp_.transpose(1, 2).contiguous().view(B_, lseq, lhidden)
                out_ranks[p] = out_ranks[p] + F.linear(a_rp_, Wo_r_[p])  # local +=

        return out_ranks

    return (tsp_attn_sim,)


@app.cell
def _(
    tsp_attn_sim, X_ranks, Wq_ranks, Wk_ranks, Wv_ranks, Wo_ranks,
    D, local_heads, head_dim, ref_attn_out, torch, mo,
):
    tsp_out_ranks = tsp_attn_sim(
        X_ranks, Wq_ranks, Wk_ranks, Wv_ranks, Wo_ranks,
        world_size=D, local_heads_=local_heads, head_dim_=head_dim,
    )
    tsp_attn_out = torch.cat(tsp_out_ranks, dim=1)   # stitch sequence shards back

    torch.testing.assert_close(tsp_attn_out, ref_attn_out, rtol=1e-5, atol=1e-5)
    max_err_attn = (tsp_attn_out - ref_attn_out).abs().max().item()

    mo.callout(
        mo.md(
            f"""
            ✅ **TSP attention matches the single-process reference.**

            `torch.testing.assert_close` passed; max absolute difference =
            `{max_err_attn:.2e}`. The looped, weight-circulating, sequence-sharded
            algorithm produces *exactly* the same numbers as plain attention — with no
            final `all_reduce`.
            """
        ),
        kind="success",
    )
    return (tsp_attn_out,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## TSP for the MLP

        Same spirit, simpler: weight shards **rotate** across ranks while each rank
        accumulates the output for its *local tokens*. Because the MLP is
        token-independent, there's **no K/V gather** — but the weight movement and the
        local accumulation remain.

        We shard `W_in` `[I, H]` by rows (intermediate dim) and `W_out` `[H, I]` by
        columns. Then `Σ_r gelu(X_p @ W_in_r.T) @ W_out_r.T` equals the full MLP,
        because GELU is elementwise (a chunk of the activations is the activation of the
        chunk).
        """
    )
    return


@app.cell
def _(X_full, X_ranks, H, D, F, torch, sim_broadcast, mo):
    I = 32                              # intermediate size (divisible by D)
    p_I = I // D

    torch.manual_seed(99)
    s_mlp = H ** -0.5
    W_in = torch.randn(I, H) * s_mlp
    W_out = torch.randn(H, I) * (I ** -0.5)

    def mlp_reference(X, Win, Wout):
        return F.linear(F.gelu(F.linear(X, Win)), Wout)

    ref_mlp_out = mlp_reference(X_full, W_in, W_out)

    Win_ranks = [W_in[r * p_I : (r + 1) * p_I, :].contiguous() for r in range(D)]
    Wout_ranks = [W_out[:, r * p_I : (r + 1) * p_I].contiguous() for r in range(D)]

    def tsp_mlp_sim(X_ranks_, Win_ranks_, Wout_ranks_, world_size):
        out_ranks = [torch.zeros_like(X_ranks_[p]) for p in range(world_size)]
        for r in range(world_size):
            Win_r = sim_broadcast(Win_ranks_, src=r)     # rotate weight shard r
            Wout_r = sim_broadcast(Wout_ranks_, src=r)
            for p in range(world_size):
                hidden = F.gelu(F.linear(X_ranks_[p], Win_r[p]))
                out_ranks[p] = out_ranks[p] + F.linear(hidden, Wout_r[p])  # local +=
        return out_ranks

    tsp_mlp_ranks = tsp_mlp_sim(X_ranks, Win_ranks, Wout_ranks, D)
    tsp_mlp_out = torch.cat(tsp_mlp_ranks, dim=1)

    torch.testing.assert_close(tsp_mlp_out, ref_mlp_out, rtol=1e-5, atol=1e-5)
    max_err_mlp = (tsp_mlp_out - ref_mlp_out).abs().max().item()

    mo.callout(
        mo.md(
            f"""
            ✅ **TSP MLP matches the single-process reference.** Max abs diff =
            `{max_err_mlp:.2e}`. No gather needed — just rotate weights and accumulate
            locally. Shapes: `W_in_r` is `{tuple(Win_ranks[0].shape)}`, `W_out_r` is
            `{tuple(Wout_ranks[0].shape)}`.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tradeoff in three charts

        We've proven TSP is *correct*. Now the engineering question: what does it cost?
        These use the test's **real** numbers from `report.md` (8× H100, attention
        region, sweeping sequence length).
        """
    )
    return


@app.cell
def _():
    # Real benchmark numbers from report.md (attention region).
    seq_labels = ["8K", "16K", "32K", "64K"]
    mem_data = {            # peak memory per GPU, GiB (lower is better)
        "TP":  [1.85, 3.60, 7.10, 14.10],
        "SP":  [2.84, 5.59, 11.09, 22.09],
        "TSP": [1.58, 3.07, 6.14, 13.42],
    }
    lat_data = {            # slowest-rank latency, ms (lower is better)
        "TP":  [19, 53, 139, 583],
        "SP":  [61, 149, 450, 1733],
        "TSP": [35, 123, 285, 1893],
    }
    palette = {"TP": "#57b6f5", "SP": "#f0c674", "TSP": "#5fd38a"}
    return seq_labels, mem_data, lat_data, palette


@app.cell
def _(seq_labels, mem_data, palette, np, plt):
    def _draw_mem():
        fig_mem, ax_mem = plt.subplots(figsize=(8, 4))
        x_mem = np.arange(len(seq_labels))
        w_mem = 0.26
        for i, name in enumerate(["TP", "SP", "TSP"]):
            bars = ax_mem.bar(x_mem + (i - 1) * w_mem, mem_data[name], w_mem,
                              label=name, color=palette[name])
            for b in bars:
                ax_mem.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                            f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)
        ax_mem.set_xticks(x_mem)
        ax_mem.set_xticklabels(seq_labels)
        ax_mem.set_xlabel("sequence length")
        ax_mem.set_ylabel("peak memory / GPU (GiB)")
        ax_mem.set_title("Peak memory — TSP is lowest at every size (lower is better)")
        ax_mem.legend()
        ax_mem.spines[["top", "right"]].set_visible(False)
        fig_mem.tight_layout()
        return ax_mem

    _draw_mem()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        TSP (green) has the **shortest** memory bar everywhere — the paper's central
        claim, reproduced. Note the surprise: **SP (yellow) uses the *most* memory**,
        because this implementation `all_gather`s the whole K/V sequence onto every GPU.
        Now the time chart tells the other half of the story.
        """
    )
    return


@app.cell
def _(seq_labels, lat_data, palette, np, plt):
    def _draw_lat():
        fig_lat, ax_lat = plt.subplots(figsize=(8, 4))
        x_lat = np.arange(len(seq_labels))
        w_lat = 0.26
        for i, name in enumerate(["TP", "SP", "TSP"]):
            bars = ax_lat.bar(x_lat + (i - 1) * w_lat, lat_data[name], w_lat,
                              label=name, color=palette[name])
            for b in bars:
                ax_lat.text(b.get_x() + b.get_width() / 2, b.get_height() + 15,
                            f"{int(b.get_height())}", ha="center", va="bottom", fontsize=7)
        ax_lat.set_xticks(x_lat)
        ax_lat.set_xticklabels(seq_labels)
        ax_lat.set_xlabel("sequence length")
        ax_lat.set_ylabel("latency (ms)")
        ax_lat.set_title("Latency — TSP pays for its memory win in time (lower is better)")
        ax_lat.legend()
        ax_lat.spines[["top", "right"]].set_visible(False)
        fig_lat.tight_layout()
        return ax_lat

    _draw_lat()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **TP is consistently fastest** — it does the least communication. TSP's
        un-overlapped weight broadcasts make it the slowest at the largest size. Space
        for time: that's the whole decision.

        ## Interactive: explore one sequence length at a time
        """
    )
    return


@app.cell
def _(mo, seq_labels):
    size_picker = mo.ui.dropdown(
        options=seq_labels, value="32K", label="Sequence length"
    )
    size_picker
    return (size_picker,)


@app.cell
def _(size_picker, seq_labels, mem_data, lat_data, palette, plt, mo):
    def _draw_pick():
        idx = seq_labels.index(size_picker.value)
        names = ["TP", "SP", "TSP"]
        colors = [palette[n] for n in names]

        fig_pick, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.8))
        mem_vals = [mem_data[n][idx] for n in names]
        lat_vals = [lat_data[n][idx] for n in names]

        bars_a = ax_a.bar(names, mem_vals, color=colors)
        ax_a.set_title(f"Peak memory @ {size_picker.value} (GiB)")
        ax_a.set_ylabel("GiB")
        for b in bars_a:
            ax_a.text(b.get_x() + b.get_width() / 2, b.get_height(),
                      f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)

        bars_b = ax_b.bar(names, lat_vals, color=colors)
        ax_b.set_title(f"Latency @ {size_picker.value} (ms)")
        ax_b.set_ylabel("ms")
        for b in bars_b:
            ax_b.text(b.get_x() + b.get_width() / 2, b.get_height(),
                      f"{int(b.get_height())}", ha="center", va="bottom", fontsize=8)

        for ax in (ax_a, ax_b):
            ax.spines[["top", "right"]].set_visible(False)
        fig_pick.suptitle(
            f"At {size_picker.value}: TSP cheapest in memory, near the top in time",
            fontsize=11, weight="bold",
        )
        fig_pick.tight_layout()
        return ax_a

    _draw_pick()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### And how memory scales with `D`

        TSP's promise is that *both* memory classes scale as `≈1/D` at once. Drag the
        slider to see the theoretical curve (relative to a 1-GPU baseline). TP+SP, with
        a fixed budget `D = T·Σ`, can only split the reduction between the two axes — it
        can't drive both to `1/D`.
        """
    )
    return


@app.cell
def _(mo):
    d_slider = mo.ui.slider(1, 16, value=4, label="D (number of GPUs)")
    d_slider
    return (d_slider,)


@app.cell
def _(d_slider, np, plt, mo):
    d_val = d_slider.value
    ds = np.arange(1, 17)

    fig_scale, ax_scale = plt.subplots(figsize=(8, 4))
    ax_scale.plot(ds, 1 / ds, marker="o", color="#5fd38a",
                  label="TSP: weights AND activations ≈ 1/D")
    # TP+SP best-balanced split: T = Σ = sqrt(D) -> each class ~ 1/sqrt(D)
    ax_scale.plot(ds, 1 / np.sqrt(ds), marker="s", color="#57b6f5", linestyle="--",
                  label="TP+SP (balanced): each class ≈ 1/√D")
    ax_scale.axvline(d_val, color="#f0986b", linestyle=":", linewidth=1.5)
    ax_scale.scatter([d_val], [1 / d_val], color="#5fd38a", s=90, zorder=5)
    ax_scale.annotate(f"D={d_val}: TSP → {1/d_val:.3f}×",
                      (d_val, 1 / d_val), textcoords="offset points",
                      xytext=(10, 12), fontsize=9, color="#2b7a4b")
    ax_scale.set_xlabel("D (GPUs)")
    ax_scale.set_ylabel("memory fraction vs 1 GPU (lower is better)")
    ax_scale.set_title("Why TSP exists: both memory classes fall as 1/D on one budget")
    ax_scale.legend()
    ax_scale.spines[["top", "right"]].set_visible(False)
    fig_scale.tight_layout()
    ax_scale
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **The honest tradeoff.** TSP wins on memory and pays in communication/time.
            It `broadcast`s weight shards *during every forward pass* **and**
            `all_gather`s K/V — once per loop iteration. And because each rank touches
            every head shard, it doesn't even get TP's compute savings, only the memory
            savings.

            This teaching implementation is deliberately the *floor*, not a verdict: it
            **skips** async overlap of weight movement, keeping shards resident, and
            ring-attention for K/V. Each of those would narrow the runtime gap. The
            memory win, though, is structural and real regardless.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Check your understanding""")
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Why does TSP attention NOT end with an `all_reduce`, even though it sums over head shards?":
            mo.md(
                r"""
                Because each rank owns a **different set of output tokens** (it's
                sequence-sharded), so there is nothing to reduce *across ranks*. The
                sum over head shards happens **locally**, as the `Y_p += …`
                accumulation inside the loop — each rank walks every head shard itself
                and adds the contributions to its own tokens. The cross-rank reduction
                TP needs is replaced by a local sum over loop iterations.
                """
            ),
            "TSP vs TP+SP, in one line?":
            mo.md(
                r"""
                **TP+SP** is a 2D grid of `T·Σ` GPUs where every (head, token) cell
                physically exists (no weight movement). **TSP** folds both roles onto
                the **same `D` GPUs** (the diagonal) and reconstructs the off-diagonal
                combinations by circulating weight shards in a loop — fewer GPUs for the
                same two reductions, paid for in weight traffic.
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

        - **TSP ≠ TP+SP.** TP+SP is a filled 2D grid (`T·Σ` GPUs); TSP folds both onto
          the **diagonal** of the *same* `D` GPUs.
        - The **loop over weight shards** is the defining feature — it reconstructs the
          missing off-diagonal (token, head) combinations by broadcasting each shard.
        - **Algorithm 1**, implemented faithfully: broadcast `[Wq, Wk, Wv, Wo]`, project
          local tokens, `all_gather` K/V over the sequence, causal-attend, and
          accumulate `Y_p += A_rp @ Wo_r.T` — **no final `all_reduce`**.
        - The **MLP** version is the same idea without the K/V gather.
        - Both match a single-process reference exactly (`torch.testing.assert_close`).
        - The tradeoff: **TSP wins memory, costs time.** Its benchmark timings are an
          un-optimized floor.

        **Course chapters:** `../course/07-tensor-sequence-parallelism.html` (the
        algorithm) and `../course/08-tradeoffs.html` (the benchmark numbers and the
        decision guide).
        """
    )
    return


if __name__ == "__main__":
    app.run()
