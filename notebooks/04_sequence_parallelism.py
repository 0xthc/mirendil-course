import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Sequence parallelism (SP), built from scratch

        Tensor parallelism (TP) splits the **weights** across GPUs. Sequence
        parallelism is the mirror image: every GPU keeps a **full copy of the
        weights**, but each one only handles a **chunk of the tokens**. That is
        exactly what kills *activation* memory — nobody holds the whole sequence.

        **What you'll build, step by step, on a tiny CPU example:**

        1. Shard an input `X = [B, S, H]` along the **sequence (token) axis** into
           per-rank chunks `X_p = [B, S/D, H]`.
        2. Prove the **MLP needs zero communication** under SP (it's token-wise):
           run it per-chunk, concatenate, and match a single-process reference.
        3. See **why attention is hard**: a query needs the keys/values of *all*
           earlier tokens, which live on *other* ranks.
        4. Implement the **K/V `all_gather`** that rebuilds the full keys/values on
           every rank while keeping queries local.
        5. Get the **causal mask offset right** (the subtle bug) so the sharded
           output *exactly equals* the reference.
        6. Measure **causal load imbalance** and fix it with **zigzag** splitting.

        Everything is simulated in ONE process: `D` ranks are just a Python list of
        tensors, and the collectives (`broadcast`, `all_reduce`, `all_gather`) are
        plain functions. We check every step against a single-process reference with
        `torch.testing.assert_close`.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5.** Give each reader (GPU) a different stack of **pages** from the
            same book — and every reader keeps their own full copy of the dictionary
            (the weights). Now the activations shrink, because nobody holds the whole
            book at once.

            But there's a twist. The task is *"summarize everything up to your page."*
            To do that, a reader needs the earlier pages held by other readers — so
            they pass their pages around. That passing is the **K/V `all_gather`**.

            The private thinking step (the **MLP**) needs no passing at all: each
            reader can think about their own pages alone.
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

    torch.manual_seed(0)
    return F, mo, np, plt, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. The setup: tiny tensors, replicated weights

        We use deliberately tiny shapes so every cell runs instantly and the
        sharding is easy to see by eye:

        - `B = 1` (batch)
        - `S = 8` (sequence length / number of tokens)
        - `D = 4` ranks (simulated GPUs) → each rank holds `S/D = 2` tokens
        - `H = 16` (hidden size), `heads = 4` → `head_dim = 4`

        The weights `W_q, W_k, W_v, W_o` are **replicated**: every rank stores the
        full matrix. Under SP we only shard the *activations*, never the weights.
        Recall the convention `F.linear(X, W)` computes `X @ W.T`, with weights
        stored `[out_features, in_features]`.
        """
    )
    return


@app.cell
def _(mo, torch):
    B, S, H = 1, 8, 16
    D = 4               # number of simulated ranks (GPUs)
    num_heads = 4
    head_dim = H // num_heads
    local_S = S // D    # tokens per rank = 2

    def init_weight(shape):
        # small fan-in scaled init, like the real repo
        return torch.randn(*shape) * (shape[-1] ** -0.5)

    torch.manual_seed(1234)
    W_q = init_weight((H, H))
    W_k = init_weight((H, H))
    W_v = init_weight((H, H))
    W_o = init_weight((H, H))

    torch.manual_seed(5678)
    X = torch.randn(B, S, H)

    mo.md(
        f"""
        Created the full (un-sharded) input and the replicated weights:

        ```
        X      : {tuple(X.shape)}      # [B, S, H] — the whole sequence
        W_q/k/v: {tuple(W_q.shape)}     # [H, H] — replicated on every rank
        W_o    : {tuple(W_o.shape)}     # [H, H] — replicated on every rank
        D = {D} ranks, local_S = S/D = {local_S} tokens per rank, head_dim = {head_dim}
        ```
        """
    )
    return B, D, H, S, W_k, W_o, W_q, W_v, X, head_dim, local_S, num_heads


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Shard the sequence axis

        Splitting along `S` means rank `p` gets a contiguous block of tokens:
        rank 0 gets tokens `0,1`, rank 1 gets `2,3`, and so on. Each rank's local
        input is `X_p = [B, S/D, H]`.

        Because activations scale with the number of tokens a rank processes, and
        each rank now processes only `S/D` of them, **activation memory drops by a
        factor of D**. That's the win TP can't give you.
        """
    )
    return


@app.cell
def _(D, X, mo):
    def shard_sequence(tensor, n_ranks):
        "split [B, S, H] into a list of n_ranks contiguous [B, S/n, H] chunks along dim=1"
        return list(tensor.chunk(n_ranks, dim=1))

    X_chunks = shard_sequence(X, D)

    _lines = "\n".join(
        f"rank {p}: X_{p} shape {tuple(c.shape)}  (tokens {p * c.shape[1]}–{p * c.shape[1] + c.shape[1] - 1})"
        for p, c in enumerate(X_chunks)
    )
    mo.md(
        f"""
        `X` is now a list of {D} per-rank chunks. Same total data, just split across
        ranks along the token axis:

        ```
        {_lines}
        ```
        """
    )
    return X_chunks, shard_sequence


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. The collectives, simulated in one process

        We can't spin up 4 real GPUs in a notebook, so we represent the ranks as a
        Python **list of tensors** and implement the three collective operations as
        pure functions. These are the only communication primitives we'll need.
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

    return sim_all_gather, sim_all_reduce, sim_broadcast


@app.cell
def _(F, torch):
    # --- single-process reference implementations (the "ground truth") ---

    def to_heads(projected, num_heads, head_dim):
        "[B, S, H] -> [B, num_heads, S, head_dim]"
        B, S, _H = projected.shape
        return projected.view(B, S, num_heads, head_dim).transpose(1, 2)

    def ref_attn(X, W_q, W_k, W_v, W_o, num_heads):
        "plain single-GPU causal multi-head attention"
        B, S, H = X.shape
        head_dim = H // num_heads
        q = to_heads(F.linear(X, W_q), num_heads, head_dim)
        k = to_heads(F.linear(X, W_k), num_heads, head_dim)
        v = to_heads(F.linear(X, W_v), num_heads, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        return F.linear(out, W_o)

    def ref_mlp(X, W_in, W_out):
        "plain single-GPU token-wise MLP"
        return F.linear(F.gelu(F.linear(X, W_in)), W_out)

    return ref_attn, ref_mlp, to_heads


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. The MLP is trivial under SP (no communication)

        The MLP is **token-independent**: each token's output depends only on that
        token. So a rank holding tokens `0,1` can compute their full MLP outputs
        alone — it never needs another rank's tokens. **Zero communication.**

        Let's prove it: run the MLP on each rank's chunk, concatenate the results,
        and check it equals the single-process reference run on the whole sequence.
        """
    )
    return


@app.cell
def _(F, H, X, X_chunks, mo, ref_mlp, torch):
    # MLP weights (replicated on every rank). I = 4*H is the usual expansion.
    I = 4 * H
    torch.manual_seed(99)
    W_in = torch.randn(I, H) * (H ** -0.5)     # [I, H]
    W_out = torch.randn(H, I) * (I ** -0.5)    # [H, I]

    def sp_mlp_per_rank(x_chunks, W_in, W_out):
        "each rank runs the FULL mlp on its own token chunk — no collectives"
        return [F.linear(F.gelu(F.linear(X_p, W_in)), W_out) for X_p in x_chunks]

    mlp_out_chunks = sp_mlp_per_rank(X_chunks, W_in, W_out)
    mlp_sp = torch.cat(mlp_out_chunks, dim=1)          # stitch tokens back together
    mlp_reference = ref_mlp(X, W_in, W_out)            # single-process ground truth

    torch.testing.assert_close(mlp_sp, mlp_reference, rtol=1e-5, atol=1e-5)
    mlp_ok = True

    mo.md(
        f"""
        ```
        per-rank MLP outputs : {[tuple(c.shape) for c in mlp_out_chunks]}
        concatenated (SP)    : {tuple(mlp_sp.shape)}
        single-process ref   : {tuple(mlp_reference.shape)}
        max abs difference   : {(mlp_sp - mlp_reference).abs().max().item():.2e}
        ```
        """
    )
    return mlp_ok,


@app.cell
def _(mlp_ok, mo):
    mo.callout(
        mo.md(
            f"""
            ✅ **SP MLP matches the reference exactly** (assert passed: `{mlp_ok}`).
            Each rank ran the full MLP on its own tokens, with **no `all_gather`,
            no `all_reduce`** — nothing crossed between ranks. This is the easy half
            of SP.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Why attention is hard

        Attention is **token-dependent**. With causal masking, a query at token `i`
        must attend to the keys and values of *every* token `0…i`. But under SP
        those earlier tokens live on *other* ranks.

        Concretely, with `S=8` and `D=4` (2 tokens per rank):

        ```
        rank 0  holds tokens 0,1
        rank 1  holds tokens 2,3   <- query token 3 needs keys/values 0,1,2,3
        rank 2  holds tokens 4,5        ...but 0,1 are on rank 0, and 2 is local
        rank 3  holds tokens 6,7
        ```

        Rank 1's queries need keys/values for tokens `0,1` (on rank 0) plus its own
        `2,3`. **Local queries need global keys and values.** So before computing
        attention, every rank must collect everyone's K/V chunks — an `all_gather`
        along the sequence axis. Queries stay local.
        """
    )
    return


@app.cell
def _(D, S, local_S, np, plt):
    # Diagram: gather K/V to full length, keep Q local.
    fig_g, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(11, 3.2))
    sp_color = "#f0c674"
    act_color = "#f0986b"

    def draw_row(ax, title, q_full, kv_full):
        ax.set_title(title, fontsize=11)
        for p in range(D):
            tok_lo, tok_hi = p * local_S, (p + 1) * local_S - 1
            # Q box
            q_lab = f"{tok_lo}-{tok_hi}" if not q_full else "0-7"
            ax.add_patch(plt.Rectangle((p, 1.05), 0.9, 0.8, color=sp_color))
            ax.text(p + 0.45, 1.45, f"Q\n{tok_lo}-{tok_hi}", ha="center", va="center", fontsize=8)
            # K/V box
            kv_lab = "0-7" if kv_full else f"{tok_lo}-{tok_hi}"
            ax.add_patch(plt.Rectangle((p, 0.05), 0.9, 0.8, color=act_color))
            ax.text(p + 0.45, 0.45, f"K,V\n{kv_lab}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if kv_full else "normal")
            ax.text(p + 0.45, 1.95, f"rank {p}", ha="center", va="center", fontsize=8)
        ax.set_xlim(-0.1, D)
        ax.set_ylim(-0.1, 2.2)
        ax.axis("off")

    draw_row(ax_before, "Before: each rank has only its local K/V", False, False)
    draw_row(ax_after, "After all_gather(K), all_gather(V): full K/V everywhere", False, True)
    fig_g.suptitle("Gather K/V to full length, keep Q local", fontsize=12)
    fig_g.tight_layout()
    ax_after
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. The K/V `all_gather`

        Each rank projects its local chunk to per-head `q_p, k_p, v_p` of shape
        `[B, heads, S/D, head_dim]`. Then we `all_gather` only `k_p` and `v_p` along
        the sequence axis (`dim=2`) so every rank holds the **full** K/V of shape
        `[B, heads, S, head_dim]`. Queries are left local.
        """
    )
    return


@app.cell
def _(
    F,
    W_k,
    W_q,
    W_v,
    X_chunks,
    head_dim,
    mo,
    num_heads,
    sim_all_gather,
    to_heads,
):
    # Project each rank's chunk, then gather K/V to full sequence length.
    q_local = [to_heads(F.linear(X_p, W_q), num_heads, head_dim) for X_p in X_chunks]
    k_local = [to_heads(F.linear(X_p, W_k), num_heads, head_dim) for X_p in X_chunks]
    v_local = [to_heads(F.linear(X_p, W_v), num_heads, head_dim) for X_p in X_chunks]

    K_full = sim_all_gather(k_local, dim=2)   # list; each [B, heads, S, head_dim]
    V_full = sim_all_gather(v_local, dim=2)

    mo.md(
        f"""
        ```
        q_local[p] (stays local) : {tuple(q_local[0].shape)}   # [B, heads, S/D, head_dim]
        k_local[p] (before gather): {tuple(k_local[0].shape)}   # [B, heads, S/D, head_dim]
        K_full[p]  (after gather) : {tuple(K_full[0].shape)}   # [B, heads, S,   head_dim]
        V_full[p]  (after gather) : {tuple(V_full[0].shape)}   # [B, heads, S,   head_dim]
        ```

        The sequence axis on K/V grew from `S/D = 2` back to the full `S = 8`. Q kept
        its local length of `2`. Each rank now has what it needs to answer *its own*
        query rows against the *whole* key/value history.
        """
    )
    return K_full, V_full, q_local


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. The causal mask subtlety (the real bug)

        Here's where it's easy to go wrong. `is_causal=True` assumes local query
        row `i` may see key columns `0…i`. But on rank 1, **local query row 0 is
        really global token 2** — it should see keys `0,1,2`, not just key `0`.

        So the mask must use **rank-relative query positions against global key
        positions**:

        ```
        q_positions = arange(p*local_S, (p+1)*local_S)   # GLOBAL token ids for this rank
        k_positions = arange(S)                           # all global tokens
        mask = k_positions <= q_positions                 # [local_S, S] boolean
        ```

        The naive version forgets the `p*local_S` offset and uses `arange(local_S)`,
        which silently lets later ranks attend to the *wrong* tokens. Let's build
        both and see the difference.
        """
    )
    return


@app.cell
def _(S, local_S, torch):
    def causal_mask(rank, local_S, S, offset=True):
        "[local_S, S] boolean mask; True = query may attend to that key"
        if offset:
            q_pos = torch.arange(rank * local_S, (rank + 1) * local_S)
        else:
            q_pos = torch.arange(local_S)               # BUG: forgets rank offset
        k_pos = torch.arange(S)
        return k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

    # Inspect rank 1 (the first rank where offset actually matters).
    mask_correct_r1 = causal_mask(1, local_S, S, offset=True)
    mask_naive_r1 = causal_mask(1, local_S, S, offset=False)
    return causal_mask, mask_correct_r1, mask_naive_r1


@app.cell
def _(mask_correct_r1, mask_naive_r1, np, plt):
    fig_m, (ax_n, ax_c) = plt.subplots(1, 2, figsize=(10, 2.6))

    for _ax, _mask, _title in (
        (ax_n, mask_naive_r1, "WRONG: naive mask (no offset)"),
        (ax_c, mask_correct_r1, "CORRECT: offset mask"),
    ):
        _ax.imshow(_mask.numpy().astype(int), cmap="YlGn", vmin=0, vmax=1, aspect="auto")
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("global key position 0–7")
        _ax.set_ylabel("local query row")
        _ax.set_xticks(range(_mask.shape[1]))
        _ax.set_yticks(range(_mask.shape[0]))
        for _i in range(_mask.shape[0]):
            for _j in range(_mask.shape[1]):
                _ax.text(_j, _i, int(_mask[_i, _j].item()), ha="center", va="center", fontsize=7)

    fig_m.suptitle("Rank 1 causal mask (holds global tokens 2,3)", fontsize=12)
    fig_m.tight_layout()
    ax_c
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            Look at the heatmaps. On **rank 1**, local query row 0 is global token 2,
            so the **correct** mask opens keys `0,1,2`. The **naive** mask treats it
            as token 0 and opens only key `0` — it drops the two earlier tokens
            entirely. That's a silent correctness bug: training still runs, the model
            just attends to the wrong history.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8. Full SP attention, end to end

        Now assemble it: per rank, project → `all_gather` K/V → masked attention for
        the local queries → `W_o`. Concatenate the per-rank outputs and compare to
        the single-process reference. We run it **both** ways — with the correct
        offset mask and with the buggy naive mask — to show only the offset version
        matches.

        Note the asymmetry with TP: SP attention ends with **no `all_reduce`**. Each
        rank owns a *disjoint set of output tokens*, so its output is already final.
        The communication moved to the *middle* (the K/V gather).
        """
    )
    return


@app.cell
def _(
    B,
    F,
    K_full,
    S,
    V_full,
    W_o,
    X,
    causal_mask,
    head_dim,
    local_S,
    num_heads,
    q_local,
    ref_attn,
    torch,
    W_k,
    W_q,
    W_v,
):
    def sp_attention(q_local, K_full, V_full, W_o, offset):
        "per-rank: masked attention of local Q against global K/V, then W_o"
        outs = []
        D = len(q_local)
        for p in range(D):
            mask = causal_mask(p, local_S, S, offset=offset)
            attn_p = F.scaled_dot_product_attention(
                q_local[p], K_full[p], V_full[p], attn_mask=mask
            )
            attn_p = attn_p.transpose(1, 2).contiguous().view(B, local_S, num_heads * head_dim)
            outs.append(F.linear(attn_p, W_o))
        return outs

    attn_reference = ref_attn(X, W_q, W_k, W_v, W_o, num_heads)

    out_offset = torch.cat(sp_attention(q_local, K_full, V_full, W_o, offset=True), dim=1)
    out_naive = torch.cat(sp_attention(q_local, K_full, V_full, W_o, offset=False), dim=1)

    naive_gap = (out_naive - attn_reference).abs().max().item()
    offset_gap = (out_offset - attn_reference).abs().max().item()

    torch.testing.assert_close(out_offset, attn_reference, rtol=1e-5, atol=1e-5)
    sp_attn_ok = True
    return naive_gap, offset_gap, sp_attn_ok


@app.cell
def _(mo, naive_gap, offset_gap, sp_attn_ok):
    mo.callout(
        mo.md(
            f"""
            ✅ **SP attention with the offset mask matches the reference exactly.**

            ```
            max |offset_SP  - reference| = {offset_gap:.2e}   <- assert passed: {sp_attn_ok}
            max |naive_SP   - reference| = {naive_gap:.2e}   <- WRONG (offset bug)
            ```

            The offset version is correct to floating-point tolerance; the naive
            version is off by a large margin because later ranks dropped their earlier
            tokens. Same code path, one missing `p*local_S` offset.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 9. Causal load imbalance (and the zigzag fix)

        A subtle performance problem shows up only with **causal attention + sequence
        sharding**. Attention cost grows with token position: token `t` attends to
        `t+1` keys. So if you split the sequence into **contiguous** blocks, the rank
        holding the *latest* tokens does far more work — and everyone waits for the
        slowest rank.

        The fix is **zigzag / striped** partitioning: pair an early chunk with a late
        chunk so each rank gets a balanced mix. Same total work, evenly shared — like
        *dealing cards* instead of *cutting the deck*.
        """
    )
    return


@app.cell
def _():
    def work_contiguous(S, D):
        "attention work per rank for contiguous splitting; token t costs t+1"
        L = S // D
        return [sum(t + 1 for t in range(p * L, (p + 1) * L)) for p in range(D)]

    def work_zigzag(S, D):
        "pair chunk p with its mirror chunk (2D-1-p) to balance causal load"
        c = S // (2 * D)  # requires S divisible by 2*D
        works = []
        for p in range(D):
            toks = list(range(p * c, (p + 1) * c)) + list(
                range((2 * D - 1 - p) * c, (2 * D - p) * c)
            )
            works.append(sum(t + 1 for t in toks))
        return works

    return work_contiguous, work_zigzag


@app.cell
def _(D, np, plt, work_contiguous, work_zigzag):
    S_demo = 32  # larger than the toy S so the imbalance is vivid (and 32 % (2*4) == 0)
    cont = work_contiguous(S_demo, D)
    zig = work_zigzag(S_demo, D)

    fig_b, ax_b = plt.subplots(figsize=(7, 3.6))
    xs = np.arange(D)
    w = 0.38
    bars_c = ax_b.bar(xs - w / 2, cont, w, label="Contiguous split", color="#f0986b")
    bars_z = ax_b.bar(xs + w / 2, zig, w, label="Zigzag split", color="#5fd38a")
    for _group in (bars_c, bars_z):
        for _bar in _group:
            ax_b.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 1,
                      f"{int(_bar.get_height())}", ha="center", va="bottom", fontsize=8)
    ax_b.set_title(f"Attention work per rank (S={S_demo}, D={D})")
    ax_b.set_xlabel("rank")
    ax_b.set_ylabel("relative attention work (key comparisons)")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([f"rank {p}" for p in range(D)])
    ax_b.legend()
    ratio = max(cont) / min(cont)
    ax_b.annotate(f"contiguous: rank {D-1} does {ratio:.0f}× rank 0",
                  xy=(D - 1 - w / 2, max(cont)), xytext=(0.2, max(cont) * 0.92),
                  fontsize=9, color="#b5530a")
    fig_b.tight_layout()
    ax_b
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            Orange (contiguous): the last rank does several times the work of the
            first, and the whole step waits for it. Green (zigzag): identical total
            work, split evenly, so no rank sits idle. This is the same trick as
            dealing pages out like cards instead of handing one reader the whole last
            chapter.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 10. Interactive: sweep the sequence length

        Drag the slider to change `S` (with `D = 4` ranks fixed). Watch two things:
        the **load-imbalance** bars (contiguous vs zigzag), and the **activation
        memory** per rank (single-GPU vs SP). SP always stores `1/D` of the tokens'
        activations — the win grows with `S`.
        """
    )
    return


@app.cell
def _(mo):
    s_slider = mo.ui.slider(8, 64, step=8, value=32, label="S (sequence length)")
    s_slider
    return s_slider,


@app.cell
def _(D, np, plt, s_slider, work_contiguous, work_zigzag):
    S_i = s_slider.value
    cont_i = work_contiguous(S_i, D)
    zig_i = work_zigzag(S_i, D)

    fig_i, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.6))

    # Left: load imbalance redraws with S
    xs_i = np.arange(D)
    wi = 0.38
    axL.bar(xs_i - wi / 2, cont_i, wi, label="Contiguous", color="#f0986b")
    axL.bar(xs_i + wi / 2, zig_i, wi, label="Zigzag", color="#5fd38a")
    axL.set_title(f"Load imbalance (S={S_i}, D={D})")
    axL.set_xlabel("rank")
    axL.set_ylabel("attention work")
    axL.set_xticks(xs_i)
    axL.legend(fontsize=8)

    # Right: activation memory per rank (elements proportional to tokens held)
    single_gpu = S_i           # one GPU holds all S tokens' activations
    sp_per_rank = S_i / D      # SP rank holds S/D
    axR.bar(["Single GPU", "SP (per rank)"], [single_gpu, sp_per_rank],
            color=["#57b6f5", "#f0c674"])
    axR.set_title(f"Activation memory ∝ tokens held (D={D})")
    axR.set_ylabel("relative activation size")
    for _idx, _val in enumerate([single_gpu, sp_per_rank]):
        axR.text(_idx, _val + S_i * 0.01, f"{_val:.0f}", ha="center", va="bottom", fontsize=9)
    axR.annotate(f"{D}× smaller", xy=(1, sp_per_rank), xytext=(0.55, single_gpu * 0.5),
                 fontsize=10, color="#b5750a",
                 arrowprops=dict(arrowstyle="->", color="#b5750a"))

    fig_i.tight_layout()
    axR
    return


@app.cell
def _(D, S, np, plt):
    # Static activation-memory chart for the toy config, with the K/V-gather caveat.
    fig_a, ax_a = plt.subplots(figsize=(6.5, 3.4))
    labels = ["Single GPU\n(all activations)", "SP per rank\n(X, Q local)", "SP per rank\n+ gathered K/V"]
    # X/Q activations ∝ S/D; gathered K/V swell back to ∝ S (×2 for K and V).
    single = S
    sp_local = S / D
    sp_with_kv = S / D + 2 * S   # local X/Q (S/D) plus full K and V (S each)
    vals = [single, sp_local, sp_with_kv]
    colors = ["#57b6f5", "#f0c674", "#f0986b"]
    bars_a = ax_a.bar(labels, vals, color=colors)
    for _bar, _v in zip(bars_a, vals):
        ax_a.text(_bar.get_x() + _bar.get_width() / 2, _v + S * 0.02, f"{_v:.0f}",
                  ha="center", va="bottom", fontsize=9)
    ax_a.set_title(f"Activation memory under SP (S={S}, D={D})")
    ax_a.set_ylabel("relative activation size")
    fig_a.tight_layout()
    ax_a
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **SP's blind spot.** The `X`/`Q` activations shrink by `D`, but the K/V
            `all_gather` swells the keys and values back to full length `S` on *every*
            rank — and every rank still stores *all* the weights. So this simple SP
            actually spends a lot of memory on gathered K/V and replicated weights.
            TP had the opposite blind spot. That's the whole motivation for combining
            them (tensor-sequence parallelism, module 07).
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 11. Check your understanding""")
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Why does the SP MLP need no communication, but SP attention needs an all_gather?": mo.md(
                r"""
                The MLP is **token-independent**: each token's output depends only on
                itself, so a rank holding tokens 0–1 can fully compute their MLP
                outputs alone. Attention is token-**dependent**: a causal query must
                attend to all earlier tokens' keys/values, which live on other ranks —
                so the K/V chunks must be `all_gather`ed before attention can run.
                """
            ),
            "Why is a contiguous causal split load-imbalanced, and what fixes it?": mo.md(
                r"""
                Causal attention cost grows with token position (token `t` attends to
                `t+1` keys). With a contiguous split, the rank holding the latest
                tokens does the most work and everyone waits on it. **Zigzag/striped**
                partitioning pairs early chunks with late chunks so each rank gets a
                balanced mix — same total work, evenly shared.
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

        - **SP shards the token axis**: `X_p = [B, S/D, H]`. Weights stay
          **replicated**; only activations are split, so activation memory drops by
          `D`.
        - **The MLP is free under SP** — token-wise, so each rank runs the full MLP
          on its chunk with *no* communication. We verified it matches the reference.
        - **Attention needs a K/V `all_gather`**: local queries need the *global*
          keys/values, so K/V are gathered to full length while Q stays local.
        - **The causal mask must use the rank offset** (`q_pos = arange(p*S/D, …)`
          vs global `k_pos`). Forget it and later ranks silently attend to the wrong
          tokens — we showed the buggy mask fails and the offset mask matches exactly.
        - **SP attention ends with no `all_reduce`**: each rank owns disjoint output
          tokens. Communication moved to the middle.
        - **Contiguous causal splits are load-imbalanced**; **zigzag** fixes it.
        - **SP's blind spot**: gathered K/V and replicated weights — fixed by
          combining with TP.

        Companion course chapter: `../course/06-sequence-parallelism.html`.
        Next up: tensor-sequence parallelism in `../course/07-tensor-sequence-parallelism.html`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
