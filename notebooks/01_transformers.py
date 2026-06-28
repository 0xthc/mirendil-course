import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 01 · How a transformer computes

        This notebook builds a tiny transformer **by hand**, end to end, on the CPU —
        small enough to print every tensor and watch the numbers appear. By the time
        you reach the bottom you will have:

        - turned text into integer **token ids**,
        - looked them up in an **embedding table** to get the first activation,
        - pushed that tensor down the **residual stream**,
        - computed an **MLP** from scratch (two matmuls + a nonlinearity),
        - **stacked** embed → (attention + MLP) × L → un-embed into next-token scores,
        - and **counted** the parameters and activations so the memory story is concrete.

        The whole course rests on one distinction: **weights vs. activations**. This
        notebook is where you *feel* it. We will label every tensor as one or the
        other, relentlessly.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **ELI5 — recipes vs. ingredients on the stove**

            A transformer is a kitchen. The **weights** are the *recipes* — printed
            once, the same for every customer, sitting in a binder on the shelf. They
            don't get bigger when more orders come in. The **activations** are the
            *ingredients currently on the stove* for the order you're cooking right
            now — they appear when an order arrives and disappear when the plate goes
            out. Order a banquet (a long sequence) and the stove fills up; the recipe
            binder is unchanged.

            Same split as a web server: compiled code + loaded config (weights) is
            fixed; per-request scratch memory (activations) comes and goes. Hold this
            picture — it *is* the weights/activations distinction.
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

    # Shared palette (from the authoring guide) so every chart matches.
    COL_WEIGHT = "#c099f0"   # weights (recipes)
    COL_ACT = "#f0986b"      # activations (ingredients on the stove)
    COL_ATTN = "#57b6f5"     # attention component
    COL_MLP = "#5fd38a"      # mlp component
    COL_ACCENT = "#f0c674"   # highlight
    return COL_ACCENT, COL_ACT, COL_ATTN, COL_MLP, COL_WEIGHT, F, mo, np, plt, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The toy scale we'll use

        Everything below runs at a **tiny** scale so it fits on screen and runs
        instantly on CPU. We keep the *names* identical to a real model — only the
        numbers shrink:

        | symbol | meaning | toy value | real reference |
        |---|---|---|---|
        | `vocab` | size of the vocabulary | 20 | ~50,000 |
        | `H` | model / hidden width | 16 | 4096 |
        | `I` | MLP intermediate width (`= 4·H`) | 64 | 16384 |
        | `heads` | attention heads | 4 | 32 |
        | `L` | number of stacked layers | 4 | 32 |
        | `B` | batch (sequences at once) | 1 | varies |
        | `S` | sequence length (tokens) | 6 | up to 65,536 |

        We'll do the real-scale **arithmetic** at the end to make the memory bill real.
        """
    )
    return


@app.cell
def _():
    # The toy configuration. Defined once; every later cell reads these.
    vocab = 20
    H = 16
    I = 4 * H        # 64 — the MLP "expansion" width
    heads = 4
    L = 4
    B = 1
    S = 6
    return B, H, I, L, S, heads, vocab


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 0 — Tokens: text becomes integers

        A neural network can't read letters; it reads integers. So the first step is
        **tokenization**: chop the text into pieces and map each piece to an id via a
        lookup dictionary. Real tokenizers use word-pieces and ~50k entries; ours is a
        toy word-level vocabulary of 20 tokens.

        The raw input to the model is therefore a grid of integers of shape
        `[B, S]` — `B` sequences, each `S` tokens long. **Nothing is learned yet** —
        this is pure dictionary indexing.
        """
    )
    return


@app.cell
def _(torch):
    # A tiny word-level vocabulary: token string -> integer id.
    toy_vocab = {
        "<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5,
        "dog": 6, "ran": 7, "fast": 8, "a": 9, "big": 10, "red": 11,
        "ball": 12, "and": 13, "small": 14, "blue": 15, "box": 16,
        "is": 17, "here": 18, "now": 19,
    }
    id_to_word = {i: w for w, i in toy_vocab.items()}

    phrase = "the cat sat on the mat"
    token_ids = torch.tensor([[toy_vocab[w] for w in phrase.split()]])  # [B, S]
    return id_to_word, phrase, token_ids, toy_vocab


@app.cell
def _(mo, phrase, token_ids):
    mo.md(
        f"""
        Tokenizing **"{phrase}"**:

        ```
        text        : {phrase!r}
        token ids   : {token_ids.tolist()}
        shape [B,S] : {tuple(token_ids.shape)}
        ```

        Each integer is just an address into the embedding table we build next.
        `the` appears twice → the same id (`1`) appears twice. No math has happened.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 1 — Embedding: integers become vectors

        The model can't do arithmetic on the id `2`. So we look every id up in a big
        table `E` of shape `[vocab, H]` — **one row of `H` numbers per possible
        token**. That table is the **embedding matrix**, and it is a **weight**: it's
        learned once and is identical for every request. Its size depends on `vocab`
        and `H`, *never* on your input.

        Looking up the ids produces our **first activation** `X` of shape `[B, S, H]`:
        it exists only because we fed in *this* phrase, and it grows when the phrase
        gets longer.
        """
    )
    return


@app.cell
def _(H, mo, token_ids, torch, vocab):
    # E is a WEIGHT: [vocab, H], learned once, input-independent.
    E = torch.randn(vocab, H) * 0.5

    # X is an ACTIVATION: look up each id -> [B, S, H]. Depends on the input.
    X_embed = E[token_ids]

    mo.md(
        f"""
        ```
        E  (embedding table)   shape {tuple(E.shape)}      <- WEIGHT      (vocab x H)
        token_ids              shape {tuple(token_ids.shape)}                <- the input
        X = E[token_ids]       shape {tuple(X_embed.shape)}             <- ACTIVATION  (B x S x H)
        ```

        Same operation as `df.loc[ids]` on a lookup table — pure gather, no matmul yet.
        """
    )
    return E, X_embed


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Your first weight and your first activation, side by side.**
            `E` is `[vocab, H]` — feed in 6 tokens or 6 million, it is byte-for-byte
            the same. `X` is `[B, S, H]` — it only exists for this batch and grows with
            the number of tokens. **If a tensor's shape contains `B` or `S`, it's an
            activation.** That one rule will carry you through the entire course.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(E, COL_WEIGHT, id_to_word, plt, vocab):
    # Heatmap of the embedding table — a WEIGHT. One row per token.
    fig_e, ax_e = plt.subplots(figsize=(7, 5))
    im_e = ax_e.imshow(E.numpy(), aspect="auto", cmap="PuOr")
    ax_e.set_title("Embedding table E  [vocab=20, H=16]  — a WEIGHT", color=COL_WEIGHT)
    ax_e.set_xlabel("hidden dimension (H)")
    ax_e.set_ylabel("token id")
    ax_e.set_yticks(range(vocab))
    ax_e.set_yticklabels([f"{i}:{id_to_word[i]}" for i in range(vocab)], fontsize=7)
    fig_e.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04, label="value")
    ax_e
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Every **row** above is one token's learned coordinates in a 16-dimensional
        "meaning space". Tokenizing the phrase just selects rows `[1, 2, 3, 4, 1, 5]`
        and stacks them into `X`. The table is fixed; *which rows we pull* is what
        depends on the input.

        ## Step 2 — The residual stream: the data bus through the model

        From here on, **that `[B, S, H]` tensor is the only thing flowing through the
        network.** Every layer reads it, computes an *update*, and **adds the update
        back** — it never replaces the stream:

        ```
        X = X + attention(X)     # add what attention figured out
        X = X + mlp(X)           # add what the MLP figured out
        ```

        This is the **residual stream**: a shared running buffer. Because each
        sub-layer maps `[B, S, H]` to a same-shaped update that is *added on*, the
        shape `[B, S, H]` is **invariant** all the way down — only the values change.
        That's also exactly why you can stack identical blocks.
        """
    )
    return


@app.cell
def _(B, H, S, X_embed, mo, torch):
    # A residual update has the SAME shape as the stream, so adding it is shape-safe.
    fake_update = torch.randn(B, S, H) * 0.1
    X_after = X_embed + fake_update

    mo.md(
        f"""
        ```
        X (stream in)        shape {tuple(X_embed.shape)}    ACTIVATION
        update = sublayer(X)  shape {tuple(fake_update.shape)}    ACTIVATION  (same shape!)
        X = X + update       shape {tuple(X_after.shape)}    ACTIVATION  (shape unchanged)
        ```

        Shape preserved → the next layer can consume it identically. Stacking is just
        running this same step `L` times.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 3 — The MLP, computed by hand

        Half of every transformer block is the **MLP** (feed-forward network): two
        matrix multiplies with a nonlinearity (GELU) between them. It processes **each
        token independently** — no token looks at any other here (that's attention's
        job, in the next notebook).

        Its two **weights**:

        ```
        W_in : [I, H]    expands width  H -> I   (here 64 x 16)
        W_out: [H, I]    contracts it   I -> H   (here 16 x 64)
        ```

        Convention used throughout the course and the real repo: `F.linear(X, W)`
        computes `X @ W.T`, and weights are stored **output-first** as `[out, in]`.

        Watch the **new activation** that appears in the middle — `hidden` — and notice
        it carries the `S` axis, so it scales with sequence length.
        """
    )
    return


@app.cell
def _(F, H, I, X_embed, torch):
    # MLP weights: sizes depend ONLY on H and I, never on the input.
    W_in = torch.randn(I, H) * (H ** -0.5)    # [I, H]  WEIGHT
    W_out = torch.randn(H, I) * (I ** -0.5)   # [H, I]  WEIGHT

    # Forward pass on the residual stream. F.linear(X, W) == X @ W.T
    hidden = F.gelu(F.linear(X_embed, W_in))  # [B, S, I]  ACTIVATION
    mlp_out = F.linear(hidden, W_out)         # [B, S, H]  ACTIVATION
    return W_in, W_out, hidden, mlp_out


@app.cell
def _(W_in, W_out, X_embed, hidden, mlp_out, mo):
    mo.md(
        f"""
        ```
        X        {tuple(X_embed.shape)!s:<12} ACTIVATION   (the residual stream coming in)

        W_in     {tuple(W_in.shape)!s:<12} WEIGHT       [I, H]
        hidden = gelu(X @ W_in.T)
                 {tuple(hidden.shape)!s:<12} ACTIVATION   [B, S, I]  <- the wide middle tensor

        W_out    {tuple(W_out.shape)!s:<12} WEIGHT       [H, I]
        out    = hidden @ W_out.T
                 {tuple(mlp_out.shape)!s:<12} ACTIVATION   [B, S, H]  <- back to stream width
        ```

        The stream went in `[B,S,H]`, ballooned to `[B,S,I]` in the middle (4x wider),
        and came back to `[B,S,H]` so it can be added onto the residual stream.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Read the two kinds of numbers in this block.** `W_in` and `W_out` are
            **weights** — their sizes (`64×16` and `16×64`) depend only on `H` and `I`.
            Feed in 6 tokens or 6 million; identical. `hidden` and `out` are
            **activations** — shape `[B, S, …]`, so they scale directly with how many
            tokens you push through. Double the sequence → double these tensors. *This
            is why long context blows up activation memory but not weight memory.*
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 4 — Stack it: the whole transformer in one function

        A transformer is structurally simple: **embed → repeat {attention, MLP} L
        times → un-embed**. Below we wire that up. We use a **trivial attention
        stand-in** (a plain linear mix) so the shapes flow correctly — *real attention
        is the entire next notebook*. The point here is the skeleton and the shapes.

        The final step, **un-embedding**, reuses the embedding table transposed:
        `logits = X @ E.T` projects each token's `H`-vector back out to a score over
        the whole vocabulary, giving `[B, S, vocab]` — the next-token scores.
        """
    )
    return


@app.cell
def _(F, H, I, L, torch, vocab):
    # Build per-layer weights once (a real model has L private copies of each).
    def make_layer_weights(H_, I_, L_):
        layers = []
        for _ in range(L_):
            layers.append({
                # trivial attention stand-in: a single [H,H] mix (NOT real attention)
                "W_attn": torch.randn(H_, H_) * (H_ ** -0.5),
                "W_in": torch.randn(I_, H_) * (H_ ** -0.5),
                "W_out": torch.randn(H_, I_) * (I_ ** -0.5),
            })
        return layers

    def attention_placeholder(X, W_attn):
        # STAND-IN ONLY. Real attention (Q/K/V, softmax) is the next notebook.
        return F.linear(X, W_attn)

    def mlp(X, W_in, W_out):
        return F.linear(F.gelu(F.linear(X, W_in)), W_out)

    def transformer_forward(token_ids_, E_, layers):
        X = E_[token_ids_]                                  # embed -> [B,S,H] ACTIVATION
        for lw in layers:
            X = X + attention_placeholder(X, lw["W_attn"])  # residual add
            X = X + mlp(X, lw["W_in"], lw["W_out"])         # residual add
        logits = X @ E_.T                                   # un-embed -> [B,S,vocab]
        return logits

    layer_weights = make_layer_weights(H, I, L)
    return layer_weights, transformer_forward


@app.cell
def _(E, X_embed, id_to_word, layer_weights, mo, token_ids, transformer_forward):
    logits = transformer_forward(token_ids, E, layer_weights)

    # Greedy "next token" prediction for the final position (toy weights => toy result).
    last_scores = logits[0, -1]
    pred_id = int(last_scores.argmax())

    mo.md(
        f"""
        ```
        token_ids   {tuple(token_ids.shape)!s:<12} input
        X = embed   {tuple(X_embed.shape)!s:<12} ACTIVATION
        ... {len(layer_weights)} layers of (+attention_placeholder, +mlp), shape unchanged ...
        logits      {tuple(logits.shape)!s:<12} ACTIVATION  [B, S, vocab]
        ```

        The model produced a score for every vocab token at every position. Taking the
        last position's argmax (with random toy weights, so the answer is meaningless):
        predicted next token id = **{pred_id}** (`{id_to_word[pred_id]!r}`).

        The mechanics are real even though the trained values aren't — this is exactly
        the shape pipeline a 6-billion-parameter model runs.
        """
    )
    return (logits,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Chart 1 — Where do the parameters live? (attention vs. MLP)

        Each layer's weights split into two buckets: the **attention** projections
        (`W_q, W_k, W_v, W_o` — four `[H,H]` matrices in a real model) and the **MLP**
        (`W_in, W_out` — two `[I,H]` matrices). With `I = 4H`, the MLP is the heavier
        of the two. Let's count parameters per component, per layer, and ×L.
        """
    )
    return


@app.cell
def _(COL_ATTN, COL_MLP, H, I, L, plt):
    attn_params_layer = 4 * H * H        # W_q, W_k, W_v, W_o
    mlp_params_layer = 2 * I * H         # W_in, W_out

    cats = ["Attention\n(4·H·H)", "MLP\n(2·I·H)"]
    per_layer = [attn_params_layer, mlp_params_layer]
    times_L = [attn_params_layer * L, mlp_params_layer * L]

    fig_p, (axp1, axp2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax_, vals, title in ((axp1, per_layer, "Per layer"),
                             (axp2, times_L, f"Total (× L={L})")):
        bars = ax_.bar(cats, vals, color=[COL_ATTN, COL_MLP])
        ax_.set_title(title)
        ax_.set_ylabel("parameters")
        for bar_, v in zip(bars, vals):
            ax_.text(bar_.get_x() + bar_.get_width() / 2, v, f"{v:,}",
                     ha="center", va="bottom", fontsize=9)
        ax_.margins(y=0.15)
    fig_p.suptitle("Parameter counts: attention vs. MLP (toy scale)", fontweight="bold")
    fig_p.tight_layout()
    axp2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The MLP carries ~2× the attention parameters (because `I = 4H` makes `2·I·H =
        8H²` vs attention's `4H²`). Notice neither bar mentions `B` or `S` — **the
        parameter count is completely independent of input length.**

        ## Chart 2 — The headline: weights are flat, activations grow with S

        This is *the* chart of the whole course. Slide the sequence length `S` below
        and watch: the **weight** memory (recipes) stays perfectly flat, while the
        **activation** memory (ingredients on the stove) climbs linearly. Drag it up
        and activations overtake the weights entirely — that's a long-context OOM in
        one picture.
        """
    )
    return


@app.cell
def _(mo):
    # UI element defined in its OWN cell, then read downstream so the chart reacts.
    s_slider = mo.ui.slider(
        start=1, stop=4096, step=64, value=512,
        label="sequence length S", show_value=True,
    )
    s_slider
    return (s_slider,)


@app.cell
def _(COL_ACT, COL_WEIGHT, H, I, L, np, plt, s_slider, vocab):
    # Reference (real) scale for a believable memory curve.
    H_ref, I_ref, L_ref, vocab_ref, B_ref = 4096, 16384, 32, 50000, 1

    # Weights are constant in S: embeddings + per-layer (attn 4H^2 + mlp 2IH).
    weight_params = vocab_ref * H_ref + L_ref * (4 * H_ref * H_ref + 2 * I_ref * H_ref)

    # Activation elements per token, summed over layers (stream + hidden, roughly).
    act_per_token = L_ref * (H_ref + I_ref)

    S_axis = np.arange(1, 8193, 64)
    weight_curve = np.full_like(S_axis, weight_params, dtype=float)
    act_curve = act_per_token * S_axis * B_ref

    S_now = s_slider.value
    act_now = act_per_token * S_now * B_ref

    fig_h, ax_h = plt.subplots(figsize=(9, 5))
    ax_h.plot(S_axis, weight_curve / 1e9, color=COL_WEIGHT, lw=3,
              label="weights (recipes) — constant")
    ax_h.plot(S_axis, act_curve / 1e9, color=COL_ACT, lw=3,
              label="activations (on the stove) — grow with S")
    ax_h.axvline(S_now, color="#888", ls="--", lw=1)
    ax_h.scatter([S_now], [act_now / 1e9], color=COL_ACT, zorder=5, s=60)
    ax_h.annotate(
        f"S={S_now}\nactivations ≈ {act_now/1e9:.2f} B elems",
        xy=(S_now, act_now / 1e9), xytext=(10, 10),
        textcoords="offset points", fontsize=9, color=COL_ACT,
    )
    ax_h.set_title("Weights stay flat; activations scale with sequence length",
                   fontweight="bold")
    ax_h.set_xlabel("sequence length S (tokens)")
    ax_h.set_ylabel("billions of elements")
    ax_h.legend(loc="upper left")
    # Silence unused toy-scale imports being flagged; they document scale only.
    _ = (H, I, L, vocab)
    ax_h
    return (act_now, weight_params)


@app.cell
def _(act_now, mo, s_slider, weight_params):
    crossover = weight_params  # activations exceed weights once act_now > this
    verdict = (
        "activations now EXCEED the weights — welcome to a long-context OOM."
        if act_now > crossover
        else "weights still dominate — plenty of headroom."
    )
    mo.callout(
        mo.md(
            f"""
            At **S = {s_slider.value}**, activations ≈ **{act_now/1e9:.2f} B** elements
            vs. fixed weights ≈ **{weight_params/1e9:.2f} B** elements. {verdict}

            The weight line never moves no matter where you drag the slider. That
            asymmetry — fixed model state vs. workload-driven scratch memory — is the
            reason the rest of the course exists.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Chart 3 — The residual stream as a heatmap

        Here is the actual residual stream for our 6-token phrase: one **row per
        token**, `H=16` columns. This is an **activation** — it exists only because we
        embedded *this* phrase. The two `the` rows (positions 0 and 4) start identical
        because embedding is a pure lookup; in a real model, attention would make them
        diverge as each `the` gathers different context.
        """
    )
    return


@app.cell
def _(COL_ACT, X_embed, id_to_word, phrase, plt, token_ids):
    fig_x, ax_x = plt.subplots(figsize=(8, 3.5))
    im_x = ax_x.imshow(X_embed[0].numpy(), aspect="auto", cmap="coolwarm")
    ax_x.set_title("Residual stream X[0]  [S=6, H=16]  — an ACTIVATION", color=COL_ACT)
    ax_x.set_xlabel("hidden dimension (H)")
    ax_x.set_ylabel("token (position)")
    words = [id_to_word[i] for i in token_ids[0].tolist()]
    ax_x.set_yticks(range(len(words)))
    ax_x.set_yticklabels([f"{p}:{w}" for p, w in enumerate(words)])
    fig_x.colorbar(im_x, ax=ax_x, fraction=0.046, pad=0.04, label="value")
    _ = phrase
    fig_x.tight_layout()
    ax_x
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 5 — Counting it for real (H=4096, I=16384, L=32)

        The toy scale shows the *mechanics*; real numbers show the *stakes*. The
        function below computes the parameter count from `(H, I, L, vocab)` alone —
        notice there is **no `B` or `S` argument**, because parameters don't depend on
        the input.
        """
    )
    return


@app.cell
def _():
    def count_params(H, I, L, vocab):
        embed = vocab * H                       # embedding table (also reused for un-embed)
        attn_per_layer = 4 * H * H              # W_q, W_k, W_v, W_o
        mlp_per_layer = 2 * I * H               # W_in, W_out
        per_layer = attn_per_layer + mlp_per_layer
        total = embed + L * per_layer
        return {
            "embedding": embed,
            "attn_per_layer": attn_per_layer,
            "mlp_per_layer": mlp_per_layer,
            "per_layer": per_layer,
            "total": total,
        }

    ref = count_params(H=4096, I=16384, L=32, vocab=50000)
    return count_params, ref


@app.cell
def _(mo, ref):
    def fmt(n):  # human-friendly
        return f"{n/1e9:.2f} B" if n >= 1e9 else f"{n/1e6:.1f} M"

    mo.md(
        f"""
        ### Weights (model state) — fixed, independent of input

        | weight | shape | parameters |
        |---|---|---|
        | embedding `E` | `[50000, 4096]` | {fmt(ref['embedding'])} |
        | attention `W_q,W_k,W_v,W_o` (per layer) | `4 × [4096, 4096]` | {fmt(ref['attn_per_layer'])} |
        | MLP `W_in, W_out` (per layer) | `2 × [16384, 4096]` | {fmt(ref['mlp_per_layer'])} |
        | **per layer total** | — | **{fmt(ref['per_layer'])}** |
        | **× 32 layers + embedding** | — | **{fmt(ref['total'])}** |

        That's the ~6.4-billion-parameter reference model. **`B` and `S` appear
        nowhere** — it's 6.4 B parameters whether you prompt it with 5 tokens or
        50,000. *That's what Tensor Parallelism slices across GPUs.*

        ### Activations — scale with batch and sequence

        Just the MLP's `hidden` tensor, one layer, `B=1`:

        ```
        hidden: [B, S, I] = [1, S, 16384]
          S =  8,192  ->  ~134 M elements   (~0.27 GB in bf16)
          S = 65,536  ->  ~1.07 B elements  (~2.1 GB in bf16)   # ONE tensor, ONE layer!
        ```

        Add Q, K, V, attention scores, `hidden`, and the residual stream across all 32
        layers and the activation footprint becomes enormous at long context. *That's
        what Sequence Parallelism splits across GPUs.*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## One token's journey (the whole machine, in shapes)

        Follow the word **"cat"** (id `2`) from raw text to a prediction:

        | stage | operation | shape | kind |
        |---|---|---|---|
        | raw token | `"cat" → 2` | scalar | input |
        | embed | `E[2]` | `[H] = [16]` | **activation** (via weight `E`) |
        | + attention ×L | `X = X + attn(X)` | `[B, S, H]` | activation (via attn weights) |
        | + MLP ×L | `X = X + mlp(X)` | `[B, S, H]` | activation (via MLP weights) |
        | un-embed | `X @ E.T` | `[B, S, vocab]` | activation (via weight `E`) |

        Every arrow is a matrix multiply between an **activation** (the stream) and a
        **weight** (a learned matrix). The shape `[B, S, H]` is invariant through all
        `L` layers; only the values get refined. That is the entire machine.
        """
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
            "Q1 — Classify each as a weight or an activation: (a) E, (b) X, (c) W_in, (d) hidden.":
            mo.md(
                r"""
                **(a) E — weight.** Learned lookup table; size depends on `vocab` and
                `H`, not on the input. **(b) X — activation.** Exists only for a
                specific batch; shape `[B,S,H]` scales with input. **(c) W_in —
                weight.** Learned `[I,H]` matrix, constant per request. **(d) hidden —
                activation.** Produced during the forward pass; shape `[B,S,I]` scales
                with input. **Rule of thumb: if the shape contains `B` or `S`, it's an
                activation.**
                """
            ),
            "Q2 — You double the sequence length from 8K to 16K. What grows — parameters, activations, both, or neither?":
            mo.md(
                r"""
                Only **activation memory** grows (roughly doubles). The parameter count
                is a property of the matrix shapes (`H, I, L, vocab`) and is completely
                independent of how many tokens you process. Activations carry the `S`
                axis, so they scale with sequence length. This asymmetry is exactly why
                a model whose *weights* fit fine can still OOM on long inputs.
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

            - Text → **token ids** `[B,S]` (pure indexing, nothing learned).
            - **Embedding** table `E` (a **weight**) → residual stream `X` `[B,S,H]`
              (an **activation**).
            - The **residual stream** keeps shape `[B,S,H]` because every sub-layer
              *adds* a same-shaped update: `X = X + sublayer(X)`.
            - The **MLP** is two matmuls around GELU; its `hidden` `[B,S,I]` is the wide
              activation that scales with `S`.
            - A transformer = **embed → (attention + MLP) × L → un-embed** into logits
              `[B,S,vocab]`.
            - **Weights** are driven by model size (`H, I, L, vocab`) and are constant
              per request; **activations** are driven by workload (`B, S`) and vanish
              when the request ends. Two memory bills, two different drivers.

            Next: open up the most interesting sub-layer — **attention** — and replace
            our stand-in with the real thing.

            Companion chapter: `../course/02-transformers.html`.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
