# Mirendil TSP — interactive marimo notebooks

Runnable companions to the [HTML course](../course/). Where the course explains,
these notebooks let you **execute, poke, and verify** every idea — on your laptop
CPU, no GPUs required.

The trick: multi-GPU parallelism is **simulated in a single process**. Each "rank"
is just an entry in a list of tensors, and the three collectives (`broadcast`,
`all_reduce`, `all_gather`) are small pure functions. Every parallel result is
checked against a plain single-process reference with `torch.testing.assert_close`,
so you can trust the algorithm is correct, not just plausible.

## The notebooks (read in order)

| # | File | What you build |
|---|------|----------------|
| 01 | `01_transformers.py` | Tokens → embeddings → residual stream → MLP. Weights vs. activations made concrete with live charts. |
| 02 | `02_attention.py` | Scaled dot-product attention from scratch: scores, causal mask, softmax, multi-head — with heatmaps. Verified against PyTorch SDPA. |
| 03 | `03_tensor_parallelism.py` | Head sharding, column- vs row-parallel, `all_reduce`. Simulated ranks, verified correct. |
| 04 | `04_sequence_parallelism.py` | Token sharding, K/V `all_gather`, the causal-offset mask, load imbalance + zigzag fix. |
| 05 | `05_tensor_sequence_parallelism.py` | The folded TSP loop algorithm, TSP vs TP+SP, real benchmark charts. |

## Running them

From the repo root (`/Users/Shared/mirendil/mirendil-test`):

```bash
# Interactive, reactive editor (recommended — sliders/toggles work live):
uv run marimo edit notebooks/01_transformers.py

# Read-only app view:
uv run marimo run notebooks/02_attention.py

# Export a static HTML snapshot (also how each notebook is validated):
uv run marimo export html notebooks/03_tensor_parallelism.py -o /tmp/out.html
```

Dependencies (`marimo`, `numpy`, `matplotlib`, `torch`) are already installed in
the project venv. If you ever need them again:

```bash
uv pip install marimo numpy matplotlib
```

## How they're built (for maintainers)

- Each cell is an `@app.cell` function; cells pass data via return values →
  arguments (marimo's reactive dataflow). Variable names are unique across the
  whole file (marimo forbids redefining a global in two cells).
- Charts render by making the `Axes`/`Figure` the cell's last expression.
- Interactive controls (`mo.ui.slider`, `mo.ui.dropdown`, `mo.ui.switch`) are
  defined in one cell and read via `.value` in a downstream cell.
- See `NOTEBOOK_GUIDE.md` for the full authoring conventions and the collective
  simulation helpers.

Every notebook is validated by exporting to HTML and confirming all cells execute
without error (including the correctness assertions).
