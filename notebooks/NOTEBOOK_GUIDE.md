# Marimo notebook authoring guide (read this first)

You are building ONE exhaustive, teaching-grade marimo notebook for a full-stack
engineer (NOT a data scientist). It must be runnable on CPU, beautiful, and
heavily explained. These notebooks are the runnable companion to an HTML course
that already exists in `../course/` — match its tone (slow, concrete, ELI5 +
precise) and reuse its analogies where natural.

## Environment (already set up)
- Python 3.13, run everything via `uv run` from the repo root `/Users/Shared/mirendil/mirendil-test`.
- Available: `marimo` 0.23.11, `torch` 2.12 (CPU), `numpy`, `matplotlib`.
- Keep all tensors TINY (e.g. B=1–2, S=8–16, H=16–64, heads=2–4) so cells run instantly on CPU.

## marimo file format (CRITICAL — copy `_template.py` exactly)
```python
import marimo
__generated_with = "0.23.11"
app = marimo.App(width="medium")

@app.cell
def _(mo):                       # args = variables this cell consumes
    mo.md(r"""# Heading""")
    return                       # return the vars other cells need

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    return mo, np, torch, plt

if __name__ == "__main__":
    app.run()
```
Rules:
- Each cell is `@app.cell` over a function named `_`. It **returns a tuple** of the
  variables downstream cells use; downstream cells take them as **arguments**.
- marimo is REACTIVE and forbids re-defining the same variable in two cells. Give
  variables unique names across the whole notebook (e.g. `X_tp`, `X_sp`), or wrap
  throwaway work in a function. Do NOT reassign an imported name.
- Show a matplotlib chart by making the **last expression** of the cell the `Axes`
  or `Figure` object (e.g. `ax`). Never call `plt.show()`.
- Render rich text/values with `mo.md(f"...")`. Show shapes/tensors inside fenced
  code blocks in markdown for clean output cells.
- Interactive UI: define the element in ONE cell and return it
  (`slider = mo.ui.slider(1, 16, value=8, label="S"); slider`), then READ
  `slider.value` in a SEPARATE downstream cell so it reacts. Use
  `mo.ui.slider`, `mo.ui.dropdown`, `mo.ui.switch` where they genuinely help
  (e.g. a sequence-length slider that redraws a memory chart, or picking a query
  token to highlight in an attention heatmap).
- Use `mo.callout(mo.md("..."), kind="info"|"success"|"warn"|"danger")` for ELI5 /
  key-insight / gotcha boxes. Use `mo.accordion({"Show answer": mo.md("...")})`
  for check-your-understanding toggles.

## Visual style
Use this palette consistently in matplotlib (hex):
- TP `#57b6f5`, SP `#f0c674`, TSP `#5fd38a`, weights `#c099f0`, activations `#f0986b`, neutral/accent `#57b6f5`.
Charts must have titles, axis labels, legends, and value annotations where useful.
Prefer clear bar charts, heatmaps (`imshow`), and simple box/grid diagrams.

## Simulating multi-GPU in ONE process (use this for TP/SP/TSP)
We can't run 8 GPUs in a notebook, so we SIMULATE ranks as a Python list of
tensors `[rank0_tensor, rank1_tensor, ...]` and implement the three collectives
as pure functions. Put these helpers in a cell and explain them:
```python
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
```
Always VERIFY correctness: compute a plain single-process reference, then the
"parallel" (looped-over-ranks) version, and assert they match with
`torch.testing.assert_close(...)`, then print a nice ✅ success message via `mo.md`.

## Required structure for every notebook
1. Title + one-paragraph "what you'll build" + ELI5 callout.
2. Imports cell.
3. Concept built up in small, narrated steps: a markdown cell EXPLAINING the next
   code, then the code cell, then an output/chart cell interpreting the result.
4. At least 2–3 charts/diagrams (heatmaps, bar charts, shape diagrams).
5. At least one interactive `mo.ui` element.
6. Correctness verification with assertions + success callout (for TP/SP/TSP).
7. A "what you learned" recap and a pointer to the matching course chapter file.

## Accuracy anchors (read these for correct shapes & conventions)
- Real repo implementations: `../tensor_parallelism.py`, `../sequence_parallelism.py`, `../ts_parallelism.py`.
- Course chapters (tone + content): `../course/02-transformers.html`, `03-attention-refresher.html`,
  `05-tensor-parallelism.html`, `06-sequence-parallelism.html`, `07-tensor-sequence-parallelism.html`, `08-tradeoffs.html`.
- Convention: `F.linear(X, W)` computes `X @ W.T`; weights are stored `[out_features, in_features]`.
- TP attention = column-parallel Q/K/V (split out features / heads) + row-parallel W_o (split in features) + `all_reduce`.
- SP = shard the sequence (token) axis; `all_gather` K/V; MLP needs no comms; mind the causal offset per rank; zigzag fixes load imbalance.
- TSP = fold both onto one axis (the diagonal); loop over weight shards, `broadcast` each, `all_gather` K/V, accumulate locally, NO final all_reduce.

## Validation (MUST pass before you finish)
Run from repo root and ensure exit code 0 with no tracebacks:
```
uv run marimo export html notebooks/<your_file>.py -o /tmp/<your_file>.html
```
If it errors, fix the notebook and re-run until clean. Also keep cells fast
(< a few seconds total).
