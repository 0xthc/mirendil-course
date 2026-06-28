import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Template notebook

        This is a **known-good** marimo pattern. Copy it. Notes:

        - Every cell is an `@app.cell` function. A cell *returns* the variables it
          wants downstream cells to use. Downstream cells receive them as arguments.
        - `mo.md(...)` renders markdown (and LaTeX with `$...$`).
        - To show a matplotlib figure, make the **last expression of the cell** the
          `Axes` or `Figure` object (here: `ax`). Don't call `plt.show()`.
        - Keep tensors tiny (CPU). These notebooks teach by *simulating* multi-GPU
          ranks in a single process.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    return mo, np, torch, plt


@app.cell
def _(np, plt):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(10), np.arange(10) ** 2, marker="o", color="#57b6f5")
    ax.set_title("a chart renders by returning the Axes")
    ax.set_xlabel("x")
    ax.set_ylabel("x²")
    ax
    return


@app.cell
def _(mo, torch):
    x = torch.randn(2, 3)
    mo.md(f"A tensor of shape `{tuple(x.shape)}`:\n\n```\n{x}\n```")
    return


if __name__ == "__main__":
    app.run()
