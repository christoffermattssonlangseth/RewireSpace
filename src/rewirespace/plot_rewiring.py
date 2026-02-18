from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _stage_order(stage: pd.Series) -> list[Any]:
    if isinstance(stage.dtype, pd.CategoricalDtype):
        return list(stage.cat.categories)

    values = [v for v in pd.unique(stage) if pd.notna(v)]
    try:
        return sorted(values)
    except TypeError:
        return list(values)


def _type_order(df_long: pd.DataFrame) -> list[str]:
    if "type_i" not in df_long.columns or "type_j" not in df_long.columns:
        raise KeyError("df_long must contain `type_i` and `type_j` columns.")

    vals = pd.unique(pd.concat([df_long["type_i"], df_long["type_j"]], ignore_index=True))
    vals = [v for v in vals if pd.notna(v)]
    try:
        return sorted(vals)
    except TypeError:
        return list(vals)


def _stage_matrix(
    df_long: pd.DataFrame,
    stage: Any,
    value_col: str,
    stage_col: str,
    type_order: Sequence[str],
) -> pd.DataFrame:
    sub = df_long.loc[df_long[stage_col] == stage, ["type_i", "type_j", value_col]].copy()
    mat = sub.pivot_table(
        index="type_i",
        columns="type_j",
        values=value_col,
        aggfunc="mean",
        observed=False,
    )
    mat = mat.reindex(index=type_order, columns=type_order)
    return mat


def plot_stage_heatmap(
    df_long: pd.DataFrame,
    stage: Any,
    value_col: str = "Z",
    stage_col: str = "stage",
    type_order: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "coolwarm",
    vlim: Optional[float] = None,
    colorbar: bool = True,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot type-type mean value heatmap for one stage.

    Parameters
    ----------
    df_long
        Long table from `per_sample_contacts`.
    stage
        Stage value to visualize.
    value_col
        Column to aggregate (typically "Z").
    stage_col
        Stage column name.
    type_order
        Optional ordering for row/column types.
    """
    required = {"type_i", "type_j", value_col, stage_col}
    missing = required.difference(df_long.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if type_order is None:
        type_order = _type_order(df_long)

    mat = _stage_matrix(df_long, stage, value_col, stage_col, type_order)
    arr = mat.to_numpy(dtype=float)

    if vlim is None:
        vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    else:
        vmax = float(vlim)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(type_order)))
    ax.set_yticks(np.arange(len(type_order)))
    ax.set_xticklabels(type_order, rotation=90)
    ax.set_yticklabels(type_order)
    ax.set_xlabel("type_j")
    ax.set_ylabel("type_i")
    ax.set_title(title if title is not None else f"{value_col} heatmap | stage={stage}")

    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col)

    fig.tight_layout()
    return fig, ax


def plot_stage_heatmaps(
    df_long: pd.DataFrame,
    value_col: str = "Z",
    stage_col: str = "stage",
    type_order: Optional[Sequence[str]] = None,
    stages: Optional[Sequence[Any]] = None,
    ncols: int = 3,
    cmap: str = "coolwarm",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a panel of per-stage type-type heatmaps of mean values.
    """
    required = {"type_i", "type_j", value_col, stage_col}
    missing = required.difference(df_long.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if type_order is None:
        type_order = _type_order(df_long)

    if stages is None:
        stages = _stage_order(df_long[stage_col])

    stages = list(stages)
    if not stages:
        raise ValueError("No stages available to plot.")

    mats = [_stage_matrix(df_long, s, value_col, stage_col, type_order) for s in stages]
    vmax = 1.0
    finite_vals = [m.to_numpy(dtype=float) for m in mats if np.isfinite(m.to_numpy(dtype=float)).any()]
    if finite_vals:
        vmax = max(np.nanmax(np.abs(arr)) for arr in finite_vals)

    ncols = max(int(ncols), 1)
    nrows = int(np.ceil(len(stages) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)

    for idx, stage in enumerate(stages):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        arr = mats[idx].to_numpy(dtype=float)
        im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(type_order)))
        ax.set_yticks(np.arange(len(type_order)))
        ax.set_xticklabels(type_order, rotation=90)
        ax.set_yticklabels(type_order)
        ax.set_title(f"{stage_col}={stage}")

    for idx in range(len(stages), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label=value_col)
    fig.tight_layout()
    return fig, axes


def plot_rewiring_curves(
    df_long: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    value_col: str = "Z",
    stage_col: str = "stage",
    ax: Optional[plt.Axes] = None,
    show_sem: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot stage-wise rewiring curves (mean +/- SEM) for selected type pairs.

    Parameters
    ----------
    pairs
        Sequence of (type_i, type_j) tuples.
    """
    required = {"type_i", "type_j", value_col, stage_col}
    missing = required.difference(df_long.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if not pairs:
        raise ValueError("`pairs` must contain at least one (type_i, type_j) tuple.")

    stage_order = _stage_order(df_long[stage_col])
    x = np.arange(len(stage_order), dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for type_i, type_j in pairs:
        sub = df_long.loc[
            (df_long["type_i"] == type_i) & (df_long["type_j"] == type_j),
            [stage_col, value_col],
        ].copy()

        if sub.empty:
            continue

        stats = sub.groupby(stage_col, observed=False)[value_col].agg(["mean", "count", "std"])
        stats = stats.reindex(stage_order)
        mean = stats["mean"].to_numpy(dtype=float)
        sem = (stats["std"] / np.sqrt(stats["count"].clip(lower=1))).to_numpy(dtype=float)

        label = f"{type_i} -> {type_j}"
        ax.plot(x, mean, marker="o", linewidth=2, label=label)

        if show_sem:
            low = mean - sem
            high = mean + sem
            ax.fill_between(x, low, high, alpha=0.18)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in stage_order], rotation=45, ha="right")
    ax.set_xlabel(stage_col)
    ax.set_ylabel(value_col)
    ax.set_title("Rewiring curves across stage")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=1)

    fig.tight_layout()
    return fig, ax


__all__ = [
    "plot_stage_heatmap",
    "plot_stage_heatmaps",
    "plot_rewiring_curves",
]
