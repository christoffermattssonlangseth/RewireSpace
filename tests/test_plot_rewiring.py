import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rewirespace.plot_rewiring import (
    plot_rewiring_curves,
    plot_stage_heatmap,
    plot_stage_heatmaps,
)


def _df_long() -> pd.DataFrame:
    rows = []
    for stage, shift in [("early", -0.5), ("mid", 0.0), ("late", 0.6)]:
        for sample in ["s1", "s2", "s3"]:
            rows.extend(
                [
                    {"sample_id": sample, "stage": stage, "type_i": "T", "type_j": "T", "Z": 0.3 + shift},
                    {"sample_id": sample, "stage": stage, "type_i": "T", "type_j": "B", "Z": -0.1 + shift},
                    {"sample_id": sample, "stage": stage, "type_i": "B", "type_j": "T", "Z": 0.2 + shift},
                    {"sample_id": sample, "stage": stage, "type_i": "B", "type_j": "B", "Z": -0.3 + shift},
                ]
            )

    df = pd.DataFrame(rows)
    df["stage"] = pd.Categorical(df["stage"], categories=["early", "mid", "late"], ordered=True)
    return df


def test_plot_stage_heatmap_runs():
    df = _df_long()
    fig, ax = plot_stage_heatmap(df, stage="early", value_col="Z", stage_col="stage")
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_stage_heatmaps_runs():
    df = _df_long()
    fig, axes = plot_stage_heatmaps(df, value_col="Z", stage_col="stage", ncols=2)
    assert fig is not None
    assert axes is not None
    plt.close(fig)


def test_plot_rewiring_curves_runs():
    df = _df_long()
    pairs = [("T", "B"), ("B", "T")]
    fig, ax = plot_rewiring_curves(df, pairs=pairs, value_col="Z", stage_col="stage")
    assert fig is not None
    assert ax is not None
    plt.close(fig)
