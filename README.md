# RewireSpace

Neighbor-matrix-centric **graph rewiring over disease course** for spatial omics data in `AnnData`.

RewireSpace computes per-sample type-type contact matrices from spatial adjacency graphs, estimates permutation-based enrichment Z-scores, and fits stage-wise models to rank dynamically rewiring edges.

## Features

- Efficient type-type contact computation using `C = X^T A X`
- Permutation-based enrichment Z-score per type pair
- Per-sample tidy output for downstream analysis
- Stage-effect modeling with optional subject-level clustered SE
- Plotting utilities for stage heatmaps and rewiring curves

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## Quick Start

```python
from rewirespace.spatial_rewiring import per_sample_contacts, fit_stage_effect
from rewirespace.plot_rewiring import plot_stage_heatmaps, plot_rewiring_curves

# adata.obs must include cell_type/stage/sample columns (subject optional)
# adata.obsp["spatial_connectivities"] must contain sparse adjacency

df_long = per_sample_contacts(
    adata,
    adj_key="spatial_connectivities",
    cell_type_key="cell_type",
    sample_key="sample_id",
    stage_key="stage",
    subject_key="mouse_id",  # or None
    n_perm=200,
)

results = fit_stage_effect(df_long, value_col="Z", stage_col="stage", subject_col="subject")

fig, _ = plot_stage_heatmaps(df_long, value_col="Z", stage_col="stage")
```

## Repository Layout

```text
.
├── src/rewirespace/
│   ├── __init__.py
│   ├── spatial_rewiring.py
│   └── plot_rewiring.py
├── tests/
├── examples/
│   └── example_rewiring.ipynb
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Example Notebook

A runnable showcase notebook is provided at:

- `examples/example_rewiring.ipynb`

## Testing

```bash
pytest
```

## License

MIT License. See `LICENSE`.
