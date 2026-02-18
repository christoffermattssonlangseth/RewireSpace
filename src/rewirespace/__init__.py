"""RewireSpace: graph rewiring analysis for spatial omics."""

from .spatial_rewiring import (
    fit_stage_effect,
    get_onehot_types,
    per_sample_contacts,
    permute_enrichment_z,
    type_type_contacts,
)
from .plot_rewiring import (
    plot_rewiring_curves,
    plot_stage_heatmap,
    plot_stage_heatmaps,
)

__all__ = [
    "get_onehot_types",
    "type_type_contacts",
    "permute_enrichment_z",
    "per_sample_contacts",
    "fit_stage_effect",
    "plot_stage_heatmap",
    "plot_stage_heatmaps",
    "plot_rewiring_curves",
]
