import numpy as np
import pandas as pd
from scipy import sparse

from rewirespace.spatial_rewiring import (
    fit_stage_effect,
    get_onehot_types,
    per_sample_contacts,
    permute_enrichment_z,
    type_type_contacts,
)


class MockAnnData:
    def __init__(self, obs: pd.DataFrame, A: sparse.csr_matrix, adj_key: str = "spatial_connectivities"):
        self.obs = obs
        self.obsp = {adj_key: A}
        self.uns = {}
        self.n_obs = obs.shape[0]


def _make_mock_adata() -> MockAnnData:
    obs = pd.DataFrame(
        {
            "cell_type": ["T", "B", "T", "B", "T", "B", "T", "B", "T", "B", "T", "B"],
            "sample_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2", "s3", "s3", "s3", "s3"],
            "stage": pd.Categorical(
                ["early", "early", "early", "early", "mid", "mid", "mid", "mid", "late", "late", "late", "late"],
                categories=["early", "mid", "late"],
                ordered=True,
            ),
            "mouse_id": ["m1", "m1", "m1", "m1", "m1", "m1", "m1", "m1", "m2", "m2", "m2", "m2"],
        }
    )

    blocks = []
    for _ in ["s1", "s2", "s3"]:
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
        blocks.append(sparse.csr_matrix(A))

    A_global = sparse.block_diag(blocks, format="csr")
    return MockAnnData(obs=obs, A=A_global)


def test_get_onehot_types_preserves_categorical_order():
    labels = pd.Series(pd.Categorical(["B", "T", "B"], categories=["T", "B"], ordered=True))
    X, types = get_onehot_types(labels)

    assert types == ["T", "B"]
    assert X.shape == (3, 2)
    np.testing.assert_array_equal(X.sum(axis=1), np.ones(3))


def test_type_type_contacts_small_graph():
    A = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    labels = pd.Series(["A", "B"])
    C, types = type_type_contacts(A, labels)

    assert types == ["A", "B"]
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(C, expected)


def test_permute_enrichment_zero_variance_returns_zero_z():
    A = sparse.csr_matrix(np.zeros((4, 4), dtype=float))
    labels = pd.Series(["A", "A", "B", "B"])

    Z, C_obs, C_mean, C_std, _ = permute_enrichment_z(A, labels, n_perm=20, random_state=1)

    np.testing.assert_allclose(C_obs, 0.0)
    np.testing.assert_allclose(C_mean, 0.0)
    np.testing.assert_allclose(C_std, 0.0)
    np.testing.assert_allclose(Z, 0.0)


def test_per_sample_contacts_and_fit_stage_effect():
    adata = _make_mock_adata()

    df_long = per_sample_contacts(
        adata=adata,
        adj_key="spatial_connectivities",
        cell_type_key="cell_type",
        sample_key="sample_id",
        stage_key="stage",
        subject_key="mouse_id",
        n_perm=20,
    )

    required_cols = {
        "sample_id",
        "stage",
        "subject",
        "type_i",
        "type_j",
        "C_obs",
        "Z",
        "n_cells",
        "n_edges",
    }
    assert required_cols.issubset(df_long.columns)
    assert not df_long.empty

    results = fit_stage_effect(df_long, value_col="Z", stage_col="stage", subject_col="subject")
    assert not results.empty
    assert {"rank", "type_i", "type_j", "p_value", "q_value", "effect_range"}.issubset(results.columns)
