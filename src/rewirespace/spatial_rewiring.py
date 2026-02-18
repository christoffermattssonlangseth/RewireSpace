from __future__ import annotations

import warnings
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import sparse


def _as_series(labels: pd.Series) -> pd.Series:
    if isinstance(labels, pd.Series):
        return labels
    return pd.Series(labels)


def _stage_order(stage: pd.Series) -> list[Any]:
    if isinstance(stage.dtype, pd.CategoricalDtype):
        return list(stage.cat.categories)

    values = [v for v in pd.unique(stage) if pd.notna(v)]
    try:
        return sorted(values)
    except TypeError:
        return list(values)


def _onehot_from_codes(codes: np.ndarray, n_types: int) -> np.ndarray:
    n_cells = codes.shape[0]
    X = np.zeros((n_cells, n_types), dtype=np.float32)
    if n_types == 0:
        return X

    valid = codes >= 0
    if np.any(valid):
        rows = np.flatnonzero(valid)
        X[rows, codes[valid]] = 1.0
    return X


def _maybe_tqdm(iterable: Iterable, **kwargs: Any) -> Iterable:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def _extract_matrix_from_entry(entry: Any) -> sparse.csr_matrix:
    if sparse.issparse(entry):
        return entry.tocsr()

    if isinstance(entry, np.ndarray):
        return sparse.csr_matrix(entry)

    if isinstance(entry, Mapping):
        for key in ("A", "adjacency", "matrix", "adj"):
            if key in entry:
                return _extract_matrix_from_entry(entry[key])

    raise TypeError(
        "Unsupported per-sample adjacency entry. Expected sparse matrix, dense "
        "array, or mapping with one of keys: A/adjacency/matrix/adj."
    )


def _resolve_adjacency_mode(adata: Any, adj_key: str) -> tuple[str, Any]:
    if adj_key in adata.obsp:
        A = adata.obsp[adj_key]
        if sparse.issparse(A):
            A = A.tocsr()
            if A.shape != (adata.n_obs, adata.n_obs):
                raise ValueError(
                    f"`adata.obsp[{adj_key!r}]` has shape {A.shape}, expected "
                    f"({adata.n_obs}, {adata.n_obs})."
                )
            return "global", A

        if isinstance(A, Mapping):
            return "per_sample", A

        raise TypeError(
            f"`adata.obsp[{adj_key!r}]` must be a sparse matrix or a per-sample mapping."
        )

    if adj_key in adata.uns and isinstance(adata.uns[adj_key], Mapping):
        return "per_sample", adata.uns[adj_key]

    raise KeyError(
        f"Could not find adjacency under adata.obsp[{adj_key!r}] or "
        f"per-sample mapping under adata.uns[{adj_key!r}]."
    )


def _sample_single_value(values: pd.Series, sample_id: Any, key: str) -> Any:
    non_null = values.dropna().unique()
    if len(non_null) == 0:
        return np.nan
    if len(non_null) > 1:
        warnings.warn(
            f"Sample {sample_id!r} has multiple values in `{key}`; using first value "
            f"{non_null[0]!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return non_null[0]


def get_onehot_types(labels: pd.Series) -> tuple[np.ndarray, list[str]]:
    """
    Create a dense one-hot matrix of cell types.

    Parameters
    ----------
    labels
        Cell type labels (length n_cells).

    Returns
    -------
    X, types
        X is shape (n_cells, n_types), and types are in column order.
    """
    labels = _as_series(labels)

    if isinstance(labels.dtype, pd.CategoricalDtype):
        categories = list(labels.cat.categories)
        codes = labels.cat.codes.to_numpy(copy=False)
    else:
        categories = list(pd.unique(labels.dropna()))
        cat = pd.Categorical(labels, categories=categories, ordered=False)
        codes = cat.codes

    X = _onehot_from_codes(np.asarray(codes, dtype=np.int64), len(categories))
    types = [str(t) for t in categories]
    return X, types


def type_type_contacts(
    A: sparse.spmatrix, labels: pd.Series
) -> tuple[np.ndarray, list[str]]:
    """
    Compute type-type contacts: C = X.T @ A @ X.

    Parameters
    ----------
    A
        Sparse adjacency matrix (n_cells x n_cells).
    labels
        Cell type labels (length n_cells).

    Returns
    -------
    C, types
        Contact matrix C (n_types x n_types) and type order.
    """
    labels = _as_series(labels)
    A = A.tocsr() if sparse.issparse(A) else sparse.csr_matrix(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if A.shape[0] != len(labels):
        raise ValueError(
            f"Adjacency size ({A.shape[0]}) does not match number of labels ({len(labels)})."
        )

    X, types = get_onehot_types(labels)
    AX = A @ X
    C = X.T @ AX
    return np.asarray(C, dtype=np.float64), types


def permute_enrichment_z(
    A: sparse.spmatrix,
    labels: pd.Series,
    n_perm: int = 200,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Compute permutation-based enrichment Z scores for type-type contacts.

    Z = (C_obs - mean(C_perm)) / std(C_perm)
    where labels are randomly shuffled for each permutation.
    """
    if n_perm <= 0:
        raise ValueError("`n_perm` must be >= 1.")

    labels = _as_series(labels)
    A = A.tocsr() if sparse.issparse(A) else sparse.csr_matrix(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if A.shape[0] != len(labels):
        raise ValueError(
            f"Adjacency size ({A.shape[0]}) does not match number of labels ({len(labels)})."
        )

    if isinstance(labels.dtype, pd.CategoricalDtype):
        categories = list(labels.cat.categories)
        codes = labels.cat.codes.to_numpy(copy=False)
    else:
        categories = list(pd.unique(labels.dropna()))
        cat = pd.Categorical(labels, categories=categories, ordered=False)
        codes = cat.codes

    codes = np.asarray(codes, dtype=np.int64)
    n_types = len(categories)

    X_obs = _onehot_from_codes(codes, n_types)
    C_obs = np.asarray(X_obs.T @ (A @ X_obs), dtype=np.float64)

    C_sum = np.zeros_like(C_obs, dtype=np.float64)
    C_sumsq = np.zeros_like(C_obs, dtype=np.float64)

    rng = np.random.default_rng(random_state)
    for _ in range(n_perm):
        perm_codes = rng.permutation(codes)
        X_perm = _onehot_from_codes(perm_codes, n_types)
        C_perm = np.asarray(X_perm.T @ (A @ X_perm), dtype=np.float64)
        C_sum += C_perm
        C_sumsq += C_perm * C_perm

    C_mean = C_sum / float(n_perm)
    C_var = (C_sumsq / float(n_perm)) - (C_mean * C_mean)
    C_var = np.maximum(C_var, 0.0)
    C_std = np.sqrt(C_var)

    eps = np.finfo(np.float64).eps
    Z = np.zeros_like(C_obs, dtype=np.float64)
    valid = C_std > eps
    Z[valid] = (C_obs[valid] - C_mean[valid]) / (C_std[valid] + eps)
    Z[~valid] = 0.0

    types = [str(t) for t in categories]
    return Z, C_obs, C_mean, C_std, types


def per_sample_contacts(
    adata: Any,
    adj_key: str,
    cell_type_key: str,
    sample_key: str,
    stage_key: str,
    subject_key: Optional[str] = None,
    n_perm: int = 200,
) -> pd.DataFrame:
    """
    Compute observed contact counts and permutation Z-scores per sample.

    Returns a tidy long DataFrame with columns:
    sample_id, stage, [subject], type_i, type_j, C_obs, Z, n_cells, n_edges
    """
    required_obs = [cell_type_key, sample_key, stage_key]
    if subject_key is not None:
        required_obs.append(subject_key)

    missing = [k for k in required_obs if k not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required adata.obs columns: {missing}")

    mode, A_source = _resolve_adjacency_mode(adata, adj_key)

    obs = adata.obs.copy()
    stage_order = _stage_order(obs[stage_key])

    cell_types = obs[cell_type_key]
    if isinstance(cell_types.dtype, pd.CategoricalDtype):
        global_types = list(cell_types.cat.categories)
    else:
        global_types = list(pd.unique(cell_types.dropna()))

    sample_values = obs[sample_key]
    sample_ids = list(pd.unique(sample_values))
    sample_seeds = np.random.default_rng(0).integers(
        0, np.iinfo(np.int32).max, size=len(sample_ids), dtype=np.int64
    )

    chunks: list[pd.DataFrame] = []
    iterator = _maybe_tqdm(
        enumerate(sample_ids),
        total=len(sample_ids),
        desc="Per-sample contacts",
    )

    for i, sample_id in iterator:
        mask = sample_values == sample_id
        idx = np.flatnonzero(mask.to_numpy())
        if idx.size == 0:
            continue

        sample_obs = obs.iloc[idx]
        labels_sample = pd.Series(
            pd.Categorical(
                sample_obs[cell_type_key],
                categories=global_types,
                ordered=False,
            ),
            index=sample_obs.index,
        )

        stage_val = _sample_single_value(sample_obs[stage_key], sample_id, stage_key)
        subject_val = (
            _sample_single_value(sample_obs[subject_key], sample_id, subject_key)
            if subject_key is not None
            else None
        )

        if mode == "global":
            A_sample = A_source[idx][:, idx].tocsr()
        else:
            entry = None
            if sample_id in A_source:
                entry = A_source[sample_id]
            elif str(sample_id) in A_source:
                entry = A_source[str(sample_id)]
            if entry is None:
                raise KeyError(
                    f"No per-sample adjacency found for sample_id={sample_id!r}."
                )
            A_sample = _extract_matrix_from_entry(entry)
            if A_sample.shape != (idx.size, idx.size):
                raise ValueError(
                    f"Per-sample adjacency for sample {sample_id!r} has shape "
                    f"{A_sample.shape}, expected {(idx.size, idx.size)}."
                )

        Z, C_obs, _, _, types = permute_enrichment_z(
            A_sample,
            labels_sample,
            n_perm=n_perm,
            random_state=int(sample_seeds[i]),
        )

        n_types = len(types)
        n_pairs = n_types * n_types
        sample_df = pd.DataFrame(
            {
                "sample_id": np.repeat(sample_id, n_pairs),
                "stage": np.repeat(stage_val, n_pairs),
                "type_i": np.repeat(types, n_types),
                "type_j": np.tile(types, n_types),
                "C_obs": C_obs.reshape(-1),
                "Z": Z.reshape(-1),
                "n_cells": np.repeat(idx.size, n_pairs),
                "n_edges": np.repeat(int(A_sample.nnz), n_pairs),
            }
        )

        if subject_key is not None:
            sample_df["subject"] = subject_val

        chunks.append(sample_df)

    if not chunks:
        base_cols = [
            "sample_id",
            "stage",
            "type_i",
            "type_j",
            "C_obs",
            "Z",
            "n_cells",
            "n_edges",
        ]
        if subject_key is not None:
            base_cols.append("subject")
        return pd.DataFrame(columns=base_cols)

    out = pd.concat(chunks, ignore_index=True)
    if stage_order:
        out["stage"] = pd.Categorical(out["stage"], categories=stage_order, ordered=True)

    ordered_cols = [
        "sample_id",
        "stage",
        "type_i",
        "type_j",
        "C_obs",
        "Z",
        "n_cells",
        "n_edges",
    ]
    if subject_key is not None:
        ordered_cols.insert(2, "subject")

    return out[ordered_cols]


def fit_stage_effect(
    df_long: pd.DataFrame,
    value_col: str = "Z",
    stage_col: str = "stage",
    subject_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fit stage effect per (type_i, type_j).

    Model:
      value ~ C(stage)

    If subject_col is provided and has >=2 unique subjects for an edge,
    cluster-robust SE by subject is used.

    Returns
    -------
    DataFrame with one row per edge and columns including effect sizes,
    p_value, q_value (BH FDR), and rank.
    """
    required = {"type_i", "type_j", value_col, stage_col}
    missing = required.difference(df_long.columns)
    if missing:
        raise KeyError(f"Missing required columns in df_long: {sorted(missing)}")

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.multitest import multipletests
    except ImportError as e:
        raise ImportError(
            "fit_stage_effect requires `statsmodels` to be installed."
        ) from e

    df = df_long.copy()
    df = df.dropna(subset=["type_i", "type_j", value_col, stage_col])

    if df.empty:
        return pd.DataFrame(
            columns=[
                "type_i",
                "type_j",
                "n_obs",
                "n_stages",
                "effect_range",
                "coef_abs_max",
                "p_value",
                "q_value",
                "rank",
                "model",
                "r_squared",
            ]
        )

    stage_order = _stage_order(df[stage_col])
    if stage_order:
        df[stage_col] = pd.Categorical(df[stage_col], categories=stage_order, ordered=True)

    formula = f"Q('{value_col}') ~ C(Q('{stage_col}'))"
    term_prefix = f"C(Q('{stage_col}'))"

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["type_i", "type_j"], sort=False, observed=True)

    iterator = _maybe_tqdm(grouped, total=grouped.ngroups, desc="Stage models")
    for (type_i, type_j), g in iterator:
        stage_means = g.groupby(stage_col, observed=False)[value_col].mean()
        stage_means = stage_means.reindex(stage_order) if stage_order else stage_means

        if stage_means.notna().any():
            effect_range = float(stage_means.max(skipna=True) - stage_means.min(skipna=True))
        else:
            effect_range = np.nan

        n_stages = int(g[stage_col].nunique(dropna=True))
        row: dict[str, Any] = {
            "type_i": type_i,
            "type_j": type_j,
            "n_obs": int(len(g)),
            "n_stages": n_stages,
            "effect_range": effect_range,
            "coef_abs_max": np.nan,
            "p_value": np.nan,
            "model": "unfit",
            "r_squared": np.nan,
        }

        if "sample_id" in g.columns:
            row["n_samples"] = int(g["sample_id"].nunique(dropna=True))
        if subject_col is not None and subject_col in g.columns:
            row["n_subjects"] = int(g[subject_col].nunique(dropna=True))

        if n_stages < 2:
            rows.append(row)
            continue

        try:
            if subject_col is not None and subject_col in g.columns:
                groups = g[subject_col]
                enough_groups = (groups.nunique(dropna=True) >= 2) and (len(g) > n_stages)
            else:
                enough_groups = False

            if enough_groups:
                try:
                    fit = smf.ols(formula, data=g).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": groups.astype(str)},
                    )
                    row["model"] = "ols_cluster_subject"
                except Exception:
                    fit = smf.ols(formula, data=g).fit()
                    row["model"] = "ols"
            else:
                fit = smf.ols(formula, data=g).fit()
                row["model"] = "ols"
        except Exception as e:
            row["model"] = f"failed: {type(e).__name__}"
            rows.append(row)
            continue

        coef_names = [name for name in fit.params.index if name.startswith(term_prefix)]
        if coef_names:
            row["coef_abs_max"] = float(np.abs(fit.params[coef_names]).max())
            R = np.zeros((len(coef_names), len(fit.params)), dtype=float)
            for r, coef_name in enumerate(coef_names):
                R[r, fit.params.index.get_loc(coef_name)] = 1.0
            try:
                try:
                    wald = fit.wald_test(R, scalar=True)
                except TypeError:
                    wald = fit.wald_test(R)
                row["p_value"] = float(np.asarray(wald.pvalue).reshape(-1)[0])
            except Exception:
                row["p_value"] = np.nan
        else:
            row["coef_abs_max"] = 0.0

        row["r_squared"] = float(getattr(fit, "rsquared", np.nan))

        rows.append(row)

    out = pd.DataFrame(rows)

    out["q_value"] = np.nan
    valid = np.isfinite(out["p_value"].to_numpy(dtype=float))
    if np.any(valid):
        qvals = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
        out.loc[valid, "q_value"] = qvals

    out = out.sort_values(
        ["q_value", "p_value", "effect_range"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    preferred = [
        "rank",
        "type_i",
        "type_j",
        "n_obs",
        "n_samples",
        "n_subjects",
        "n_stages",
        "effect_range",
        "coef_abs_max",
        "p_value",
        "q_value",
        "model",
        "r_squared",
    ]
    present = [c for c in preferred if c in out.columns]
    rest = [c for c in out.columns if c not in present]
    return out[present + rest]


__all__ = [
    "get_onehot_types",
    "type_type_contacts",
    "permute_enrichment_z",
    "per_sample_contacts",
    "fit_stage_effect",
]
