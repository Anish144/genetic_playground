"""Prepare AnnData rows for transfer models.

Rows are cells or image crops. obs["perturbation"] names each category, with
"control" reserved for controls; obs["batch"] names each environment.
G counts perturbations, D counts batches, J counts measured outputs, and P counts
observed (perturbation, batch) pairs. Missing pairs carry no response weight.
"""

from collections.abc import Mapping
from typing import NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def perturbation_labels(adata: ad.AnnData) -> np.ndarray:
    """Read perturbation labels as strings.

    Parameters
    ----------
    adata : anndata.AnnData
        Rows with an obs["perturbation"] column.

    Returns
    -------
    numpy.ndarray of shape (n_cells,)
        One label per input row.
    """
    return adata.obs["perturbation"].astype(str).to_numpy()


def batch_labels(adata: ad.AnnData) -> np.ndarray:
    """Read batch labels as strings.

    Parameters
    ----------
    adata : anndata.AnnData
        Rows with an obs["batch"] column.

    Returns
    -------
    numpy.ndarray of shape (n_cells,)
        One batch label per input row.
    """
    return adata.obs["batch"].astype(str).to_numpy()


def is_control(adata: ad.AnnData) -> np.ndarray:
    """Select rows whose perturbation label is "control".

    Parameters
    ----------
    adata : anndata.AnnData
        Rows with an obs["perturbation"] column.

    Returns
    -------
    numpy.ndarray of shape (n_cells,)
        Boolean mask of control rows.
    """
    return perturbation_labels(adata) == "control"


def training_batch_counts(train: ad.AnnData) -> pd.Series:
    """Count distinct training batches containing each perturbation.

    Parameters
    ----------
    train : anndata.AnnData
        Training rows with perturbation and batch labels in obs.

    Returns
    -------
    pandas.Series
        Number of batches indexed by perturbation label, excluding controls.
    """
    obs = train.obs.loc[~is_control(train)]
    return obs.groupby("perturbation", observed=True)["batch"].nunique()


def label_positions(sorted_labels: np.ndarray, labels) -> np.ndarray:
    """Find label indices and reject labels missing from the source array.

    Parameters
    ----------
    sorted_labels : numpy.ndarray
        Sorted, unique source labels.
    labels : array-like
        Labels to locate. Every label must occur in sorted_labels.

    Returns
    -------
    numpy.ndarray
        Indices in the order requested.

    Raises
    ------
    ValueError
        If a requested label is missing from sorted_labels.
    """
    positions = pd.Index(sorted_labels).get_indexer(labels)
    if np.any(positions < 0):
        missing = np.asarray(labels)[positions < 0].tolist()
        raise ValueError(f"Unknown labels: {missing}")
    return positions


def _onehot(idx: np.ndarray, n_groups: int) -> sparse.csr_matrix:
    """Build a sparse (n_groups, n_rows) matrix with one entry per column."""
    ones = np.ones(len(idx))
    return sparse.csr_matrix((ones, (idx, np.arange(len(idx)))), shape=(n_groups, len(idx)))


def as_dense(m) -> np.ndarray:
    """Read a sparse or dense matrix as a NumPy array.

    Parameters
    ----------
    m : array-like or scipy.sparse matrix
        Input values.

    Returns
    -------
    numpy.ndarray
        Dense values. An already-dense array may be returned without copying.
    """
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


def _group_sums(idx: np.ndarray, n_groups: int, X) -> np.ndarray:
    """Sum rows of X by group; return a dense (n_groups, J) array."""
    return as_dense(_onehot(idx, n_groups) @ X)


def _group_means(idx: np.ndarray, n_groups: int, X) -> np.ndarray:
    """Average rows of X by group; every group must contain a row."""
    counts = np.bincount(idx, minlength=n_groups)
    return _group_sums(idx, n_groups, X) / counts[:, None]


def _require_controls(perturbed_batches: np.ndarray, control_batches: np.ndarray) -> None:
    """Reject response batches that have no control samples."""
    if not np.isin(perturbed_batches, control_batches).all():
        raise ValueError("Every batch with perturbed cells needs control cells to define W_e.")


def _batch_counts(labels: np.ndarray, batch: np.ndarray) -> pd.DataFrame:
    """Count responses by perturbation and batch from their labels.

    Return a DataFrame with sorted perturbation rows and batch columns.
    Control-only batches get zero columns; response batches need controls.
    """
    ctrl = labels == "control"
    batches = np.unique(batch[ctrl])
    _require_controls(batch[~ctrl], batches)
    perts, p_idx = np.unique(labels[~ctrl], return_inverse=True)
    b_idx = label_positions(batches, batch[~ctrl])
    pair_idx = p_idx * len(batches) + b_idx
    counts = np.bincount(pair_idx, minlength=len(perts) * len(batches))
    counts = counts.reshape(len(perts), len(batches))
    return pd.DataFrame(counts, index=perts, columns=batches)


def summary_stats(adata: ad.AnnData) -> dict:
    """Summarise category counts, response means and batch controls.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells by outputs, with perturbation and batch labels in obs.
        Controls use the perturbation label "control".

    Returns
    -------
    stats : dict
        Sorted labels: perturbations (G,) and batches (D,).
        Counts: n (G,), batch_n (D,) and pert_batch_n (G, D).
        Responses: y_mean (G, J) and batch_y_sum (D, J).
        Controls: batch means W (D, J) and category means w_mean (G, J).
        w_mean weights W by each category's response counts.
    """
    labels, batch = perturbation_labels(adata), batch_labels(adata)
    counts = _batch_counts(labels, batch)
    perts, batches = counts.index.to_numpy(), counts.columns.to_numpy()
    ctrl = labels == "control"
    ctrl_b = label_positions(batches, batch[ctrl])
    p_idx = label_positions(perts, labels[~ctrl])
    b_idx = label_positions(batches, batch[~ctrl])
    Y = adata.X[~ctrl]

    W = _group_means(ctrl_b, len(batches), adata.X[ctrl])  # (batches, genes)
    n = np.bincount(p_idx, minlength=len(perts))
    pert_batch_n = counts.to_numpy()
    return {
        "perturbations": perts,
        "n": n,
        "y_mean": _group_means(p_idx, len(perts), Y),
        "w_mean": pert_batch_n @ W / n[:, None],
        "batches": batches,
        "pert_batch_n": pert_batch_n,
        "W": W,
        "batch_n": np.bincount(b_idx, minlength=len(batches)),
        "batch_y_sum": _group_sums(b_idx, len(batches), Y),
    }


def matched_control_mean(adata: ad.AnnData, perturbations) -> np.ndarray:
    """Average batch control means using each perturbation's response counts.

    Parameters
    ----------
    adata : anndata.AnnData
        Control values and perturbation/batch labels. Response values are unused.
    perturbations : list of str
        Observed perturbations to select, in the desired order.

    Returns
    -------
    numpy.ndarray of shape (len(perturbations), J)
        Control means matched to the requested perturbations' batch distributions.

    Raises
    ------
    ValueError
        If a label is missing or a response batch has no controls.
    """
    labels, batch = perturbation_labels(adata), batch_labels(adata)
    counts = _batch_counts(labels, batch)
    ctrl = labels == "control"
    ctrl_b = label_positions(counts.columns.to_numpy(), batch[ctrl])
    means = _group_means(ctrl_b, len(counts.columns), adata.X[ctrl])
    selected = counts.to_numpy()[label_positions(counts.index.to_numpy(), perturbations)]
    return (selected @ means) / selected.sum(axis=1, keepdims=True)


class BridgeData(NamedTuple):
    """Observed perturbation/batch means and controls for a transfer bridge.

    Attributes
    ----------
    perturbations : numpy.ndarray of shape (G,)
        Sorted perturbation labels, excluding controls.
    ctrl : list of numpy.ndarray
        Control samples for each batch, in sorted batch order.
    pert : numpy.ndarray of shape (P,)
        Perturbation index of each observed pair.
    batch : numpy.ndarray of shape (P,)
        Batch index of each observed pair.
    y : numpy.ndarray of shape (P, J)
        Mean target for each observed pair.
    weight : numpy.ndarray of shape (P,)
        Response count for each pair, or one when count weighting is disabled.

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> import pandas as pd
    >>> obs = pd.DataFrame(
    ...     {"perturbation": ["control", "a", "control", "a"],
    ...      "batch": ["b0", "b0", "b1", "b1"]}, index=["0", "1", "2", "3"])
    >>> data = ad.AnnData(np.array([[0.], [1.], [2.], [5.]]), obs=obs)
    >>> from genetic_perturbation_playground.data.transfer import bridge_data
    >>> grouped = bridge_data(data)
    >>> grouped.perturbations.tolist()
    ['a']
    >>> grouped.y.tolist()
    [[1.0], [5.0]]
    """

    perturbations: np.ndarray  # sorted perturbation labels
    ctrl: list[np.ndarray]  # control cells of each batch
    pert: np.ndarray  # perturbation index of each pair
    batch: np.ndarray  # batch index of each pair
    y: np.ndarray  # mean expression of each pair
    weight: np.ndarray  # cell count of each pair, or 1 if not weighted by cells


def bridge_data(adata: ad.AnnData, weight_by_cells: bool = True) -> BridgeData:
    """Group response rows into observed perturbation/batch pairs.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells by outputs, with perturbation and batch labels in obs.
        Controls use the perturbation label "control".
    weight_by_cells : bool, default=True
        Use pair response counts as weights; False gives each observed pair weight one.

    Returns
    -------
    BridgeData
        Pair means, weights and per-batch control samples. Missing pairs are omitted.
    """
    ctrl = is_control(adata)
    batch = batch_labels(adata)
    _require_controls(batch[~ctrl], batch[ctrl])
    batches, b_idx = np.unique(batch, return_inverse=True)
    perts, p_idx = np.unique(perturbation_labels(adata)[~ctrl], return_inverse=True)
    n_b = len(batches)
    pairs, pair_idx = np.unique(p_idx * n_b + b_idx[~ctrl], return_inverse=True)
    n = np.bincount(pair_idx)
    return BridgeData(
        perturbations=perts,
        ctrl=_per_batch_controls(adata, ctrl, b_idx, n_b),
        pert=pairs // n_b,
        batch=pairs % n_b,
        y=_group_means(pair_idx, len(pairs), adata.X[~ctrl]),
        weight=n if weight_by_cells else np.ones_like(n),
    )


def _per_batch_controls(adata, ctrl: np.ndarray, b_idx: np.ndarray, n_batches: int) -> list:
    """Collect dense control arrays in batch-index order."""
    return [as_dense(adata.X[ctrl & (b_idx == b)]) for b in range(n_batches)]


def batch_weights(test: ad.AnnData, perturbations) -> np.ndarray:
    """Calculate each perturbation's share of responses in each batch.

    Parameters
    ----------
    test : anndata.AnnData
        Test rows with perturbation and batch labels. Only obs is read.
    perturbations : list of str
        Observed perturbations to select, in the desired order.

    Returns
    -------
    numpy.ndarray of shape (len(perturbations), D)
        Response fractions, with rows summing to one and sorted batch columns.
        Control-only batches have zero weight.

    Raises
    ------
    ValueError
        If a label is missing or a response batch has no controls.
    """
    counts = _batch_counts(perturbation_labels(test), batch_labels(test))
    selected = counts.to_numpy()[label_positions(counts.index.to_numpy(), perturbations)]
    return selected / selected.sum(axis=1, keepdims=True)


def batch_controls(test: ad.AnnData) -> list[np.ndarray]:
    """Collect control samples from each batch.

    Parameters
    ----------
    test : anndata.AnnData
        Control values and perturbation/batch labels in obs.

    Returns
    -------
    list of numpy.ndarray
        Dense control arrays in sorted batch order.

    Raises
    ------
    ValueError
        If a response batch has no controls.
    """
    ctrl, batch = is_control(test), batch_labels(test)
    _require_controls(batch[~ctrl], batch[ctrl])
    batches, b_idx = np.unique(batch, return_inverse=True)
    return _per_batch_controls(test, ctrl, b_idx, len(batches))


def validate_features(adata: ad.AnnData, feature_names: np.ndarray) -> None:
    """Require the same unique feature names, in the same order, as training.

    Parameters
    ----------
    adata : anndata.AnnData
        Prediction or observed-response data.
    feature_names : numpy.ndarray
        Column names recorded during fitting.

    Returns
    -------
    None
        Raise ValueError when feature names are empty, duplicated or mismatched.
    """
    expected = pd.Index(feature_names)
    if (
        not len(expected)
        or not expected.is_unique
        or not adata.var_names.is_unique
        or not adata.var_names.equals(expected)
    ):
        raise ValueError("Input feature names must be unique and match training features in order.")


def reference_weights(
    reference: ad.AnnData,
    *,
    feature_names: np.ndarray,
    environment_weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Validate reference controls and align their environment probabilities.

    Parameters
    ----------
    reference : anndata.AnnData
        Nonempty controls only, with finite X and perturbation/batch labels in obs.
    feature_names : numpy.ndarray
        Unique training feature names, in training order.
    environment_weights : mapping of str to float or None, default=None
        Probability of every reference batch. None gives each batch equal weight,
        regardless of its control count. Supplied probabilities must be finite,
        nonnegative and sum to one within absolute tolerance 1e-8.

    Returns
    -------
    pandas.Series
        Normalized probabilities in sorted batch order, named probability,
        with index named batch. Invalid inputs raise ValueError.
    """
    validate_features(reference, feature_names)
    if not reference.n_obs:
        raise ValueError("reference must contain control samples.")
    for column in ("perturbation", "batch"):
        if column not in reference.obs or reference.obs[column].isna().any():
            raise ValueError(f"reference.obs must contain nonmissing {column} labels.")
    if not is_control(reference).all():
        raise ValueError("reference perturbation labels must all be control.")
    _require_finite(reference.X, "reference.X")
    batches = np.unique(batch_labels(reference))
    if environment_weights is None:
        weights = np.full(len(batches), 1 / len(batches))
    else:
        if not isinstance(environment_weights, Mapping) or set(environment_weights) != set(batches):
            raise ValueError("environment_weights must name exactly the reference batches.")
        try:
            weights = np.array([environment_weights[b] for b in batches], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("environment_weights must contain numeric probabilities.") from error
        if (
            weights.shape != (len(batches),)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError("environment_weights must be finite, nonnegative and sum to one.")
    return pd.Series(
        weights / weights.sum(), index=pd.Index(batches, name="batch"), name="probability"
    )


def _require_finite(X, name: str) -> None:
    """Reject unavailable or nonfinite measurements without densifying sparse X."""
    if X is None or not np.isfinite(X.data if sparse.issparse(X) else X).all():
        raise ValueError(f"{name} must contain finite measurements.")


def observed_batch_means(
    observed: ad.AnnData, perturbations: list[str], batch: str
) -> pd.DataFrame:
    """Average measured responses for requested perturbations in one batch.

    Parameters
    ----------
    observed : anndata.AnnData
        Response rows with perturbation/batch labels. Controls are optional and
        ignored. Missing pairs have no rows; zero responses are valid measurements.
    perturbations : list of str
        Requested labels, in result order. Unobserved labels are omitted.
    batch : str
        Environment to select, using the string form of its batch label.

    Returns
    -------
    pandas.DataFrame
        Present perturbations by output features. Selected measurements must be
        finite. No selected responses gives an empty frame with all output columns.
    """
    labels = perturbation_labels(observed)
    selected = (
        (labels != "control")
        & pd.Index(labels).isin(perturbations)
        & (batch_labels(observed) == batch)
    )
    if not selected.any():
        return pd.DataFrame(
            index=pd.Index([], name="perturbation"), columns=observed.var_names, dtype=float
        )
    values = observed[selected].X
    _require_finite(values, "observed.X")
    present, groups = np.unique(labels[selected], return_inverse=True)
    means = pd.DataFrame(
        _group_means(groups, len(present), values),
        index=pd.Index(present, name="perturbation"),
        columns=observed.var_names,
    )
    return means.loc[[p for p in perturbations if p in means.index]]
