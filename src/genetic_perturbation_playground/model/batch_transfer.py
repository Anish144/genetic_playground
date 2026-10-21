from typing import NamedTuple

import numpy as np
import scanpy as sc
from scipy import sparse

from genetic_perturbation_playground.model.proxy_unconfounded import (
    MarginalEmbeddingBridge,
    mean_inner,
)
from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralMarginalEmbeddingBridge,
)
from genetic_perturbation_playground.utils.kernels import RBF, median_heuristic


def perturbation_labels(adata: sc.AnnData) -> np.ndarray:
    """Perturbation of every cell, as strings."""
    return adata.obs["perturbation"].astype(str).to_numpy()


def batch_labels(adata: sc.AnnData) -> np.ndarray:
    """Batch of every cell, as strings."""
    return adata.obs["batch"].astype(str).to_numpy()


def is_control(adata: sc.AnnData) -> np.ndarray:
    """Boolean mask of the control cells."""
    return perturbation_labels(adata) == "control"


def _positions(sorted_labels: np.ndarray, labels) -> np.ndarray:
    """Index of each of `labels` in `sorted_labels`."""
    return np.searchsorted(sorted_labels, labels)


def _onehot(idx: np.ndarray, n_groups: int) -> sparse.csr_matrix:
    """Sparse (n_groups x n_cells) indicator matrix."""
    ones = np.ones(len(idx))
    return sparse.csr_matrix((ones, (idx, np.arange(len(idx)))), shape=(n_groups, len(idx)))


def _dense(m) -> np.ndarray:
    """Dense array from a sparse or dense matrix."""
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


def _group_sums(idx: np.ndarray, n_groups: int, X) -> np.ndarray:
    """Sum of the rows of X within each group, shape (n_groups, n_columns)."""
    return _dense(_onehot(idx, n_groups) @ X)


def _group_means(idx: np.ndarray, n_groups: int, X) -> np.ndarray:
    """Mean of the rows of X within each group, shape (n_groups, n_columns)."""
    counts = np.bincount(idx, minlength=n_groups)
    return _group_sums(idx, n_groups, X) / counts[:, None]


def _require_controls(perturbed_batches: np.ndarray, control_batches: np.ndarray) -> None:
    """Raise if a batch with perturbed cells has no control cells."""
    if not np.isin(perturbed_batches, control_batches).all():
        raise ValueError("Every batch with perturbed cells needs control cells to define W_e.")


def summary_stats(adata: sc.AnnData) -> dict:
    """Per-batch control means W_e and per-perturbation sufficient statistics.

    Returns perturbations, n (cells per perturbation), y_mean (mean of y over the perturbation's
    cells), w_mean (W_e averaged over the same cells, i.e. weighted by the perturbation's batch
    distribution), plus per-batch sums used to fit the slope.
    """
    ctrl = is_control(adata)
    batch = batch_labels(adata)
    batches, ctrl_b = np.unique(batch[ctrl], return_inverse=True)
    _require_controls(batch[~ctrl], batches)
    perts, p_idx = np.unique(perturbation_labels(adata)[~ctrl], return_inverse=True)
    b_idx = _positions(batches, batch[~ctrl])
    Y = adata.X[~ctrl]

    W = _group_means(ctrl_b, len(batches), adata.X[ctrl])  # (batches, genes)
    n = np.bincount(p_idx, minlength=len(perts))
    pert_batch_n = _dense(_onehot(p_idx, len(perts)) @ _onehot(b_idx, len(batches)).T)
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


def matched_control_mean(adata: sc.AnnData, perturbations) -> np.ndarray:
    """Control mean matched to each perturbation's batches in `adata`, shape (n_perts, n_genes)."""
    s = summary_stats(adata)
    return s["w_mean"][_positions(s["perturbations"], perturbations)]


def _within_perturbation_slope(s: dict) -> np.ndarray:
    """Per-gene least-squares slope of y on W_e, within perturbations (Frisch-Waugh)."""
    y, w, n = s["y_mean"], s["w_mean"], s["n"][:, None]
    num = (s["W"] * s["batch_y_sum"]).sum(axis=0) - (n * y * w).sum(axis=0)
    den = (s["batch_n"][:, None] * s["W"] ** 2).sum(axis=0) - (n * w**2).sum(axis=0)
    return num / den


class BatchTransferModel:
    """Out-of-batch transfer using the batch's control-cell mean W_e as environment proxy.

    Per gene, the mean of perturbation x in batch e is
        Y_e(x) = gamma_x + b * W_e,    b = beta_2 / beta_1,
    with the constant alpha_2 - b * alpha_1 absorbed into gamma_x. Fit by cell-level least squares
    (perturbation fixed effects + one slope per gene). For a new batch e*:
        Y_{e*}(x) = gamma_x + b * W_{e*}.

    slope=None fits b per gene; slope=1 is the beta_1 = beta_2 case (subtract W_e, no transfer
    needed); slope=0 ignores batch entirely (pooled perturbation means).
    """

    def __init__(self, slope: float | None = None):
        self.slope = slope

    def fit(self, adata: sc.AnnData) -> "BatchTransferModel":
        s = summary_stats(adata)
        self.slope_ = self._slope(s)
        self.perturbations_ = s["perturbations"]
        self.gamma_ = s["y_mean"] - self.slope_ * s["w_mean"]
        return self

    def predict(self, perturbations: list[str], test: sc.AnnData) -> np.ndarray:
        """Mean expression of each perturbation in `test`, given only `test`'s control cells."""
        gamma = self.gamma_[_positions(self.perturbations_, perturbations)]
        return gamma + self.slope_ * matched_control_mean(test, perturbations)

    def _slope(self, s: dict) -> np.ndarray:
        """The fixed slope for every gene, or the fitted one if slope is None."""
        if self.slope is None:
            return _within_perturbation_slope(s)
        return np.full(s["y_mean"].shape[1], float(self.slope))


class BridgeData(NamedTuple):
    """Bridge training data: one row per observed (perturbation, batch) pair."""

    perturbations: np.ndarray  # sorted perturbation labels
    ctrl: list[np.ndarray]  # control cells of each batch
    pert: np.ndarray  # perturbation index of each pair
    batch: np.ndarray  # batch index of each pair
    y: np.ndarray  # mean expression of each pair
    weight: np.ndarray  # cell count of each pair, or 1 if not weighted by cells


def _bridge_data(adata: sc.AnnData, weight_by_cells: bool = True) -> BridgeData:
    """Per-(perturbation, batch) means of `adata` and the control cells of each batch."""
    s = summary_stats(adata)
    ctrl = is_control(adata)
    n_b = len(s["batches"])
    b_idx = _positions(s["batches"], batch_labels(adata))
    p_idx = _positions(s["perturbations"], perturbation_labels(adata)[~ctrl])
    pairs, pair_idx = np.unique(p_idx * n_b + b_idx[~ctrl], return_inverse=True)
    n = np.bincount(pair_idx)
    return BridgeData(
        perturbations=s["perturbations"],
        ctrl=_per_batch_controls(adata, ctrl, b_idx, n_b),
        pert=pairs // n_b,
        batch=pairs % n_b,
        y=_group_means(pair_idx, len(pairs), adata.X[~ctrl]),
        weight=n if weight_by_cells else np.ones_like(n),
    )


def _per_batch_controls(adata, ctrl: np.ndarray, b_idx: np.ndarray, n_batches: int) -> list:
    """Control cells of each batch, as dense arrays."""
    return [_dense(adata.X[ctrl & (b_idx == b)]) for b in range(n_batches)]


def _batch_mix(test: sc.AnnData, perturbations) -> np.ndarray:
    """Share of each perturbation's cells in each batch of `test` (rows sum to 1)."""
    s = summary_stats(test)
    mix = s["pert_batch_n"] / s["n"][:, None]
    return mix[_positions(s["perturbations"], perturbations)]


def _test_controls(test: sc.AnnData) -> list[np.ndarray]:
    """Control cells of each batch of `test`."""
    batches = summary_stats(test)["batches"]
    b_idx = _positions(batches, batch_labels(test))
    return _per_batch_controls(test, is_control(test), b_idx, len(batches))


class KernelBridgeTransfer:
    """MarginalEmbeddingBridge on cells, with a delta kernel on the perturbation x.

    w is a control cell, so mu_e is the mean embedding of batch e's control cells. The delta kernel
    makes the cell-level Gram block-diagonal over perturbations, so the one cell-level bridge splits
    exactly into one bridge per perturbation over the batches it was seen in (x constant within a
    block), fit on its per-batch means. All blocks share M = <mu_e, mu_e'>, computed once.

    weight_by_cells=False gives every (perturbation, batch) mean equal weight instead.
    lengthscale_scale multiplies the median-heuristic lengthscale of the RBF kernel on w.
    """

    def __init__(
        self, reg: float = 1e-6, weight_by_cells: bool = True, lengthscale_scale: float = 1.0
    ):
        self.reg = reg
        self.weight_by_cells = weight_by_cells
        self.lengthscale_scale = lengthscale_scale

    def fit(self, adata: sc.AnnData) -> "KernelBridgeTransfer":
        data = _bridge_data(adata, self.weight_by_cells)
        self.perturbations_ = data.perturbations
        self.ctrl_ = data.ctrl
        lengthscale = self.lengthscale_scale * median_heuristic(np.concatenate(self.ctrl_))
        self.kernel_w_ = RBF(lengthscales=lengthscale)
        inner = mean_inner(self.kernel_w_, self.ctrl_)
        self.bridges_ = [self._fit_block(data, inner, p) for p in range(len(self.perturbations_))]
        return self

    def predict(self, perturbations: list[str], test: sc.AnnData) -> np.ndarray:
        """Mean expression of each perturbation in `test`, given only `test`'s control cells."""
        inner = mean_inner(self.kernel_w_, self.ctrl_, _test_controls(test))
        targets = _batch_mix(test, perturbations) @ inner.T
        bridges = [self.bridges_[p] for p in _positions(self.perturbations_, perturbations)]
        preds = [b.predict_embedded(np.zeros(1), t) for b, t in zip(bridges, targets, strict=True)]
        return np.concatenate(preds)

    def _fit_block(self, data: BridgeData, inner: np.ndarray, p: int) -> MarginalEmbeddingBridge:
        """Bridge for perturbation p alone, fit on its per-batch means."""
        rows = np.flatnonzero(data.pert == p)
        reg = self.reg * data.weight.sum() / data.weight[rows].sum()  # global N, as on cells
        bridge = MarginalEmbeddingBridge(RBF(), self.kernel_w_, reg)
        x = np.zeros(len(rows))  # constant within the block
        return bridge.fit_embedded(x, data.y[rows], data.batch[rows], inner, data.weight[rows])


class NeuralBridgeTransfer:
    """NeuralMarginalEmbeddingBridge with x = one-hot perturbation and w = control cell.

    Fit on the same count-weighted per-(perturbation, batch) means as KernelBridgeTransfer. Unlike
    the delta kernel, phi(x) and A are shared, so perturbations borrow strength from each other.
    weight_by_cells=False gives every (perturbation, batch) mean equal weight instead.
    """

    def __init__(self, weight_by_cells: bool = True, **bridge_kwargs):
        self.weight_by_cells = weight_by_cells
        self.bridge_kwargs = bridge_kwargs

    def fit(self, adata: sc.AnnData, checkpoints=(), on_checkpoint=None) -> "NeuralBridgeTransfer":
        """Fit on `adata`; checkpoints and on_checkpoint as in NeuralMarginalEmbeddingBridge.fit."""
        data = _bridge_data(adata, self.weight_by_cells)
        self.perturbations_ = data.perturbations
        envs = [data.batch == b for b in range(len(data.ctrl))]
        xs = [self._onehot(data.pert[e]) for e in envs]
        ys = [data.y[e] for e in envs]
        counts = [data.weight[e] for e in envs]
        # set before fitting: checkpoint callbacks predict with it
        self.bridge_ = NeuralMarginalEmbeddingBridge(**self.bridge_kwargs)
        self.bridge_.fit(xs, ys, data.ctrl, counts, checkpoints, on_checkpoint)
        return self

    def predict(self, perturbations: list[str], test: sc.AnnData) -> np.ndarray:
        """Mean expression of each perturbation in `test`, given only `test`'s control cells."""
        x = self._onehot(_positions(self.perturbations_, perturbations))
        mix = _batch_mix(test, perturbations)
        controls = _test_controls(test)
        # prediction is linear in mu, so a batch mixture is the mix-weighted per-batch prediction;
        # a generator, so only one batch's prediction is held at a time
        return sum(mix[:, [b]] * self.bridge_.predict(x, w) for b, w in enumerate(controls))

    def _onehot(self, idx: np.ndarray) -> np.ndarray:
        """One-hot rows for perturbation indices `idx`."""
        return np.eye(len(self.perturbations_))[idx]
