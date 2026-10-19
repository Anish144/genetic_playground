import numpy as np
import scanpy as sc
from scipy import sparse

from genetic_perturbation_playground.model.proxy_unconfounded import (
    MarginalEmbeddingBridge,
    mean_inner,
)
from genetic_perturbation_playground.utils.kernels import RBF, median_heuristic


def _onehot(idx: np.ndarray, n_groups: int) -> sparse.csr_matrix:
    """Sparse (n_groups x n_cells) indicator matrix."""
    return sparse.csr_matrix(
        (np.ones(len(idx)), (idx, np.arange(len(idx)))), shape=(n_groups, len(idx))
    )


def _dense(m) -> np.ndarray:
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


def summary_stats(adata: sc.AnnData) -> dict:
    """Per-batch control means W_e and per-perturbation sufficient statistics.

    Returns perturbations, n (cells per perturbation), y_mean (mean of y over the perturbation's
    cells), w_mean (W_e averaged over the same cells, i.e. weighted by the perturbation's batch
    distribution), plus per-batch sums used to fit the slope.
    """
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    batch_labels = adata.obs["batch"].astype(str).to_numpy()
    is_ctrl = labels == "control"

    batches, ctrl_b = np.unique(batch_labels[is_ctrl], return_inverse=True)
    ctrl_onehot = _onehot(ctrl_b, len(batches))
    W = _dense(ctrl_onehot @ adata.X[is_ctrl]) / np.asarray(
        ctrl_onehot.sum(axis=1)
    )  # (batches, genes)

    if not np.isin(batch_labels[~is_ctrl], batches).all():
        raise ValueError("Every batch with perturbed cells needs control cells to define W_e.")
    perts, p_idx = np.unique(labels[~is_ctrl], return_inverse=True)
    b_idx = np.searchsorted(batches, batch_labels[~is_ctrl])
    P, B = _onehot(p_idx, len(perts)), _onehot(b_idx, len(batches))
    Y = adata.X[~is_ctrl]

    n = np.asarray(P.sum(axis=1)).ravel()
    pert_batch_n = _dense(P @ B.T)  # (perts, batches) cell counts
    return {
        "perturbations": perts,
        "n": n,
        "y_mean": _dense(P @ Y) / n[:, None],
        "w_mean": pert_batch_n @ W / n[:, None],
        "batches": batches,
        "pert_batch_n": pert_batch_n,
        "W": W,
        "batch_n": np.asarray(B.sum(axis=1)).ravel(),
        "batch_y_sum": _dense(B @ Y),
    }


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
        y, w, n = s["y_mean"], s["w_mean"], s["n"][:, None]
        if self.slope is None:
            # Within-perturbation regression of y on W (Frisch-Waugh), summed over cells.
            num = (s["W"] * s["batch_y_sum"]).sum(axis=0) - (n * y * w).sum(axis=0)
            den = (s["batch_n"][:, None] * s["W"] ** 2).sum(axis=0) - (n * w**2).sum(axis=0)
            self.slope_ = num / den
        else:
            self.slope_ = np.full(y.shape[1], float(self.slope))
        self.perturbations_ = s["perturbations"]
        self.gamma_ = y - self.slope_ * w
        return self

    def predict(self, perturbations: list[str], w: np.ndarray) -> np.ndarray:
        """Mean expression of each perturbation in an environment with control mean `w`.

        `w` is (n_genes,) for one batch, or (n_perts, n_genes) with one row per perturbation.
        """
        return self.gamma_[np.searchsorted(self.perturbations_, perturbations)] + self.slope_ * w


class KernelBridgeTransfer:
    """MarginalEmbeddingBridge on cells, with a delta kernel on the perturbation x.

    w is a control cell, so mu_e is the mean embedding of batch e's control cells. The delta kernel
    makes the cell-level Gram block-diagonal over perturbations, so the one cell-level bridge splits
    exactly into one bridge per perturbation over the batches it was seen in (x constant within a
    block), fit on its per-batch means. All blocks share M = <mu_e, mu_e'>, computed once.
    """

    def __init__(self, reg: float = 1e-6):
        self.reg = reg

    def fit(self, adata: sc.AnnData) -> "KernelBridgeTransfer":
        s = summary_stats(adata)
        labels = adata.obs["perturbation"].astype(str).to_numpy()
        b_idx = np.searchsorted(s["batches"], adata.obs["batch"].astype(str).to_numpy())
        is_ctrl = labels == "control"

        self.ctrl_ = _per_batch_controls(adata, is_ctrl, b_idx, len(s["batches"]))
        self.kernel_w_ = RBF(lengthscales=median_heuristic(np.concatenate(self.ctrl_)))
        inner = mean_inner(self.kernel_w_, self.ctrl_)

        # per-(perturbation, batch) means of y; a block's ridge uses the global N as one bridge would
        p_idx = np.searchsorted(s["perturbations"], labels[~is_ctrl])
        pair = p_idx * len(s["batches"]) + b_idx[~is_ctrl]
        pairs, pair_idx = np.unique(pair, return_inverse=True)
        n_pe = np.bincount(pair_idx)
        y_pe = _dense(_onehot(pair_idx, len(pairs)) @ adata.X[~is_ctrl]) / n_pe[:, None]
        n_total = n_pe.sum()

        self.perturbations_, self.bridges_ = s["perturbations"], []
        for p in range(len(s["perturbations"])):
            rows = np.flatnonzero(pairs // len(s["batches"]) == p)
            bridge = MarginalEmbeddingBridge(
                RBF(), self.kernel_w_, self.reg * n_total / n_pe[rows].sum()
            )
            env = pairs[rows] % len(s["batches"])
            self.bridges_.append(
                bridge.fit_embedded(np.zeros(len(rows)), y_pe[rows], env, inner, n_pe[rows])
            )
        return self

    def predict(self, perturbations: list[str], test: sc.AnnData) -> np.ndarray:
        """Mean expression of each perturbation in `test`, given only `test`'s control cells.

        If `test` pools several batches, each perturbation's target embedding is the mixture of
        their mu_b weighted by where its cells fall, matching summary_stats' w_mean.
        """
        st = summary_stats(test)
        is_ctrl = test.obs["perturbation"].astype(str).to_numpy() == "control"
        b_idx = np.searchsorted(st["batches"], test.obs["batch"].astype(str).to_numpy())
        ctrl = _per_batch_controls(test, is_ctrl, b_idx, len(st["batches"]))
        mix = st["pert_batch_n"] / st["n"][:, None]  # (test perts, test batches)
        targets = (
            mix[np.searchsorted(st["perturbations"], perturbations)]
            @ mean_inner(self.kernel_w_, self.ctrl_, ctrl).T
        )
        rows = np.searchsorted(self.perturbations_, perturbations)
        return np.concatenate(
            [
                self.bridges_[p].predict_embedded(np.zeros(1), t)
                for p, t in zip(rows, targets, strict=True)
            ]
        )


def _per_batch_controls(adata, is_ctrl, b_idx, n_batches) -> list[np.ndarray]:
    return [_dense(adata.X[is_ctrl & (b_idx == b)]) for b in range(n_batches)]
