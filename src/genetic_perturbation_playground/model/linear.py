import numpy as np
import scanpy as sc
from scipy import sparse


class LinearPerturbationModel:
    """Cell-level ridge regression: y_cell = b + onehot(perturbation) @ beta.

    The intercept b is the control mean, so each row of beta is the average effect of a
    perturbation relative to control. With a one-hot design the ridge solution is closed-form:
    beta_p = sum_{cells in p}(y - b) / (n_p + alpha), i.e. the mean shift shrunk towards 0.
    """

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def fit(self, adata: sc.AnnData) -> "LinearPerturbationModel":
        labels = adata.obs["perturbation"].astype(str).to_numpy()
        is_ctrl = labels == "control"
        self.intercept_ = np.asarray(adata.X[is_ctrl].mean(axis=0)).ravel()

        self.perturbations_, idx = np.unique(labels[~is_ctrl], return_inverse=True)
        onehot = sparse.csr_matrix((np.ones(len(idx)), (idx, np.arange(len(idx)))))  # perts x cells
        counts = np.asarray(onehot.sum(axis=1)).ravel()
        sums = onehot @ adata.X[~is_ctrl]
        sums = sums.toarray() if sparse.issparse(sums) else np.asarray(sums)
        self.coef_ = (sums - np.outer(counts, self.intercept_)) / (counts + self.alpha)[:, None]
        return self

    def predict(self, perturbations: list[str]) -> np.ndarray:
        """Average effect (vs control) for each perturbation, shape (n_perts, n_genes)."""
        return self.coef_[np.searchsorted(self.perturbations_, perturbations)]
