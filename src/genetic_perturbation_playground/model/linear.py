import numpy as np
import scanpy as sc

from .batch_transfer import _dense, _onehot


class LinearPerturbationModel:
    """Cell-level linear regression: y_cell = b + onehot(perturbation) @ beta.

    The intercept b is the control mean, so each row of beta is the average effect of a
    perturbation relative to control. With a one-hot design the least-squares solution is
    closed-form: beta_p = mean(y over cells of p) - b, the mean shift from control.
    """

    def fit(self, adata: sc.AnnData) -> "LinearPerturbationModel":
        labels = adata.obs["perturbation"].astype(str).to_numpy()
        is_ctrl = labels == "control"
        self.intercept_ = np.asarray(adata.X[is_ctrl].mean(axis=0)).ravel()

        self.perturbations_, idx = np.unique(labels[~is_ctrl], return_inverse=True)
        onehot = _onehot(idx, len(self.perturbations_))  # perts x cells
        counts = np.asarray(onehot.sum(axis=1)).ravel()
        sums = _dense(onehot @ adata.X[~is_ctrl])
        self.coef_ = sums / counts[:, None] - self.intercept_
        return self

    def predict(self, perturbations: list[str]) -> np.ndarray:
        """Average effect (vs control) for each perturbation, shape (n_perts, n_genes)."""
        return self.coef_[np.searchsorted(self.perturbations_, perturbations)]
