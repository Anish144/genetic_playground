import numpy as np
import scanpy as sc

from .batch_transfer import is_control, summary_stats


class LinearPerturbationModel:
    """Cell-level linear regression: y_cell = b + onehot(perturbation) @ beta.

    The intercept b is the control mean, so each row of beta is the average effect of a
    perturbation relative to control. With a one-hot design the least-squares solution is
    closed-form: beta_p = mean(y over cells of p) - b, the mean shift from control.
    """

    def fit(self, adata: sc.AnnData) -> "LinearPerturbationModel":
        self.intercept_ = np.asarray(adata.X[is_control(adata)].mean(axis=0)).ravel()
        s = summary_stats(adata)
        self.perturbations_, self.coef_ = s["perturbations"], s["y_mean"] - self.intercept_
        return self

    def predict(self, perturbations: list[str]) -> np.ndarray:
        """Average effect (vs control) for each perturbation, shape (n_perts, n_genes)."""
        return self.coef_[np.searchsorted(self.perturbations_, perturbations)]
