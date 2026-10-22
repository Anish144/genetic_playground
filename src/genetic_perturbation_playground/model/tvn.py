"""
TVN transfer: Recursion's typical variation normalisation, used as an out-of-batch predictor. TVN
is the standard batch correction for Phenom embeddings, so it is the reference the bridges must beat.

Terms: a batch e is one experiment; control cells are the proxy cells (is_control); y is one cell's
embedding, a row vector; "standardise" means subtract the mean and divide by the std, per column.

TVN (tvn_on_controls in recursionpharma/EFAAR_benchmarking) maps each cell y of batch e to z:

    1. standardise y with the mean and std of all control cells
    2. rotate onto the principal axes (PCA, all components) of all control cells
    3. standardise with the mean and std of batch e's control cells
    4. CORAL: right-multiply by S_e^(-1/2) T^(1/2), where, after step 3,
           S_e = covariance of batch e's control cells + ridge * I
           T   = covariance of all control cells + ridge * I

so every batch's control cells end with the same covariance T, and z is comparable across batches.

Transfer. Each step is affine and invertible, and only steps 3-4 depend on the batch:

    fit:      steps 1-2 and T from the training control cells;
              profile_x = mean of z over perturbation x's cells, all training batches pooled
    predict:  steps 3-4 from the held-out batch's control cells; map profile_x back through
              steps 4, 3, 2, 1. Affine maps commute with means, so this is x's mean embedding there.

ridge = 0.5 (Recursion's value) keeps S_e invertible when a batch has fewer control cells than
features; TVN still predicts poorly in that regime (tests/test_tvn.py).
"""

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from genetic_perturbation_playground.model.batch_transfer import (
    _dense,
    _group_means,
    _positions,
    batch_labels,
    is_control,
    perturbation_labels,
)


class TVNTransferModel:
    """Recursion's TVN, fit on the training batches and inverted into a held-out one (module doc)."""

    def __init__(self, ridge: float = 0.5):
        self.ridge = ridge

    def fit(self, adata: sc.AnnData) -> "TVNTransferModel":
        X = _dense(adata.X)
        ctrl = is_control(adata)
        batch = batch_labels(adata)
        self.scaler_ = StandardScaler().fit(X[ctrl])
        self.pca_ = PCA().fit(self.scaler_.transform(X[ctrl]))
        pcs = self._principal_axes(X)
        for b in np.unique(batch):
            in_b = batch == b
            pcs[in_b] = StandardScaler().fit(pcs[in_b & ctrl]).transform(pcs[in_b])
        self.target_ = self._ridged_cov(pcs[ctrl])
        for b in np.unique(batch):
            in_b = batch == b
            pcs[in_b] = pcs[in_b] @ self._coral(pcs[in_b & ctrl])
        perts = perturbation_labels(adata)[~ctrl]
        self.perturbations_, p_idx = np.unique(perts, return_inverse=True)
        self.profiles_ = _group_means(p_idx, len(self.perturbations_), pcs[~ctrl])
        return self

    def predict(self, perturbations: list[str], test: sc.AnnData) -> np.ndarray:
        """Mean of each perturbation in the one batch of `test`, given only its control cells."""
        if len(np.unique(batch_labels(test))) > 1:
            raise ValueError("TVN predicts one batch at a time; test holds several")
        controls = self._principal_axes(_dense(test.X)[is_control(test)])
        batch_scaler = StandardScaler().fit(controls)
        coral = self._coral(batch_scaler.transform(controls))
        profiles = self.profiles_[_positions(self.perturbations_, perturbations)]
        pcs = batch_scaler.inverse_transform(profiles @ np.linalg.inv(coral))
        return self.scaler_.inverse_transform(self.pca_.inverse_transform(pcs))

    def _principal_axes(self, X: np.ndarray) -> np.ndarray:
        """Steps 1-2: X scaled on all training controls, in their principal axes."""
        return self.pca_.transform(self.scaler_.transform(X))

    def _coral(self, controls: np.ndarray) -> np.ndarray:
        """Step 4's map S^(-1/2) T^(1/2), with S from one batch's (step 3) control cells."""
        return _matrix_power(self._ridged_cov(controls), -0.5) @ _matrix_power(self.target_, 0.5)

    def _ridged_cov(self, X: np.ndarray) -> np.ndarray:
        """Covariance of the rows of X plus ridge * I."""
        return np.cov(X, rowvar=False, ddof=1) + self.ridge * np.eye(X.shape[1])


def _matrix_power(C: np.ndarray, p: float) -> np.ndarray:
    """C^p of a symmetric positive-definite C (scipy's fractional_matrix_power, kept real)."""
    vals, vecs = np.linalg.eigh(C)
    return (vecs * vals**p) @ vecs.T
