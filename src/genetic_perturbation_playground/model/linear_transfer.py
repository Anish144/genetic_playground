"""Linear transfer from batch control means to perturbation responses.

G is the number of perturbations, J the number of measured outputs, and N the
number of training response cells. Control samples set the input scaling;
LinearRidgeSolver fits one control slope per output.
"""

import anndata as ad
import numpy as np

from genetic_perturbation_playground.config import LinearConfig
from genetic_perturbation_playground.data.transfer import (
    as_dense,
    is_control,
    label_positions,
    matched_control_mean,
    summary_stats,
)
from genetic_perturbation_playground.utils.ridge import LinearRidgeSolver
from genetic_perturbation_playground.utils.standardisation import mean_and_std


class LinearTransfer:
    """Predict perturbation responses from a new batch's mean controls.

    Each output uses its matching batch control mean W[e], scaled when
    standardise_w=True:

        y_hat[x, e] = intercept[x] + slope * W[e]

    A fitted slope minimises squared errors plus N * reg * ||slope||^2, where N
    counts training response cells. Intercepts have no penalty. Targets and
    predictions stay in their original units.

    Parameters
    ----------
    slope : float or None, default=None
        None fits the slope with ridge. A number fixes the slope in original
        input units: 0 gives pooled means; 1 shifts responses by the control mean.
    standardise_w : bool, default=True
        Scale controls using their pooled training mean and standard deviation.
        The ridge penalty then applies to slopes in these standardised units.
    reg : float, default=1e-3
        Finite, nonnegative slope penalty. Zero gives ordinary least squares.
        Ignored for fixed slopes.

    Attributes
    ----------
    perturbations_ : numpy.ndarray of shape (G,)
        Sorted perturbation labels seen during fitting.
    coef_ : numpy.ndarray of shape (J,)
        Slopes in the control units used for fitting.
    intercept_ : numpy.ndarray of shape (G, J)
        Category intercepts for controls in the fitting units.
    slope_ : numpy.ndarray of shape (J,)
        Slopes converted to original control units.
    gamma_ : numpy.ndarray of shape (G, J)
        Category intercepts for controls in original units.
    w_mean_ : numpy.ndarray of shape (J,) or float
        Training control means; zero when scaling is disabled.
    w_std_ : numpy.ndarray of shape (J,) or float
        Training control standard deviations; one when scaling is disabled.

    Examples
    --------
    Fit one perturbation in two batches. Ridge shrinks the unpenalised slope of
    2 to 1, and the intercept changes to preserve the training mean.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import anndata as ad
    >>> from genetic_perturbation_playground.model.linear_transfer import LinearTransfer
    >>> obs = pd.DataFrame(
    ...     {"perturbation": ["control", "a", "control", "a"],
    ...      "batch": ["b0", "b0", "b1", "b1"]},
    ...     index=["0", "1", "2", "3"],
    ... )
    >>> data = ad.AnnData(np.array([[0.], [1.], [2.], [5.]]), obs=obs)
    >>> model = LinearTransfer(standardise_w=False, reg=1.0).fit(data)
    >>> model.slope_.tolist()
    [1.0]
    >>> model.gamma_.tolist()
    [[2.0]]
    """

    def __init__(
        self,
        slope: float | None = LinearConfig.slope,
        standardise_w: bool = LinearConfig.standardise_w,
        *,
        reg: float = LinearConfig.reg,
    ):
        self.slope = slope
        self.standardise_w = standardise_w
        self.reg = reg

    def fit(self, adata: ad.AnnData) -> "LinearTransfer":
        """Fit slopes and intercepts using training cells.

        Parameters
        ----------
        adata : anndata.AnnData
            Cells by outputs, with perturbation and batch labels in ``obs``.
            Control cells have perturbation label "control".

        Returns
        -------
        self : LinearTransfer
            Fitted model.
        """
        s = summary_stats(adata)
        self.w_mean_, self.w_std_ = (
            mean_and_std(as_dense(adata.X[is_control(adata)])) if self.standardise_w else (0.0, 1.0)
        )
        s["W"] = (s["W"] - self.w_mean_) / self.w_std_
        s["w_mean"] = (s["w_mean"] - self.w_mean_) / self.w_std_
        self.coef_ = self._coefficient(s)
        self.perturbations_ = s["perturbations"]
        self.intercept_ = s["y_mean"] - self.coef_ * s["w_mean"]
        self.slope_ = self.coef_ / self.w_std_
        self.gamma_ = self.intercept_ - self.slope_ * self.w_mean_
        return self

    def predict(self, perturbations: list[str], test: ad.AnnData) -> np.ndarray:
        """Predict means using test controls and each perturbation's batch counts.

        Parameters
        ----------
        perturbations : list of str
            Labels present in training and test data, in the desired order.
        test : anndata.AnnData
            Test cells with perturbation and batch labels. Response values are unused.

        Returns
        -------
        numpy.ndarray of shape (len(perturbations), J)
            Predicted means in the original target units.

        Raises
        ------
        ValueError
            If a label is missing or a response batch has no controls.
        """
        intercept = self.intercept_[label_positions(self.perturbations_, perturbations)]
        w = (matched_control_mean(test, perturbations) - self.w_mean_) / self.w_std_
        return intercept + self.coef_ * w

    def _coefficient(self, s: dict) -> np.ndarray:
        """Fit ridge slopes or convert a fixed slope to scaled control units."""
        if self.slope is None:
            return LinearRidgeSolver(reg=self.reg).solve(s)
        return np.full(s["y_mean"].shape[1], float(self.slope)) * self.w_std_
