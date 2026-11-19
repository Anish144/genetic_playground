"""Scale transfer-model controls and centre targets using training statistics.

G counts perturbations and J counts measured outputs. Every scaling statistic
comes from the training data and is reused when predicting new environments.
"""

import anndata as ad
import numpy as np

from genetic_perturbation_playground.data.transfer import as_dense, is_control


class Standardised:
    """Fit and predict with standardised control inputs and centred targets.

    For each coordinate, transform controls W and response targets Y as

        W_scaled = (W - w_mean_) / w_std_
        Y_centred = Y - y_mean_

    Use training statistics at prediction time and add y_mean_ back to outputs.
    Ridge then shrinks predictions toward the training target mean. Constant
    control coordinates use a scale of one.

    Parameters
    ----------
    model : object
        Transfer model with fit(adata), predict(perturbations, adata) and a
        fitted perturbations_ attribute. This wrapper fits the supplied instance.
    w : bool, default=True
        Whether to scale controls. Targets are centred in either case.

    Attributes
    ----------
    model : object
        Wrapped model, fitted on the transformed training data.
    w : bool
        Whether to scale control inputs.
    perturbations_ : numpy.ndarray of shape (G,)
        Perturbation labels exposed by the wrapped model.
    w_mean_ : numpy.ndarray of shape (J,) or float
        Mean training controls, or zero when w=False.
    w_std_ : numpy.ndarray of shape (J,) or float
        Training control scales, or one when w=False.
    y_mean_ : numpy.ndarray of shape (J,)
        Mean training responses, added back to predictions.

    Examples
    --------
    Centre responses while reusing the training control scale at prediction.

    >>> import anndata as ad
    >>> import numpy as np
    >>> import pandas as pd
    >>> from genetic_perturbation_playground.model.linear_transfer import LinearTransfer
    >>> from genetic_perturbation_playground.utils.standardisation import Standardised
    >>> obs = pd.DataFrame(
    ...     {"perturbation": ["control", "a", "control", "a"],
    ...      "batch": ["b0", "b0", "b1", "b1"]}, index=["0", "1", "2", "3"])
    >>> data = ad.AnnData(np.array([[1.], [4.], [3.], [6.]]), obs=obs)
    >>> model = Standardised(LinearTransfer(slope=0, standardise_w=False)).fit(data)
    >>> model.w_mean_.tolist()
    [2.0]
    >>> model.predict(["a"], data).tolist()
    [[5.0]]
    """

    def __init__(self, model, w: bool = True):
        self.model = model
        self.w = w

    @property
    def perturbations_(self) -> np.ndarray:
        """Return the fitted perturbation labels from the wrapped model."""
        return self.model.perturbations_

    def fit(self, adata: ad.AnnData, *args, **kwargs) -> "Standardised":
        """Learn scaling statistics and fit the wrapped model on transformed rows.

        Parameters
        ----------
        adata : anndata.AnnData
            Training cells with perturbation and batch labels in obs.
        *args : tuple
            Extra positional arguments forwarded to the model's fit method.
        **kwargs : dict
            Extra fit arguments, such as neural checkpoint callbacks.

        Returns
        -------
        self : Standardised
            Fitted wrapper and model.
        """
        ctrl = is_control(adata)
        self.w_mean_, self.w_std_ = mean_and_std(as_dense(adata.X[ctrl])) if self.w else (0.0, 1.0)
        self.y_mean_ = np.asarray(adata.X[~ctrl].mean(axis=0)).ravel()
        self.model.fit(self._standardise(adata), *args, **kwargs)
        return self

    def predict(self, perturbations: list[str], test: ad.AnnData) -> np.ndarray:
        """Predict with training scales and restore the target mean.

        Parameters
        ----------
        perturbations : list of str
            Training perturbations to predict, in the desired order.
        test : anndata.AnnData
            Test cells with controls and perturbation/batch labels.

        Returns
        -------
        numpy.ndarray of shape (len(perturbations), J)
            Predictions in the original target units.
        """
        return self.model.predict(perturbations, self._standardise(test)) + self.y_mean_

    def _standardise(self, adata: ad.AnnData) -> ad.AnnData:
        """Copy X, scale controls and centre responses using training statistics.

        Return dense values and obs labels; other AnnData annotations are
        omitted. The input values are not changed.
        """
        X = as_dense(adata.X).astype(float)
        ctrl = is_control(adata)
        X[ctrl] = (X[ctrl] - self.w_mean_) / self.w_std_
        X[~ctrl] -= self.y_mean_
        return ad.AnnData(X, obs=adata.obs)


def mean_and_std(X) -> tuple[np.ndarray, np.ndarray]:
    """Calculate column means and population standard deviations.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, J)
        Training control values.

    Returns
    -------
    mean : numpy.ndarray of shape (J,)
        Mean of each column.
    scale : numpy.ndarray of shape (J,)
        Population standard deviations, with one for constant columns.
    """
    X = np.asarray(X)
    std = X.std(axis=0)
    return X.mean(axis=0), np.where(std > 0, std, 1.0)
