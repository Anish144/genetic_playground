"""Kernel and neural transfer bridges for AnnData inputs.

Both bridges fit count-weighted means for each observed (perturbation, batch)
pair. Predictions use test controls and response counts, without test response
values. If a perturbation occurs in several test batches, its prediction is the
count-weighted mean of the batch predictions. predict_environment_average instead
uses a common distribution supplied with reference controls.

G counts perturbations, D counts source batches and J counts measured outputs.
Input preparation lives in data.transfer. LinearTransfer and Standardised live
in model.linear_transfer and utils.standardisation, respectively.
"""

from collections.abc import Mapping
from dataclasses import asdict

import anndata as ad
import numpy as np

from genetic_perturbation_playground.config import (
    HybridConfig,
    KernelConfig,
    PerXConfig,
    Shared2NNConfig,
)
from genetic_perturbation_playground.data.transfer import (
    BridgeData,
    batch_controls,
    batch_weights,
    bridge_data,
    label_positions,
    reference_weights,
)
from genetic_perturbation_playground.model.proxy_unconfounded import (
    MarginalEmbeddingBridge,
    mean_inner,
)
from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralHybridEmbeddingBridge,
    NeuralPerXEmbeddingBridge,
    NeuralSharedEmbeddingBridge,
)
from genetic_perturbation_playground.utils.kernels import RBF, median_heuristic
from genetic_perturbation_playground.utils.standardisation import Standardised


class KernelBridgeTransfer:
    """Fit a kernel response for each perturbation using its observed batches.

    The kernel compares batches by averaging similarities between their controls.
    All perturbations use the same batch similarities, but fit separate responses.
    Missing perturbation/batch pairs have no weight. Ridge scales by the total
    response weight across all perturbations.

    Parameters
    ----------
    reg : float, default=1e-6
        Ridge penalty before multiplying by the total training weight.
    weight_by_cells : bool, default=True
        Weight each observed pair by its cell count. False gives every pair weight one.
    lengthscale_scale : float, default=1.0
        Multiplier on the median-distance lengthscale fitted to training controls.

    Attributes
    ----------
    feature_names_in_ : numpy.ndarray of shape (J,)
        Training feature names, used to validate reference controls.
    perturbations_ : numpy.ndarray of shape (G,)
        Sorted perturbation labels seen during fitting.
    ctrl_ : list of numpy.ndarray
        Control samples for each source batch, in sorted batch order.
    kernel_w_ : RBF
        Control kernel with its fitted lengthscale.
    bridges_ : list of MarginalEmbeddingBridge
        One fitted response per perturbation, in perturbations_ order.

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> import pandas as pd
    >>> obs = pd.DataFrame(
    ...     {"perturbation": ["control", "a", "control", "a"],
    ...      "batch": ["b0", "b0", "b1", "b1"]}, index=["0", "1", "2", "3"])
    >>> data = ad.AnnData(np.array([[0.], [1.], [2.], [5.]]), obs=obs)
    >>> from genetic_perturbation_playground.model.batch_transfer import KernelBridgeTransfer
    >>> model = KernelBridgeTransfer(reg=0.01).fit(data)
    >>> model.predict(["a"], data).shape
    (1, 1)
    """

    def __init__(
        self,
        reg: float = KernelConfig.reg,
        weight_by_cells: bool = KernelConfig.weight_by_cells,
        lengthscale_scale: float = KernelConfig.lengthscale_scale,
    ):
        self.reg = reg
        self.weight_by_cells = weight_by_cells
        self.lengthscale_scale = lengthscale_scale

    def fit(self, adata: ad.AnnData) -> "KernelBridgeTransfer":
        """Fit one kernel bridge per observed perturbation.

        Parameters
        ----------
        adata : anndata.AnnData
            Cells by outputs, with perturbation and batch labels in obs.
            Controls use the perturbation label "control".

        Returns
        -------
        self : KernelBridgeTransfer
            Fitted model.
        """
        data = bridge_data(adata, self.weight_by_cells)
        self.perturbations_ = data.perturbations
        self.feature_names_in_ = adata.var_names.to_numpy(copy=True)
        self.ctrl_ = data.ctrl
        lengthscale = self.lengthscale_scale * median_heuristic(np.concatenate(self.ctrl_))
        self.kernel_w_ = RBF(lengthscales=lengthscale)
        inner = mean_inner(self.kernel_w_, self.ctrl_)
        self.bridges_ = [self._fit_block(data, inner, p) for p in range(len(self.perturbations_))]
        return self

    def predict(self, perturbations: list[str], test: ad.AnnData) -> np.ndarray:
        """Predict perturbation means from test controls and batch counts.

        Parameters
        ----------
        perturbations : list of str
            Labels present in training and test data, in the desired order.
        test : anndata.AnnData
            Test controls and perturbation/batch labels. Response values are unused.

        Returns
        -------
        numpy.ndarray of shape (len(perturbations), J)
            Predicted means, averaged over each perturbation's test-batch distribution.

        Raises
        ------
        ValueError
            If a label is missing or a response batch has no controls.
        """
        inner = mean_inner(self.kernel_w_, self.ctrl_, batch_controls(test))
        targets = batch_weights(test, perturbations) @ inner.T
        bridges = [self.bridges_[p] for p in label_positions(self.perturbations_, perturbations)]
        preds = [b.predict_embedded(np.zeros(1), t) for b, t in zip(bridges, targets, strict=True)]
        return np.concatenate(preds)

    def predict_environment_average(
        self,
        perturbations: list[str],
        reference: ad.AnnData,
        *,
        environment_weights: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Predict responses averaged over a common distribution of environments.

        Parameters
        ----------
        perturbations : list of str
            Nonempty fitted labels, in the desired order. Repeated labels are allowed.
        reference : anndata.AnnData
            Controls only, with batch labels and features in training order.
            Environments need not have appeared during fitting.
        environment_weights : mapping of str to float or None, default=None
            Probability of each reference batch. None weights batches equally,
            regardless of control counts. See data.transfer.reference_weights.

        Returns
        -------
        numpy.ndarray of shape (len(perturbations), J)
            Weighted response means in fitted target units. Invalid inputs or
            unknown perturbations raise ValueError.
        """
        if not len(perturbations):
            raise ValueError("perturbations must be nonempty.")
        positions = label_positions(self.perturbations_, perturbations)
        weights = reference_weights(
            reference, feature_names=self.feature_names_in_, environment_weights=environment_weights
        )
        controls = batch_controls(reference)
        target = sum(
            probability * mean_inner(self.kernel_w_, self.ctrl_, [w])[:, 0]
            for w, probability in zip(controls, weights, strict=True)
            if probability > 0
        )
        return np.concatenate(
            [self.bridges_[p].predict_embedded(np.zeros(1), target) for p in positions]
        )

    def _fit_block(self, data: BridgeData, inner: np.ndarray, p: int) -> MarginalEmbeddingBridge:
        """Fit perturbation p using the total weight across all perturbations.

        Rescale reg because the inner bridge multiplies it by its own weight.
        """
        rows = np.flatnonzero(data.pert == p)
        reg = self.reg * data.weight.sum() / data.weight[rows].sum()  # global N, as on cells
        bridge = MarginalEmbeddingBridge(RBF(), self.kernel_w_, reg)
        x = np.zeros(len(rows))  # constant within the block
        return bridge.fit_embedded(x, data.y[rows], data.batch[rows], inner, data.weight[rows])


class NeuralBridgeTransfer:
    """Adapt AnnData cells to a neural transfer bridge.

    Fit observed perturbation/batch means, using one-hot perturbation inputs and
    control samples from each batch. Predictions across several test batches use
    the perturbation's response-count weights.

    Parameters
    ----------
    weight_by_cells : bool, default=True
        Weight each observed pair by its cell count. False gives every pair weight one.
    variant : {"per_x", "shared_2nn", "hybrid"}, default="per_x"
        Neural model to fit. Shared 2NN adds a shared response;
        hybrid also adds one control slope per output.
    **bridge_kwargs : dict
        Settings passed to the selected neural bridge, such as reg, n_steps and device.

    Attributes
    ----------
    variant : str
        Selected model name.
    bridge_kwargs : dict
        Settings used to construct the neural bridge.
    feature_names_in_ : numpy.ndarray of shape (J,)
        Training feature names, used to validate reference controls.
    perturbations_ : numpy.ndarray of shape (G,)
        Sorted perturbation labels seen during fitting.
    bridge_ : NeuralPerXEmbeddingBridge or subclass
        Fitted neural bridge. Set before training so checkpoint callbacks can predict.

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> import pandas as pd
    >>> obs = pd.DataFrame(
    ...     {"perturbation": ["control", "a", "control", "a"],
    ...      "batch": ["b0", "b0", "b1", "b1"]}, index=["0", "1", "2", "3"])
    >>> data = ad.AnnData(np.array([[0.], [1.], [2.], [5.]]), obs=obs)
    >>> from genetic_perturbation_playground.model.batch_transfer import NeuralBridgeTransfer
    >>> model = NeuralBridgeTransfer(n_steps=1, n_features_w=2, width_w=4).fit(data)
    >>> model.predict(["a"], data).shape
    (1, 1)
    """

    def __init__(
        self,
        weight_by_cells: bool = PerXConfig.weight_by_cells,
        *,
        variant: str = PerXConfig.variant,
        **bridge_kwargs,
    ):
        bridges = {
            "per_x": NeuralPerXEmbeddingBridge,
            "shared_2nn": NeuralSharedEmbeddingBridge,
            "hybrid": NeuralHybridEmbeddingBridge,
        }
        if variant not in bridges:
            raise ValueError(f"Unknown neural variant {variant!r}; choose from {tuple(bridges)}")
        self.weight_by_cells = weight_by_cells
        self.variant = variant
        self._bridge_type = bridges[variant]
        self.bridge_kwargs = bridge_kwargs

    def fit(
        self, adata: ad.AnnData, checkpoints=(), on_checkpoint=None, on_training_loss=None
    ) -> "NeuralBridgeTransfer":
        """Prepare observed pairs and fit the selected neural bridge.

        Parameters
        ----------
        adata : anndata.AnnData
            Cells by outputs, with perturbation and batch labels in obs.
            Controls use the perturbation label "control".
        checkpoints : collection of int, default=()
            Completed update counts at which to fit coefficients and call on_checkpoint.
        on_checkpoint : callable or None, default=None
            Callback on_checkpoint(step); required when checkpoints is nonempty.
        on_training_loss : callable or None, default=None
            Callback on_training_loss(step, mse), including the final training loss.

        Returns
        -------
        self : NeuralBridgeTransfer
            Fitted model. Callbacks follow NeuralPerXEmbeddingBridge.fit.
        """
        data = bridge_data(adata, self.weight_by_cells)
        self.perturbations_ = data.perturbations
        self.feature_names_in_ = adata.var_names.to_numpy(copy=True)
        envs = [data.batch == b for b in range(len(data.ctrl))]
        xs = [self._onehot(data.pert[e]) for e in envs]
        ys = [data.y[e] for e in envs]
        counts = [data.weight[e] for e in envs]
        # set before fitting: checkpoint callbacks predict with it
        self.bridge_ = self._bridge_type(**self.bridge_kwargs)
        self.bridge_.fit(xs, ys, data.ctrl, counts, checkpoints, on_checkpoint, on_training_loss)
        return self

    def predict(self, perturbations: list[str], test: ad.AnnData) -> np.ndarray:
        """Predict perturbation means from test controls and batch counts.

        Parameters
        ----------
        perturbations : list of str
            Labels present in training and test data, in the desired order.
        test : anndata.AnnData
            Test controls and perturbation/batch labels. Response values are unused.

        Returns
        -------
        numpy.ndarray of shape (len(perturbations), J)
            Predicted means, averaged over each perturbation's test-batch distribution.

        Raises
        ------
        ValueError
            If a label is missing or a response batch has no controls.
        """
        x = self._onehot(label_positions(self.perturbations_, perturbations))
        mix = batch_weights(test, perturbations)
        controls = batch_controls(test)
        # prediction is linear in mu, so a batch mixture is the mix-weighted per-batch prediction;
        # a generator, so only one batch's prediction is held at a time
        return sum(mix[:, [b]] * self.bridge_.predict(x, w) for b, w in enumerate(controls))

    def predict_environment_average(
        self,
        perturbations: list[str],
        reference: ad.AnnData,
        *,
        environment_weights: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Predict responses averaged over a common distribution of environments.

        Parameters
        ----------
        perturbations : list of str
            Nonempty fitted labels, in the desired order. Repeated labels are allowed.
        reference : anndata.AnnData
            Controls only, with batch labels and features in training order.
            Environments need not have appeared during fitting.
        environment_weights : mapping of str to float or None, default=None
            Probability of each reference batch. None weights batches equally,
            regardless of control counts. See data.transfer.reference_weights.

        Returns
        -------
        numpy.ndarray of shape (len(perturbations), J)
            Weighted response means in fitted target units. Invalid inputs or
            unknown perturbations raise ValueError.
        """
        if not len(perturbations):
            raise ValueError("perturbations must be nonempty.")
        positions = label_positions(self.perturbations_, perturbations)
        weights = reference_weights(
            reference, feature_names=self.feature_names_in_, environment_weights=environment_weights
        )
        controls = batch_controls(reference)
        return self.bridge_.predict_mixture(self._onehot(positions), controls, weights.to_numpy())

    def _onehot(self, idx: np.ndarray) -> np.ndarray:
        """Encode perturbation indices as (n, G) one-hot rows."""
        encoded = np.zeros((len(idx), len(self.perturbations_)))
        encoded[np.arange(len(idx)), idx] = 1.0
        return encoded


def build_kernel(config: KernelConfig, *, standardise_w: bool = True) -> Standardised:
    """Construct a kernel bridge with training-data standardisation.

    Parameters
    ----------
    config : KernelConfig
        Model settings. Construction does not change them.
    standardise_w : bool, default=True
        Scale controls; targets are centred in either case.

    Returns
    -------
    Standardised
        Unfitted kernel bridge and its scaling wrapper.
    """
    return Standardised(KernelBridgeTransfer(**asdict(config)), w=standardise_w)


def build_neural(config: PerXConfig, *, standardise_w: bool = True) -> Standardised:
    """Construct the neural variant described by a settings dataclass.

    Parameters
    ----------
    config : PerXConfig, Shared2NNConfig or HybridConfig
        Model settings. Its variant must match its class.
    standardise_w : bool, default=True
        Scale controls; targets are centred in either case.

    Returns
    -------
    Standardised
        Unfitted neural bridge and its scaling wrapper.
    """
    if type(config) not in (PerXConfig, Shared2NNConfig, HybridConfig):
        raise TypeError("Expected PerXConfig, Shared2NNConfig or HybridConfig")
    if config.variant != type(config).variant:
        raise ValueError(f"Select the matching config class or Hydra group neural={config.variant}")
    return Standardised(NeuralBridgeTransfer(**asdict(config)), w=standardise_w)
