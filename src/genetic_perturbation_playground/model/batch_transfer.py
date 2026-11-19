"""Kernel and neural transfer bridges for AnnData inputs.

Both bridges fit count-weighted means for each observed (perturbation, batch)
pair. Predictions use test controls and response counts, without test response
values. If a perturbation occurs in several test batches, its prediction is the
count-weighted mean of the batch predictions.

G counts perturbations, D counts source batches and J counts measured outputs.
Input preparation lives in data.transfer. LinearTransfer and Standardised live
in model.linear_transfer and utils.standardisation, respectively.
"""

import anndata as ad
import numpy as np

from genetic_perturbation_playground.data.transfer import (
    BridgeData,
    batch_controls,
    batch_weights,
    bridge_data,
    label_positions,
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
        self, reg: float = 1e-6, weight_by_cells: bool = True, lengthscale_scale: float = 1.0
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
        weight_by_cells: bool = True,
        *,
        variant: str = "per_x",
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

    def _onehot(self, idx: np.ndarray) -> np.ndarray:
        """Encode perturbation indices as (n, G) one-hot rows."""
        encoded = np.zeros((len(idx), len(self.perturbations_)))
        encoded[np.arange(len(idx)), idx] = 1.0
        return encoded
