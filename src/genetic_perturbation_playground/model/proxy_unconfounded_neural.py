"""Predict perturbation responses from an environment's control samples.

Notation
--------
G : number of categories
    Distinct perturbations, such as gene knockdowns or siRNAs.
D : number of environments
    Source batches or experiments.
q : number of learned features
    Output size of each control network, set by n_features_w.
J : number of outputs
    Measured genes or image-embedding coordinates predicted by the model.
n_e : number of response rows in environment e
    Each row contains a category and its target vector. Counts can weight the row.
m_e : number of control samples in environment e
    Controls need not be paired with response rows.
d_w : number of input coordinates per control sample
r : number of retained control directions
    Numerical rank of the training control means, at most min(D, d_w).
    Zero when the hybrid's linear term is disabled.
B : control-to-output matrix
    Shape (d_w, J), fitted jointly with the hybrid model's neural coefficients.
N : total training weight
    Sum of response counts across all categories in the current training split.
Q : target-product matrix
    target_gram has shape (G, D, D): Q[x] = cy[x] @ cy[x].T / J.
    Uppercase Q is a matrix; lowercase q is a feature count.

Networks map controls to features, averaged within each environment.
NeuralRidgeSolver fits the response coefficients with penalties scaled by N.
Cross-fitted network training uses held-out prediction mean squared error (MSE):

    SSE = sum_i count[i] * sum_j (prediction[i, j] - target[i, j])**2 / J
    MSE = SSE / sum_i count[i]

SSE means sum of squared errors, weighted by counts and averaged over outputs.
Cross-fitting fits coefficients on some environments and scores the others.
In-sample training adds the ridge penalty, divided by J and by the total count,
so the networks minimise the same objective as the coefficients.
In the equations, @ is matrix multiplication and ||A||^2 sums squared entries.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from torch import nn
from tqdm import trange

from genetic_perturbation_playground.config import HybridConfig, PerXConfig, Shared2NNConfig
from genetic_perturbation_playground.data.neural import GroupedRidgeData
from genetic_perturbation_playground.utils.neural import as_tensor, checkpointed_sse, control_mlp
from genetic_perturbation_playground.utils.ridge import GramRidgeSolver, NeuralRidgeSolver


@dataclass(eq=False)
class NeuralPerXEmbeddingBridge:
    """Fit a separate response to learned control features for each category.

    Average learned features over controls W[e] in environment e:

        y_hat[x, e] = mean(psi(W[e])) @ A[x]

    For fixed features, fit A by minimising

        sum_i count[i] * ||prediction[i] - target[i]||^2
        + N * reg * sum_x ||A[x]||^2

    All categories share psi. Missing pairs have zero weight. Networks use
    float32 and ridge uses float64. In-sample training includes the ridge penalty.

    Parameters
    ----------
    n_features_w : int, default=16
        Number q of learned features per control network.
    width_w : int, default=64
        Number of units in each hidden layer.
    depth : int, default=1
        Number of SwiGLU hidden layers. Zero gives a linear network.
    reg : float, default=1e-3
        Coefficient penalty before multiplying by the total training count N.
    lr : float, default=1e-3
        AdamW learning rate for the control network.
    weight_decay : float, default=0.0
        AdamW penalty on network weights, separate from coefficient ridge.
    n_steps : int, default=2000
        Number of network updates, each using all training environments.
    cross_fit_folds : int, default=0
        Zero fits and scores coefficients on the same environments. A value of
        at least 2 draws new balanced folds each step and scores held-out environments.
    seed : int, default=0
        Seed for network initialisation and fold draws.
    device : str, default="cpu"
        PyTorch device, such as "cpu" or "cuda". MPS does not support float64.
    log_key : str or None, default=None
        Prefix for training-loss logs in an existing W&B run. None disables logging.
    gene_block_size : int, default=256
        Number of outputs to process at once when computing final coefficients.
        Limits temporary memory; all final coefficients stay on the model's device.

    Attributes
    ----------
    n_features_in_ : int
        Number of control coordinates seen during fitting.
    psi_ : torch.nn.Module
        Fitted control network, mapping d_w input coordinates to q features.
    A_ : torch.Tensor of shape (G, q, J)
        Fitted response coefficients for each category.

    Examples
    --------
    Fit two categories in two environments, with one measured output per row.

    >>> import numpy as np
    >>> from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    ...     NeuralPerXEmbeddingBridge,
    ... )
    >>> xs = [np.eye(2), np.eye(2)]
    >>> ys = [np.array([[1.], [2.]]), np.array([[3.], [4.]])]
    >>> ws = [np.array([[0.], [1.]]), np.array([[2.], [3.]])]
    >>> model = NeuralPerXEmbeddingBridge(n_features_w=2, width_w=4, n_steps=1)
    >>> _ = model.fit(xs, ys, ws)
    >>> model.predict(np.eye(2), ws[0]).shape
    (2, 1)
    >>> model.A_.shape
    torch.Size([2, 2, 1])
    """

    n_features_w: int = PerXConfig.n_features_w
    width_w: int = PerXConfig.width_w
    depth: int = PerXConfig.depth
    reg: float = PerXConfig.reg
    lr: float = PerXConfig.lr
    weight_decay: float = PerXConfig.weight_decay
    n_steps: int = PerXConfig.n_steps
    cross_fit_folds: int = PerXConfig.cross_fit_folds
    seed: int = PerXConfig.seed
    device: str = PerXConfig.device
    log_key: str | None = PerXConfig.log_key
    gene_block_size: int = PerXConfig.gene_block_size

    def __post_init__(self):
        """Require a positive output block size."""
        if self.gene_block_size < 1:
            raise ValueError("gene_block_size must be positive")

    def fit(
        self,
        xs: list[np.ndarray],
        ys: list[np.ndarray],
        ws: list[np.ndarray],
        counts: list[np.ndarray] | None = None,
        checkpoints: Collection[int] = (),
        on_checkpoint: Callable[[int], None] | None = None,
        on_training_loss: Callable[[int, float], None] | None = None,
    ) -> NeuralPerXEmbeddingBridge:
        """Fit control networks and response coefficients from D environments.

        Entry e in each list belongs to the same environment. Counts have the same
        effect as repeating each target row that many times.

        Parameters
        ----------
        xs : list of numpy.ndarray
            One-hot category rows; xs[e] has shape (n_e, G).
        ys : list of numpy.ndarray
            Target rows; ys[e] has shape (n_e, J).
        ws : list of numpy.ndarray
            Control samples; ws[e] has shape (m_e, d_w), or (m_e,) for one coordinate.
        counts : list of numpy.ndarray or None, default=None
            Row weights; counts[e] has shape (n_e,). None gives every row weight one.
        checkpoints : collection of int, default=()
            Update counts at which to call on_checkpoint. Zero uses the initial
            networks; coefficients are fitted before each callback.
        on_checkpoint : callable or None, default=None
            on_checkpoint(step); required when checkpoints is nonempty.
        on_training_loss : callable or None, default=None
            on_training_loss(step, loss). The loss is count-weighted prediction
            MSE, plus the scaled coefficient penalty for in-sample training.
            Step k uses the network after k updates, including k=n_steps for
            the final loss.

        Returns
        -------
        self : NeuralPerXEmbeddingBridge
            Fitted model, with final coefficients using all environments.
        """
        torch.manual_seed(self.seed)
        data = self._to_tensors(xs, ys, ws, counts)
        self._build_networks(data)
        if 0 in checkpoints:
            self._solve_A(data)
            on_checkpoint(0)
        self._train(data, checkpoints, on_checkpoint, on_training_loss)
        if self.n_steps not in checkpoints:
            self._solve_A(data)
        return self

    def predict(self, x: np.ndarray, w_target: np.ndarray) -> np.ndarray:
        """Predict category means in one target environment.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_new, G)
            One-hot categories to predict, in the desired order.
        w_target : numpy.ndarray of shape (m_target, d_w) or (m_target,)
            Control samples from the target environment.

        Returns
        -------
        numpy.ndarray of shape (n_new, J)
            Predictions in the units of the targets supplied to fit.
        """
        return self.predict_mixture(x, [w_target], np.ones(1))

    @torch.no_grad()
    def predict_mixture(
        self, x: np.ndarray, w_targets: list[np.ndarray], environment_weights: np.ndarray
    ) -> np.ndarray:
        """Predict responses averaged over a common environment distribution.

        Average the features before applying coefficients:

            z = sum_e p[e] * mean_j psi(w[e, j])

        Shared and hybrid models also retain their intercepts. The hybrid's
        embedding includes projected control means when its linear term is enabled,
        so the matrix B uses the same environment weights.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_new, G)
            One-hot fitted categories, in the desired order.
        w_targets : list of numpy.ndarray
            Nonempty control samples per environment, already in fitted input
            units. Each array has shape (m_e, d_w), or (m_e,) for one input.
        environment_weights : numpy.ndarray of shape (len(w_targets),)
            Finite, nonnegative probabilities summing to one within absolute
            tolerance 1e-8. Accepted roundoff is normalized away.

        Returns
        -------
        numpy.ndarray of shape (n_new, J)
            Averaged predictions in the fitted target units.

        Raises
        ------
        ValueError
            If controls are empty, nonfinite or have an invalid shape, or
            probabilities have an invalid shape, value or sum.
        """
        weights = np.asarray(environment_weights, dtype=float)
        if (
            not w_targets
            or weights.shape != (len(w_targets),)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError(
                "environment_weights must match controls and be probabilities summing to one."
            )
        weights = weights / weights.sum()
        expected_width = getattr(self, "n_features_in_", None)
        if expected_width is None:
            # Previously saved bridges predate the input-width metadata.
            expected_width = next(
                layer.in_features for layer in self.psi_.modules() if isinstance(layer, nn.Linear)
            )
        embedding = 0.0
        for samples, probability in zip(w_targets, weights, strict=True):
            samples = np.asarray(samples)
            if samples.ndim not in (1, 2) or not len(samples) or not np.isfinite(samples).all():
                raise ValueError("Controls must be nonempty, finite sample arrays.")
            width = 1 if samples.ndim == 1 else samples.shape[1]
            if width != expected_width:
                raise ValueError("Control feature width must match training controls.")
            if probability == 0:
                continue
            controls = as_tensor(samples, self.device)
            embedding = embedding + probability * self._control_embedding(controls)
        pred = self._predict_embedded(embedding)
        return pred[as_tensor(x, self.device).argmax(dim=1)].cpu().numpy()

    def _predict_embedded(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply per-category coefficients to mean features; return (G, J)."""
        return embedding @ self.A_

    def _to_tensors(self, xs, ys, ws, counts) -> GroupedRidgeData:
        """Group response counts and target sums by category and environment.

        Square targets before grouping so repeated pairs retain their full
        squared error. Store controls separately for learning the features.
        """
        n_cats, n_envs, n_genes = xs[0].shape[1], len(ws), ys[0].shape[1]
        c = torch.zeros(n_cats, n_envs, dtype=torch.float64, device=self.device)  # (G, D)
        cy = c.new_zeros((n_cats, n_envs, n_genes))  # (G, D, J)
        cyy = c.new_zeros((n_envs,))  # (D,)
        for e, (x, y) in enumerate(zip(xs, ys, strict=True)):
            cat = torch.as_tensor(np.asarray(x).argmax(axis=1), device=self.device)  # (n_e,)
            y = torch.as_tensor(
                y, dtype=torch.float64, device=self.device
            )  # (n_e, J), possibly empty
            count = torch.as_tensor(
                np.ones(len(x)) if counts is None else counts[e],
                dtype=torch.float64,
                device=self.device,
            )  # (n_e,)
            c[:, e].index_add_(0, cat, count)
            cy[:, e].index_add_(0, cat, count[:, None] * y)
            cyy[e] = (count * y.square().mean(dim=1)).sum()
        target_gram = None  # (G, D, D); only the Gram solver (D <= J) reads it
        if NeuralRidgeSolver.uses_gram(n_envs, n_genes):
            target_gram = (cy @ cy.transpose(1, 2)) / n_genes
        return GroupedRidgeData(
            c=c,
            cy=cy,
            cyy=cyy,
            target_gram=target_gram,
            w_envs=[as_tensor(w, self.device) for w in ws],
        )

    def _build_networks(self, data: GroupedRidgeData) -> None:
        """Create the control network psi_ on the requested device."""
        self.n_features_in_ = data.w_envs[0].shape[1]
        self.psi_ = self._new_control_network(self.n_features_in_)

    def _new_control_network(self, input_size: int) -> nn.Module:
        """Build a control network with the configured width and feature count."""
        return control_mlp(input_size, self.width_w, self.depth, self.n_features_w).to(self.device)

    def _train(self, data, checkpoints, on_checkpoint, on_training_loss) -> None:
        """Update the networks with AdamW and report losses and checkpoints.

        Refit response coefficients before each checkpoint callback so it
        can predict with the current networks.
        """
        params = self.psi_.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        logging = self.log_key and wandb.run is not None  # the caller's run, if it opened one
        if logging:  # the loss against this fit's own step
            wandb.define_metric(f"{self.log_key}/*", step_metric=f"{self.log_key}/step")
        progress = trange(self.n_steps, desc="neural bridge", leave=False)
        for step in progress:
            optimizer.zero_grad()
            loss = self._training_loss(data)
            loss.backward()
            optimizer.step()
            mse = loss.item()  # evaluated before the update above
            progress.set_postfix(loss=f"{mse:.4f}")
            if on_training_loss is not None:
                on_training_loss(step, mse)
            if logging:
                wandb.log({f"{self.log_key}/loss": mse, f"{self.log_key}/step": step})
            if step + 1 in checkpoints:
                self._solve_A(data)
                on_checkpoint(step + 1)
        if on_training_loss is not None:
            # Cross-fitting draws folds on the CPU. Preserve its RNG for later fits.
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                on_training_loss(self.n_steps, self._training_loss(data).item())

    def _training_loss(self, data) -> torch.Tensor:
        """Return prediction MSE, adding the ridge penalty only in sample."""
        if self.cross_fit_folds:
            return self._cross_fit_loss(data)
        return self._in_sample_loss(data)

    def _in_sample_loss(self, data: GroupedRidgeData) -> torch.Tensor:
        """Return in-sample MSE plus the ridge penalty scaled by 1 / (J * N)."""
        solver = self._ridge_solver(data, self._embed(data.w_envs))
        return (solver.prediction_sse() + solver.penalty()) / data.c.sum()

    def _cross_fit_loss(self, data) -> torch.Tensor:
        """Draw folds and return count-weighted MSE on held-out environments."""
        embeddings = self._embed(data.w_envs)
        folds = self._env_folds(len(embeddings))
        return self._cross_fit_sse(data, embeddings, folds) / data.c.sum()

    def _cross_fit_sse(self, data: GroupedRidgeData, embeddings, folds) -> torch.Tensor:
        """Sum held-out fold errors with checkpointed ridge calculations.

        Recompute each fold during backpropagation to limit saved tensors.
        """
        return checkpointed_sse(self._fit_and_score_sse, data, embeddings, folds)

    def _fit_and_score_sse(self, data: GroupedRidgeData, embeddings, fit, scored) -> torch.Tensor:
        """Fit on the fit mask and return the squared error on the scored mask."""
        return self._ridge_solver(data, embeddings, fit).prediction_sse(scored)

    def _env_folds(self, n_envs: int) -> torch.Tensor:
        """Draw balanced fold labels for the environments.

        Require at least two folds and two environments.
        """
        n_folds = min(self.cross_fit_folds, n_envs)
        if n_folds < 2:
            raise ValueError("cross-fitting needs at least two folds and two environments")
        env_fold = torch.randperm(n_envs) % n_folds  # balanced: fold sizes differ by at most one
        return env_fold.to(self.device)

    def _ridge_solver(self, data, embeddings, fit=None) -> NeuralRidgeSolver:
        """Prepare per-category ridge fits on the selected environments."""
        return NeuralRidgeSolver.per_category(data, embeddings, fit, reg=self.reg)

    @torch.no_grad()
    def _solve_A(self, data: GroupedRidgeData) -> None:
        """Fit A_ on all environments, one output block at a time.

        Reuse coefficient storage when its shape and device match. Output
        blocks limit temporary memory; the full A_ stays on the device.
        """
        solver = self._ridge_solver(data, self._embed(data.w_envs))
        groups, _, genes = data.cy.shape
        shape = (groups, self.n_features_w, genes)
        current = getattr(self, "A_", None)
        if current is None or current.shape != shape or current.device != data.cy.device:
            self.A_ = data.cy.new_empty(shape)
        for start in range(0, genes, self.gene_block_size):
            block = slice(start, start + self.gene_block_size)
            self.A_[:, :, block] = solver.coefficients(block).local

    def _embed(self, w_envs: list[torch.Tensor]) -> torch.Tensor:
        """Stack one mean feature vector per environment.

        Return shape (D, q) for per-x or (D, 2q) for shared models. The hybrid
        inserts r projected control coordinates between shared and deviation
        features, giving shape (D, 2q + r). Disabling its linear term sets r = 0.
        """
        return torch.stack([self._control_embedding(w) for w in w_envs])

    def _control_embedding(self, controls: torch.Tensor) -> torch.Tensor:
        """Average control-network outputs into a feature vector of shape (q,)."""
        return self.psi_(controls).mean(dim=0)


@dataclass(eq=False)
class NeuralSharedEmbeddingBridge(NeuralPerXEmbeddingBridge):
    """Fit a shared effect and a separate deviation for each category.

    Two independent networks produce mean control features s[e] and d[e], each
    with q entries. Fit the intercepts a and coefficients b and c together:

        y_hat[x, e] = a[x] + s[e] @ b + d[e] @ c[x]
        penalty = N * (reg * (sum_x ||a[x]||^2 + ||b||^2)
                       + deviation_reg * sum_x ||c[x]||^2)

    Minimise weighted squared errors plus this penalty, including the intercept
    penalty. Larger deviation_reg pulls category responses toward the shared
    effect. Network training and other settings follow NeuralPerXEmbeddingBridge;
    n_features_w, width_w and depth apply to each network.

    Parameters
    ----------
    deviation_reg : float or None, default=None
        Positive penalty for category deviations. None uses reg. Both reg and
        deviation_reg must be finite and positive.

    Attributes
    ----------
    psi_ : torch.nn.ModuleDict
        The two control networks, stored under "shared" and "deviation".
    shared_network_ : torch.nn.Module
        Network whose response coefficients are shared across categories.
    deviation_network_ : torch.nn.Module
        Network with a separate response for each category.
    intercept_ : torch.Tensor of shape (G, J)
        Category intercepts a.
    shared_coef_ : torch.Tensor of shape (q, J)
        Shared response coefficients b.
    deviation_coef_ : torch.Tensor of shape (G, q, J)
        Category deviation coefficients c.

    Notes
    -----
    Networks and coefficient attributes are available after ``fit``.

    Examples
    --------
    Fit two control networks and inspect the shared response coefficients.

    >>> import numpy as np
    >>> from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    ...     NeuralSharedEmbeddingBridge,
    ... )
    >>> xs = [np.eye(2), np.eye(2)]
    >>> ys = [np.array([[1.], [2.]]), np.array([[3.], [4.]])]
    >>> ws = [np.array([[0.], [1.]]), np.array([[2.], [3.]])]
    >>> model = NeuralSharedEmbeddingBridge(n_features_w=2, width_w=4, n_steps=1)
    >>> _ = model.fit(xs, ys, ws)
    >>> model.predict(np.eye(2), ws[0]).shape
    (2, 1)
    >>> model.shared_coef_.shape
    torch.Size([2, 1])
    """

    deviation_reg: float | None = Shared2NNConfig.deviation_reg

    def __post_init__(self):
        """Default deviation_reg to reg and require finite, positive penalties."""
        super().__post_init__()
        if self.deviation_reg is None:
            self.deviation_reg = self.reg
        for name in ("reg", "deviation_reg"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def shared_network_(self) -> nn.Module:
        """The fitted control network for shared effects."""
        return self.psi_["shared"]

    @property
    def deviation_network_(self) -> nn.Module:
        """The fitted control network for category deviations."""
        return self.psi_["deviation"]

    def _build_networks(self, data: GroupedRidgeData) -> None:
        """Create independent shared and deviation control networks."""
        super()._build_networks(data)
        input_size = data.w_envs[0].shape[1]
        # A separate initialization stream preserves the existing cross-fitting RNG.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(self.seed + 1)
            torch.set_rng_state(generator.get_state())
            deviation = self._new_control_network(input_size)
        self.psi_ = nn.ModuleDict({"shared": self.psi_, "deviation": deviation})

    def _control_embedding(self, controls: torch.Tensor) -> torch.Tensor:
        """Join shared and deviation feature means into a vector of shape (2q,)."""
        return torch.cat([network(controls).mean(dim=0) for network in self.psi_.values()])

    def _ridge_solver(self, data, embeddings, fit=None) -> NeuralRidgeSolver:
        """Prepare a joint fit with separate neural and linear penalties."""
        q = self.n_features_w
        shared_reg = embeddings.new_full(
            (embeddings.shape[1] - q,), getattr(self, "linear_reg", np.inf)
        )
        shared_reg[:q] = self.reg
        return NeuralRidgeSolver.shared_deviation(
            data,
            embeddings,
            fit,
            reg=self.reg,
            deviation_reg=self.deviation_reg,
            shared_reg=shared_reg,
        )

    def _cross_fit_sse(self, data: GroupedRidgeData, embeddings, folds) -> torch.Tensor:
        """Sum held-out fold errors.

        The Gram path (D <= J) combines the folds' target projections so it
        reads cy once. The direct path (D > J) fits each fold separately and
        recomputes it during backpropagation, holding one fold's arrays at a
        time. Hybrid shared features include r projected control coordinates,
        so the shared matrices have width q + r.
        """
        if NeuralRidgeSolver.solver_type(data) is not GramRidgeSolver:
            return super()._cross_fit_sse(data, embeddings, folds)  # checkpointed per fold
        terms = [
            self._ridge_solver(data, embeddings, folds != k).shared_loss_terms(folds == k)
            for k in range(int(folds.max()) + 1)
        ]
        return NeuralRidgeSolver.batched_shared_sse(data, terms)

    def _allocate_coefficients(self, data: GroupedRidgeData, n_shared: int) -> None:
        """Allocate coefficient arrays or reuse matching shapes and devices."""
        groups, _, genes = data.cy.shape
        shapes = {
            "intercept_": (groups, genes),
            "shared_coef_": (n_shared, genes),
            "deviation_coef_": (groups, self.n_features_w, genes),
        }
        for name, shape in shapes.items():
            current = getattr(self, name, None)
            if current is None or current.shape != shape or current.device != data.cy.device:
                setattr(self, name, data.cy.new_empty(shape))

    @torch.no_grad()
    def _solve_A(self, data: GroupedRidgeData) -> None:
        """Fit all response coefficients in output blocks.

        Write into the model's coefficient arrays without tracking gradients.
        """
        embeddings = self._embed(data.w_envs)
        fit = torch.ones(len(embeddings), dtype=torch.bool, device=self.device)
        solver = self._ridge_solver(data, embeddings, fit)
        self._allocate_coefficients(data, solver.shared_features.shape[1])
        for start in range(0, data.cy.shape[-1], self.gene_block_size):
            block = slice(start, start + self.gene_block_size)
            local, shared = solver.coefficients(block)
            self.intercept_[:, block] = local[:, 0]
            self.deviation_coef_[:, :, block] = local[:, 1:]
            self.shared_coef_[:, block] = shared

    def _predict_embedded(self, embedding: torch.Tensor) -> torch.Tensor:
        """Combine fitted effects into predictions of shape (G, J)."""
        q = self.n_features_w
        shared, deviation = embedding.split((len(embedding) - q, q))
        return self.intercept_ + shared @ self.shared_coef_ + deviation @ self.deviation_coef_


@dataclass(eq=False)
class NeuralHybridEmbeddingBridge(NeuralSharedEmbeddingBridge):
    """Add a full control-to-output matrix to the shared neural model.

    Fit the matrix B jointly with the intercepts, shared response and category
    deviations. Control and output widths may differ. Other settings and
    fitted attributes follow NeuralSharedEmbeddingBridge.

    Parameters
    ----------
    linear_reg : float, default=1e-3
        Positive penalty on the squared entries of B, scaled by total training
        weight N. Positive infinity disables the linear term and gives the
        shared model. Zero, negative values and NaN are rejected.

    Attributes
    ----------
    control_basis_ : torch.Tensor of shape (d_w, r)
        Orthonormal basis V for the retained training control directions,
        available after ``fit``. Its width r is the numerical rank of the
        training control means, or zero when the linear term is disabled.
    shared_coef_ : torch.Tensor of shape (q + r, J)
        Fitted shared coefficients b followed by R, available after ``fit``.
    full_coef_ : torch.Tensor of shape (d_w, J)
        The fitted matrix B = V @ R, allocated on access after ``fit``. Rows
        are control coordinates; columns are outputs. Zero when the linear
        term is disabled.

    Notes
    -----
    With the linear term enabled, the prediction and coefficient penalty are::

        y_hat[x, e] = a[x] + s[e] @ b + mean(W[e]) @ B + d[e] @ c[x]
        penalty = N * (reg * (||a||^2 + ||b||^2)
                       + deviation_reg * ||c||^2 + linear_reg * ||B||^2)

    The ridge solution B lies in the row space of the D training control
    means. Write B = V @ R, where V = control_basis_ and R has shape (r, J).
    Projected means mean(W[e]) @ V join the shared neural features, and R
    occupies the last r rows of shared_coef_. Orthonormal columns of V give
    ||B||^2 = ||R||^2, preserving the ridge penalty.

    A reduced SVD is computed once per fit and reused across cross-fit folds.
    Singular values at or below max(D, d_w) * eps * largest_singular_value
    are discarded, where eps is the control tensor's machine precision.
    Apart from this numerical rank cutoff, the projection preserves the fit.
    The shared solve has width q + r, with r <= min(D, d_w). Predictions use
    the factors directly; only accessing full_coef_ allocates the full matrix.
    If a wrapper scales controls, B uses those scaled units.

    Examples
    --------
    Fit two control coordinates to one output and inspect the full matrix.

    >>> import numpy as np
    >>> from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    ...     NeuralHybridEmbeddingBridge,
    ... )
    >>> xs = [np.eye(2), np.eye(2)]
    >>> ys = [np.array([[1.], [2.]]), np.array([[3.], [4.]])]
    >>> ws = [np.array([[0., 1.], [1., 0.]]), np.array([[2., 1.], [3., 1.]])]
    >>> model = NeuralHybridEmbeddingBridge(n_features_w=2, width_w=4, n_steps=1)
    >>> _ = model.fit(xs, ys, ws)
    >>> model.predict(np.eye(2), ws[0]).shape
    (2, 1)
    >>> model.full_coef_.shape
    torch.Size([2, 1])
    """

    linear_reg: float = HybridConfig.linear_reg

    def __post_init__(self):
        """Check penalties; allow positive infinity only for linear_reg."""
        super().__post_init__()
        if np.isnan(self.linear_reg) or self.linear_reg <= 0:
            raise ValueError("linear_reg must be positive (infinity disables the linear term)")

    @property
    def full_coef_(self) -> torch.Tensor:
        """The fitted control-to-output matrix B = V @ R.

        Returns
        -------
        torch.Tensor of shape (d_w, J)
            Rows correspond to control coordinates and columns to outputs.
            Allocated on each access after ``fit``; changing it does not update
            the model. Zero when linear_reg is infinite.
        """
        return self.control_basis_ @ self.shared_coef_[self.n_features_w :]

    def _build_networks(self, data: GroupedRidgeData) -> None:
        """Create the networks and a control basis, once per fit.

        Retain the numerical row space of the training control means.
        Cross-fit folds use subsets of these rows, so they share the basis.
        """
        super()._build_networks(data)
        means = torch.stack([w.mean(dim=0) for w in data.w_envs])
        _, singular, right = torch.linalg.svd(means, full_matrices=False)
        tolerance = singular.max() * max(means.shape) * torch.finfo(means.dtype).eps
        # No linear term (infinite linear_reg): an empty basis gives B = 0.
        keep = (singular > tolerance) & bool(np.isfinite(self.linear_reg))
        self.control_basis_ = right[keep].T

    def _control_embedding(self, controls: torch.Tensor) -> torch.Tensor:
        """Join shared features, projected control means and deviations.

        Return shape (2q + r,), where r is the width of control_basis_.
        """
        shared, deviation = (network(controls).mean(dim=0) for network in self.psi_.values())
        return torch.cat([shared, controls.mean(dim=0) @ self.control_basis_, deviation])
