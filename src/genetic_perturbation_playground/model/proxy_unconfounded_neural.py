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
    The hybrid needs d_w = J when its linear term is enabled.
N : total training weight
    Sum of response counts across all categories in the current training split.
Q : target-product matrix
    target_gram has shape (G, D, D): Q[x] = cy[x] @ cy[x].T / J.
    Uppercase Q is a matrix; lowercase q is a feature count.

Networks map controls to features, averaged within each environment.
NeuralRidgeSolver fits the response coefficients with penalties scaled by N.
Network training uses prediction mean squared error (MSE), without that penalty:

    SSE = sum_i count[i] * sum_j (prediction[i, j] - target[i, j])**2 / J
    MSE = SSE / sum_i count[i]

SSE means sum of squared errors, weighted by counts and averaged over outputs.
Cross-fitting fits coefficients on some environments and scores the others.
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
from genetic_perturbation_playground.utils.ridge import NeuralRidgeSolver


@dataclass(eq=False)
class NeuralPerXEmbeddingBridge:
    """Fit a separate response to learned control features for each category.

    Average learned features over controls W[e] in environment e:

        y_hat[x, e] = mean(psi(W[e])) @ A[x]

    For fixed features, fit A by minimising

        sum_i count[i] * ||prediction[i] - target[i]||^2
        + N * reg * sum_x ||A[x]||^2

    All categories share psi. Missing pairs have zero weight. Networks use
    float32 and ridge uses float64; only prediction MSE trains the network.

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
            on_training_loss(step, mse). The loss at step k uses the network
            after k updates, including k=n_steps for the final loss.

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

        Shared and hybrid models also retain their intercepts; the hybrid adds
        its fitted slope times the similarly weighted control mean.

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
            Averaged predictions in the fitted target units. Invalid controls
            or probabilities raise ValueError.
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
        embedding, control_mean = 0.0, 0.0
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
            control_mean = control_mean + probability * controls.mean(dim=0)
        pred = self._predict_embedded(embedding, control_mean)
        return pred[as_tensor(x, self.device).argmax(dim=1)].cpu().numpy()

    def _predict_embedded(
        self, embedding: torch.Tensor, control_mean: torch.Tensor
    ) -> torch.Tensor:
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
        target_gram = (cy @ cy.transpose(1, 2)) / n_genes  # (G, D, D)
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
        """Return prediction MSE using the configured environment split."""
        if self.cross_fit_folds:
            return self._cross_fit_loss(data)
        return self._in_sample_loss(data)

    def _in_sample_loss(self, data: GroupedRidgeData) -> torch.Tensor:
        """Fit and score on all environments; return count-weighted MSE."""
        mu = self._embed(data.w_envs)  # (D, q)
        all_envs = torch.ones(len(mu), dtype=bool, device=mu.device)  # (D,)
        return self._fit_and_score_sse(data, mu, all_envs, all_envs) / data.c.sum()

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
        """Stack mean features: (D, q) for per-x, (D, 2q) for shared/hybrid."""
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
    linear_coef_ : torch.Tensor of shape (J,)
        Zero slopes; used by the hybrid subclass when it adds a linear term.

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
        """Return the control network for shared effects."""
        return self.psi_["shared"]

    @property
    def deviation_network_(self) -> nn.Module:
        """Return the control network for category deviations."""
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

    def _ridge_solver(
        self, data, embeddings, fit=None, *, linear_reg=float("inf")
    ) -> NeuralRidgeSolver:
        """Prepare the joint shared/deviation fit, with optional control slopes."""
        return NeuralRidgeSolver.shared_deviation(
            data,
            embeddings,
            fit,
            reg=self.reg,
            deviation_reg=self.deviation_reg,
            linear_reg=linear_reg,
        )

    def _cross_fit_sse(self, data: GroupedRidgeData, embeddings, folds) -> torch.Tensor:
        """Sum held-out fold errors with one combined target projection.

        Each fold stores small matrices; the combined projection reads cy once.
        """
        terms = [
            self._ridge_solver(data, embeddings, folds != k).shared_loss_terms(folds == k)
            for k in range(int(folds.max()) + 1)
        ]
        return NeuralRidgeSolver.batched_shared_sse(data, terms)

    def _allocate_coefficients(self, data: GroupedRidgeData) -> None:
        """Allocate coefficient arrays, reusing storage when shapes and devices match."""
        groups, _, genes = data.cy.shape
        shapes = {
            "intercept_": (groups, genes),
            "shared_coef_": (self.n_features_w, genes),
            "deviation_coef_": (groups, self.n_features_w, genes),
            "linear_coef_": (genes,),
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
        self._allocate_coefficients(data)
        for start in range(0, data.cy.shape[-1], self.gene_block_size):
            block = slice(start, start + self.gene_block_size)
            local, shared, linear = solver.coefficients(block)
            self.intercept_[:, block] = local[:, 0]
            self.deviation_coef_[:, :, block] = local[:, 1:]
            self.shared_coef_[:, block] = shared
            self.linear_coef_[block] = linear

    def _predict_embedded(
        self, embedding: torch.Tensor, control_mean: torch.Tensor
    ) -> torch.Tensor:
        """Add intercepts, shared effects and category deviations; return (G, J)."""
        shared, deviation = embedding.chunk(2)
        return self.intercept_ + shared @ self.shared_coef_ + deviation @ self.deviation_coef_


@dataclass(eq=False)
class NeuralHybridEmbeddingBridge(NeuralSharedEmbeddingBridge):
    """Add one control slope per output to the shared two-network model.

    For output j, fit all terms together:

        y_hat[x, e, j] = a[x, j] + beta[j] * mean(W[e])_j
                        + s[e] @ b[:, j] + d[e] @ c[x, :, j]

    The penalty is the shared model's penalty plus N * linear_reg * ||beta||^2.
    Control and target coordinates must match, so d_w = J. If a wrapper scales
    controls, the linear term uses those scaled values too.

    Other settings and fitted attributes follow NeuralSharedEmbeddingBridge.
    Its reg and deviation_reg penalties must remain finite and positive.

    Parameters
    ----------
    linear_reg : float, default=1e-3
        Nonnegative penalty for control slopes. Infinity disables the linear term
        and gives the shared model. With zero penalty, an unidentifiable slope is zero.

    Attributes
    ----------
    linear_coef_ : torch.Tensor of shape (J,)
        Fitted slopes beta, one for each matching control/output coordinate.

    Examples
    --------
    The controls and targets below each have one coordinate.

    >>> import numpy as np
    >>> from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    ...     NeuralHybridEmbeddingBridge,
    ... )
    >>> xs = [np.eye(2), np.eye(2)]
    >>> ys = [np.array([[1.], [2.]]), np.array([[3.], [4.]])]
    >>> ws = [np.array([[0.], [1.]]), np.array([[2.], [3.]])]
    >>> model = NeuralHybridEmbeddingBridge(n_features_w=2, width_w=4, n_steps=1)
    >>> _ = model.fit(xs, ys, ws)
    >>> model.predict(np.eye(2), ws[0]).shape
    (2, 1)
    >>> model.linear_coef_.shape
    torch.Size([1])
    """

    linear_reg: float = HybridConfig.linear_reg

    def __post_init__(self):
        """Check neural penalties and require linear_reg >= 0; infinity disables it."""
        super().__post_init__()
        if np.isnan(self.linear_reg) or self.linear_reg < 0:
            raise ValueError("linear_reg must be nonnegative (infinity disables the linear term)")

    def _to_tensors(self, xs, ys, ws, counts) -> GroupedRidgeData:
        """Add control means and pooled targets when the linear term is enabled.

        Require matching control and target dimensions for that term.
        """
        data = super()._to_tensors(xs, ys, ws, counts)
        if np.isinf(self.linear_reg):
            return data
        means = torch.stack([w.mean(dim=0) for w in data.w_envs])
        if means.shape[1] != data.cy.shape[-1]:
            raise ValueError("Hybrid transfer requires matching proxy and target dimensions")
        return data._replace(control_mean=means, pooled_targets=data.cy.sum(dim=0))

    def _ridge_solver(self, data, embeddings, fit=None) -> NeuralRidgeSolver:
        """Prepare the joint fit with this model's control-slope penalty."""
        return super()._ridge_solver(data, embeddings, fit, linear_reg=self.linear_reg)

    def _cross_fit_sse(self, data: GroupedRidgeData, embeddings, folds) -> torch.Tensor:
        """Checkpoint hybrid fold calculations to limit saved tensors.

        Use the shared model's scoring path when the linear term is disabled.
        """
        if np.isinf(self.linear_reg):
            return super()._cross_fit_sse(data, embeddings, folds)
        return checkpointed_sse(self._fit_and_score_sse, data, embeddings, folds)

    def _predict_embedded(
        self, embedding: torch.Tensor, control_mean: torch.Tensor
    ) -> torch.Tensor:
        """Add matching-coordinate control slopes to the neural predictions."""
        neural = super()._predict_embedded(embedding, control_mean)
        if np.isinf(self.linear_reg):
            return neural
        if len(control_mean) != len(self.linear_coef_):
            raise ValueError("Hybrid transfer requires matching proxy and target dimensions")
        return neural + control_mean * self.linear_coef_
