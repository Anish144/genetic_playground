"""Ridge fits for linear and neural transfer models.

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
N : total training weight
    Sum of response counts across all categories in the current training split.
B : number of selected outputs
    Size of an output slice when coefficients are computed in blocks.
K : number of columns to solve
    Width of the right-hand side passed to the matrix solve.
Q : target-product matrix
    target_gram has shape (G, D, D): Q[x] = cy[x] @ cy[x].T / J.
    Uppercase Q is a matrix; lowercase q is a feature count.

Errors and penalties
--------------------
SSE means sum of squared errors. Here, rows have count weights and we average over
outputs. For target y, prediction y_hat, row weight c and J outputs:

    SSE = sum_i c[i] * sum_j (y_hat[i, j] - y[i, j])**2 / J
    mean_squared_error = SSE / sum_i c[i]

Coefficient fits minimise J * SSE plus a ridge penalty on squared coefficients.
Scoring returns SSE alone. In equations, ||A||^2 sums squared entries and @ is
matrix multiplication.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from genetic_perturbation_playground.data.neural import GroupedRidgeData


class LinearRidgeSolver:
    """Fit one control slope per output, with a separate intercept per category.

    For row i in category x, minimise

        loss = sum_i c[i] * ||y[i] - a[x] - W[i] * beta||^2 + N * reg * ||beta||^2

    Intercepts a have no penalty. Subtracting each category's weighted means
    from y and W gives dy and dw. Each slope then has the solution

        beta[j] = sum_i(c[i] * dw[i, j] * dy[i, j])
                  / (sum_i(c[i] * dw[i, j]**2) + N * reg)

    Category sums suffice; no full design matrix is needed. The caller recovers
    each intercept as mean(y) - beta * mean(W) within its category.

    Parameters
    ----------
    reg : float, default=1e-3
        Finite, nonnegative penalty, scaled by the total training count N.
        Zero gives ordinary least squares. Applies in the units of the supplied W.

    Attributes
    ----------
    reg : float
        Penalty for the slopes. Category intercepts remain unpenalised.

    Examples
    --------
    One category has controls 0 and 2, and targets 1 and 5. Without a penalty
    the slope is 2. With reg=1, ridge shrinks it to 1.

    >>> import numpy as np
    >>> from genetic_perturbation_playground.utils.ridge import LinearRidgeSolver
    >>> stats = {
    ...     "n": np.array([2]),
    ...     "y_mean": np.array([[3.]]),
    ...     "w_mean": np.array([[1.]]),
    ...     "W": np.array([[0.], [2.]]),
    ...     "batch_n": np.array([1, 1]),
    ...     "batch_y_sum": np.array([[1.], [5.]]),
    ... }
    >>> LinearRidgeSolver(reg=1.0).solve(stats).tolist()
    [1.0]
    >>> LinearRidgeSolver(reg=0.0).solve(stats).tolist()
    [2.0]
    """

    def __init__(self, reg: float = 1e-3):
        if not np.isfinite(reg) or reg < 0:
            raise ValueError("reg must be finite and nonnegative")
        self.reg = reg

    def solve(self, stats: dict) -> np.ndarray:
        """Fit ridge slopes from category and environment sums.

        Parameters
        ----------
        stats : dict
            Output of ``data.transfer.summary_stats``. Uses ``y_mean``,
            ``w_mean``, ``n``, ``W``, ``batch_y_sum`` and ``batch_n``.
            Controls may already be standardised.

        Returns
        -------
        numpy.ndarray of shape (J,)
            Slope per output in the supplied control units. Computed in float64.
            When reg=0 and there is no control variation within categories,
            returns zero for that output.

        Raises
        ------
        ValueError
            If the total training count is not positive.
        """
        y = np.asarray(stats["y_mean"], dtype=np.float64)
        w = np.asarray(stats["w_mean"], dtype=np.float64)
        n = np.asarray(stats["n"], dtype=np.float64)[:, None]
        control = np.asarray(stats["W"], dtype=np.float64)
        batch_n = np.asarray(stats["batch_n"], dtype=np.float64)
        batch_y_sum = np.asarray(stats["batch_y_sum"], dtype=np.float64)
        total = n.sum()
        if total <= 0:
            raise ValueError("A linear ridge fit needs positive total response weight")
        # Global shifts leave slopes unchanged because category intercepts are free.
        # Centre first to avoid subtracting nearly equal, large products below.
        control_origin = (batch_n[:, None] * control).sum(axis=0) / total
        target_origin = batch_y_sum.sum(axis=0) / total
        control = control - control_origin
        w = w - control_origin
        y = y - target_origin
        batch_y_sum = batch_y_sum - batch_n[:, None] * target_origin
        numerator = (control * batch_y_sum).sum(axis=0) - (n * y * w).sum(axis=0)
        variance = (batch_n[:, None] * control**2).sum(axis=0) - (n * w**2).sum(axis=0)
        # Subtraction can make a zero variance slightly negative through rounding.
        denominator = np.maximum(variance, 0.0) + total * self.reg
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )


class _SharedLossTerms(NamedTuple):
    """Terms needed to score one fit of the shared model.

    Attributes
    ----------
    local_sse : torch.Tensor of shape ()
        Error from the category fit alone, averaged over outputs.
    shared_gram : torch.Tensor of shape (q, q)
        Matrix for fitting the shared coefficients, including their penalty.
    score_gram : torch.Tensor of shape (q, q)
        Matrix for the squared shared prediction on the scored rows.
    fit_map : torch.Tensor of shape (G, q, D)
        Shared coefficients b solve shared_gram @ b = sum_x fit_map[x] @ cy[x].
    cross_map : torch.Tensor of shape (G, q, D)
        Maps target sums to the product of local errors and shared features.

    Examples
    --------
    Store the terms for one split. In normal use, ``shared_loss_terms`` creates
    this record from a fitted solver.

    >>> import torch
    >>> from genetic_perturbation_playground.utils.ridge import _SharedLossTerms
    >>> terms = _SharedLossTerms(
    ...     local_sse=torch.tensor(2.0),
    ...     shared_gram=torch.eye(1),
    ...     score_gram=torch.eye(1),
    ...     fit_map=torch.ones(1, 1, 2),
    ...     cross_map=torch.zeros(1, 1, 2),
    ... )
    >>> terms.fit_map.shape
    torch.Size([1, 1, 2])
    """

    local_sse: torch.Tensor
    shared_gram: torch.Tensor
    score_gram: torch.Tensor
    fit_map: torch.Tensor
    cross_map: torch.Tensor


class RidgeCoefficients(NamedTuple):
    """Fitted coefficients for each part of a neural bridge.

    Here J counts only the requested outputs.

    Attributes
    ----------
    local : torch.Tensor
        Category coefficients, shape (G, q, J) for per-category fits.
        Shared and hybrid fits use shape (G, q+1, J), with intercepts in row zero.
    shared : torch.Tensor of shape (q, J) or None
        Coefficients shared across categories; None for per-category fits.
    linear : torch.Tensor of shape (J,) or None
        One control slope per output. Zero for shared fits and None for
        per-category fits.

    Examples
    --------
    Per-category fits store only local coefficients. This example has two
    categories, three features and one output.

    >>> import torch
    >>> from genetic_perturbation_playground.utils.ridge import RidgeCoefficients
    >>> coefficients = RidgeCoefficients(local=torch.ones(2, 3, 1))
    >>> coefficients.local.shape
    torch.Size([2, 3, 1])
    >>> coefficients.shared is None and coefficients.linear is None
    True
    """

    local: torch.Tensor
    shared: torch.Tensor | None = None
    linear: torch.Tensor | None = None


class NeuralRidgeSolver:
    """Fit coefficients and score predictions for fixed features.

    Create a solver with ``per_category`` or ``shared_deviation``, then call
    ``coefficients`` or ``prediction_sse``. Fits use small matrix solves and
    support gradients through the features.

    Attributes
    ----------
    data : GroupedRidgeData
        Counts and target sums used by the fit.
    features : torch.Tensor
        Input features: (D, q) for per-category fits, (D, 2q) for shared/hybrid fits.
    fit : torch.Tensor of shape (D,) or None
        Boolean mask selecting training environments. None includes all of them.
    local_map : torch.Tensor
        Maps target sums to category coefficients before adding shared effects.
        Shape (G, q, D) for per-category fits, (G, q+1, D) for shared/hybrid fits.
    shared_features : torch.Tensor of shape (D, q) or None
        Shared features; None for per-category fits.
    local_features : torch.Tensor of shape (D, q+1)
        Intercept column followed by deviation features. Shared/hybrid fits only.
    counts : torch.Tensor of shape (G, D)
        Counts in training environments. Shared/hybrid fits only.
    total : torch.Tensor of shape ()
        Sum of training counts. Shared/hybrid fits only.
    coupling : torch.Tensor of shape (G, q+1, q)
        How category coefficients change with shared coefficients.
        Shared/hybrid fits only.
    shared_gram : torch.Tensor of shape (q, q)
        Matrix for the shared solve after accounting for category effects.
        Shared/hybrid fits only.
    shared_map : torch.Tensor of shape (G, q, D)
        Maps target sums to the shared solve's target term. Shared/hybrid fits only.
    linear_reg : float
        Penalty for the hybrid's control slopes. Infinity disables this term.

    Examples
    --------
    Fit a constant feature to targets 2 and 4 in two environments. With equal
    counts and reg=1, ridge shrinks their mean of 3 to a coefficient of 1.5.

    >>> from genetic_perturbation_playground.data.neural import GroupedRidgeData
    >>> import torch
    >>> from genetic_perturbation_playground.utils.ridge import NeuralRidgeSolver
    >>> targets = torch.tensor([[[2.], [4.]]], dtype=torch.float64)
    >>> data = GroupedRidgeData(
    ...     c=torch.ones(1, 2, dtype=torch.float64),
    ...     cy=targets,
    ...     cyy=targets.square().sum(dim=(0, 2)),
    ...     target_gram=targets @ targets.transpose(1, 2),
    ... )
    >>> features = torch.ones(2, 1, dtype=torch.float64)
    >>> solver = NeuralRidgeSolver.per_category(data, features, reg=1.0)
    >>> solver.coefficients().local.tolist()
    [[[1.5]]]

    The squared error is (2 - 1.5)^2 + (4 - 1.5)^2 = 6.5.

    >>> solver.prediction_sse().item()
    6.5
    """

    def __init__(self, data: GroupedRidgeData, features: torch.Tensor, fit: torch.Tensor | None):
        self.data = data
        self.features = features
        self.fit = fit
        self.shared_features = None
        self.linear_reg = float("inf")

    @classmethod
    def per_category(cls, data: GroupedRidgeData, features, fit=None, *, reg):
        """Fit a separate response to the features for each category.

        For category x and environment e, fit A to minimise

            y_hat[x, e] = features[e] @ A[x]
            loss = J * SSE + N * reg * sum_x ||A[x]||^2

        N includes counts from all categories in the training environments.

        Parameters
        ----------
        data : GroupedRidgeData
            Counts and target sums.
        features : torch.Tensor of shape (D, q)
            Learned features for each environment.
        fit : torch.Tensor of shape (D,) or None, default=None
            Boolean training mask; None includes all environments.
        reg : float
            Ridge penalty per coefficient, before multiplying by N.

        Returns
        -------
        NeuralRidgeSolver
            Solver storing maps A[x] = local_map[x] @ data.cy[x]. Coefficients
            are built only when requested.
        """
        solver = cls(data, features, fit)
        counts = data.c if fit is None else data.c * fit
        fit_features = features if fit is None else features * fit[:, None]
        gram = cls.grouped_gram(counts, fit_features)
        # A matrix right-hand side avoids PyTorch's vector ambiguity when G = q = D.
        rhs = fit_features.T.expand(len(counts), -1, -1)
        solver.local_map = cls.solve(gram, rhs, counts.sum(), reg)
        return solver

    @classmethod
    def shared_deviation(
        cls,
        data: GroupedRidgeData,
        embeddings,
        fit=None,
        *,
        reg,
        deviation_reg,
        linear_reg=float("inf"),
    ):
        """Fit shared effects, category deviations and optional control slopes together.

        Split embeddings into shared features s and deviation features d. Fit

            y_hat[x, e] = a[x] + s[e] @ b + d[e] @ v[x] + W[e] * beta
            penalty = N * (reg * (||a||^2 + ||b||^2)
                           + deviation_reg * ||v||^2 + linear_reg * ||beta||^2)
            loss = J * SSE + penalty

        W is data.control_mean. Omit W * beta and its penalty when linear_reg is
        infinity. The intercepts a have a ridge penalty too.

        Parameters
        ----------
        data : GroupedRidgeData
            Counts and target sums. Finite linear_reg also needs control_mean
            and pooled_targets, with matching control and output coordinates.
        embeddings : torch.Tensor of shape (D, 2q)
            Shared features followed by deviation features.
        fit : torch.Tensor of shape (D,) or None, default=None
            Boolean training mask; None includes all environments.
        reg : float
            Positive penalty for intercepts and shared coefficients.
        deviation_reg : float
            Positive penalty for category deviations.
        linear_reg : float, default=inf
            Nonnegative penalty for control slopes. Infinity disables them.

        Returns
        -------
        NeuralRidgeSolver
            Solver ready to compute coefficients or prediction errors.
        """
        if fit is None:
            fit = torch.ones(len(embeddings), dtype=torch.bool, device=embeddings.device)
        solver = cls(data, embeddings, fit)
        solver.linear_reg = linear_reg
        shared, deviation = embeddings.chunk(2, dim=1)
        solver.shared_features = shared
        solver.local_features = torch.cat((torch.ones_like(deviation[:, :1]), deviation), dim=1)
        solver.counts = data.c * fit
        solver.total = solver.counts.sum()
        if solver.total <= 0:
            raise ValueError("Each ridge fit needs positive total response weight")
        solver._eliminate_local_effects(reg, deviation_reg)
        return solver

    @staticmethod
    def penalised_gram(gram, weight, reg) -> torch.Tensor:
        """Add the ridge penalty to the diagonal of the feature-product matrix.

        Parameters
        ----------
        gram : torch.Tensor of shape (..., q, q)
            F.T @ C @ F, where F holds feature rows and C = diag(row weights).
        weight : float or scalar torch.Tensor
            Total training count N.
        reg : float or torch.Tensor of shape (q,)
            One penalty for all features, or a penalty for each feature.

        Returns
        -------
        torch.Tensor of shape (..., q, q)
            gram + N * reg * I, or gram + N * diag(reg) for separate penalties.
            I is the identity matrix.
        """
        if isinstance(reg, torch.Tensor) and reg.ndim:
            return gram + weight * torch.diag(reg)
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        return gram + reg * weight * eye

    @classmethod
    def solve(cls, gram, cross, weight, reg) -> torch.Tensor:
        """Solve (gram + N * reg * I) @ A = cross.

        Parameters
        ----------
        gram : torch.Tensor of shape (..., q, q)
            F.T @ C @ F, where F holds feature rows and C = diag(row weights).
        cross : torch.Tensor of shape (..., q, K)
            Right-hand side with K columns to solve. For coefficient fits,
            this is F.T @ C @ Y, where Y holds target rows.
        weight : float or scalar torch.Tensor
            Total training count N.
        reg : float or torch.Tensor of shape (q,)
            Ridge penalty. A vector replaces reg * I with diag(reg).

        Returns
        -------
        torch.Tensor of shape (..., q, K)
            Solution A, with gradients through the matrix solve.
        """
        return torch.linalg.solve(cls.penalised_gram(gram, weight, reg), cross)

    @staticmethod
    def grouped_gram(c: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Sum weighted feature products for each category.

        Parameters
        ----------
        c : torch.Tensor of shape (G, D)
            Counts for each category and environment.
        mu : torch.Tensor of shape (D, q)
            Features for each environment.

        Returns
        -------
        torch.Tensor of shape (G, q, q)
            gram[x] = mu.T @ diag(c[x]) @ mu for each category x.
        """
        return (c[:, :, None] * mu).transpose(1, 2) @ mu

    @staticmethod
    def grouped_sse(
        data: GroupedRidgeData, mu: torch.Tensor, R: torch.Tensor, scored: torch.Tensor
    ):
        """Score category fits without building their coefficients for every output.

        Expand each squared error as y_hat^2 - 2 * y_hat * y + y^2. The saved
        target products let us sum these terms without building A = R @ cy.

        Parameters
        ----------
        data : GroupedRidgeData
            Counts, squared targets and target products.
        mu : torch.Tensor of shape (D, q)
            Features for each environment.
        R : torch.Tensor of shape (G, q, D)
            Maps target sums to coefficients. Columns outside the training
            environments must be zero.
        scored : torch.Tensor of shape (D,)
            Boolean mask selecting environments to score.

        Returns
        -------
        torch.Tensor of shape ()
            Weighted sum of squared errors on scored rows, divided by J.
            Does not include the ridge penalty.
        """
        rq = R @ data.target_gram  # (G, q, D)
        scored_mu = mu * scored[:, None]  # (D, q)
        gram = NeuralRidgeSolver.grouped_gram(data.c * scored, mu)  # (G, q, q)
        cross_term = (rq * scored_mu.T).sum()
        quadratic = (rq * (gram @ R)).sum()
        return data.cyy[scored].sum() - 2 * cross_term + quadratic

    @staticmethod
    def project_targets(maps: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Apply maps to target sums and add the results across categories.

        Parameters
        ----------
        maps : torch.Tensor of shape (G, q, D)
            Maps from environments to features for each category.
        targets : torch.Tensor of shape (G, D, J)
            Weighted target sums. May contain only a slice of the outputs.

        Returns
        -------
        torch.Tensor of shape (q, J)
            sum_x maps[x] @ targets[x], computed without a temporary (G, q, J)
            tensor or a copy of targets.
        """
        weights = maps.permute(1, 0, 2).reshape(maps.shape[1], -1)
        return weights @ targets.view(-1, targets.shape[-1])

    @staticmethod
    def batched_shared_sse(data: GroupedRidgeData, terms: list[_SharedLossTerms]) -> torch.Tensor:
        """Sum prediction errors across splits using one combined target projection.

        Parameters
        ----------
        data : GroupedRidgeData
            Target sums shared by all splits.
        terms : list of _SharedLossTerms
            One set of fit and scoring terms per split. Must not be empty.

        Returns
        -------
        torch.Tensor of shape ()
            Sum of each split's weighted squared error, divided by J.
            Does not include the ridge penalty.
        """
        features = terms[0].shared_gram.shape[0]
        maps = torch.cat([m for term in terms for m in (term.fit_map, term.cross_map)], dim=1)
        projected = NeuralRidgeSolver.project_targets(maps, data.cy).reshape(
            len(terms), 2, features, -1
        )
        shared_gram = torch.stack([term.shared_gram for term in terms])
        score_gram = torch.stack([term.score_gram for term in terms])
        coefficients = torch.linalg.solve(shared_gram, projected[:, 0])
        cross = (coefficients * projected[:, 1]).sum()
        quadratic = (coefficients * (score_gram @ coefficients)).sum()
        return sum(term.local_sse for term in terms) + (2 * cross + quadratic) / data.cy.shape[-1]

    def _eliminate_local_effects(self, reg: float, deviation_reg: float) -> None:
        """Express category coefficients in terms of shared coefficients.

        The category equations give

            local = local_map @ cy - coupling @ shared

        Substituting this into the shared equations leaves a q-by-q solve.

        Parameters
        ----------
        reg : float
            Penalty for intercepts and shared coefficients.
        deviation_reg : float
            Penalty for category deviations.

        Returns
        -------
        None
            Stores local_map, coupling, shared_gram and shared_map on the solver.
        """
        local, shared = self.local_features, self.shared_features
        features = shared.shape[1]
        penalties = local.new_full((local.shape[1],), deviation_reg)
        penalties[0] = reg
        local_gram = self.grouped_gram(self.counts, local)
        cross = (self.counts[:, :, None] * local).transpose(1, 2) @ shared
        fit_local = (local * self.fit[:, None]).T.expand(len(self.counts), -1, -1)
        maps = self.solve(local_gram, torch.cat((fit_local, cross), dim=2), self.total, penalties)
        self.local_map, self.coupling = maps.split((len(self.fit), features), dim=2)
        # local_map: (G, q+1, D); coupling: (G, q+1, q).
        gram = shared.T @ (self.counts.sum(dim=0)[:, None] * shared)
        gram = gram - (cross.transpose(1, 2) @ self.coupling).sum(dim=0)
        self.shared_gram = self.penalised_gram(gram, self.total, reg)
        self.shared_map = (shared * self.fit[:, None]).T - cross.transpose(1, 2) @ self.local_map

    def coefficients(self, block: slice | None = None) -> RidgeCoefficients:
        """Compute fitted coefficients for the requested outputs.

        Parameters
        ----------
        block : slice or None, default=None
            Outputs to include; None selects all. Smaller slices limit temporary memory use.

        Returns
        -------
        RidgeCoefficients
            Category, shared and linear coefficients for these outputs.
            Gradients can pass through the fit.
        """
        if block is None:
            block = slice(None)
        targets = self.data.cy[:, :, block]
        if self.shared_features is None:
            return RidgeCoefficients(local=self.local_map @ targets)
        if not np.isinf(self.linear_reg):
            return self._hybrid_coefficients(block)
        shared = torch.linalg.solve(
            self.shared_gram, self.project_targets(self.shared_map, targets)
        )
        local = self.local_map @ targets - self.coupling @ shared
        return RidgeCoefficients(local, shared, targets.new_zeros(targets.shape[-1]))

    def prediction_sse(self, scored: torch.Tensor | None = None) -> torch.Tensor:
        """Return the weighted sum of squared prediction errors, averaged over outputs.

        For rows i in the scored environments:

            error = sum_i c[i] * sum_j (y_hat[i, j] - y[i, j])**2 / J

        Parameters
        ----------
        scored : torch.Tensor of shape (D,) or None, default=None
            Boolean mask selecting environments to score. None scores all of them.

        Returns
        -------
        torch.Tensor of shape ()
            Prediction error without the ridge penalty. Divide by the total
            count in the scored rows to get mean squared error.
        """
        if scored is None:
            scored = torch.ones(self.data.c.shape[1], dtype=torch.bool, device=self.data.c.device)
        if self.shared_features is None:
            return self.grouped_sse(self.data, self.features, self.local_map, scored)
        if np.isinf(self.linear_reg):
            return self.batched_shared_sse(self.data, [self.shared_loss_terms(scored)])
        data = self.data
        local, shared, linear = self.coefficients()
        effect = self.shared_features @ shared + data.control_mean * linear
        counts = data.c * scored
        target_sum = (self.local_features.T * scored) @ data.cy
        cross = (local * target_sum).sum() + (effect * data.pooled_targets * scored[:, None]).sum()
        local_gram = self.grouped_gram(counts, self.local_features)
        mixed = (self.local_features.T * counts[:, None, :]) @ effect
        quadratic = (local * (local_gram @ local)).sum() + 2 * (local * mixed).sum()
        quadratic = quadratic + (counts.sum(dim=0)[:, None] * effect.square()).sum()
        return data.cyy[scored].sum() + (quadratic - 2 * cross) / data.cy.shape[-1]

    def shared_loss_terms(self, scored: torch.Tensor) -> _SharedLossTerms:
        """Prepare the shared model's error terms for one split.

        With shared coefficients b, the error is

            error = local_sse + (2 * sum(b * cross) + sum(b * (score_gram @ b))) / J

        Here, cross = sum_x cross_map[x] @ cy[x]. Keeping the maps lets us
        combine target projections across splits in ``batched_shared_sse``.

        Parameters
        ----------
        scored : torch.Tensor of shape (D,)
            Boolean mask selecting environments to score.

        Returns
        -------
        _SharedLossTerms
            Small matrices needed to fit and score the shared coefficients.
            Used when there is no control slope; the hybrid scores each output
            with its own fitted slope instead.
        """
        local = self.local_features
        residual_shared = self.shared_features - local @ self.coupling
        counts = self.data.c * scored
        cross_map = (
            residual_shared.transpose(1, 2) @ (counts[:, :, None] * local)
        ) @ self.local_map
        cross_map = cross_map - (residual_shared * scored[None, :, None]).transpose(1, 2)
        score_gram = (residual_shared.transpose(1, 2) @ (counts[:, :, None] * residual_shared)).sum(
            dim=0
        )
        return _SharedLossTerms(
            self.grouped_sse(self.data, local, self.local_map, scored),
            self.shared_gram,
            score_gram,
            self.shared_map,
            cross_map,
        )

    def _hybrid_coefficients(self, block: slice) -> RidgeCoefficients:
        """Fit the hybrid's category, shared and linear coefficients together.

        Account for category and shared effects first. Each output then needs
        only a scalar division for its control slope. All outputs use the same
        q-by-q matrix for their shared coefficients.

        Parameters
        ----------
        block : slice
            Outputs to include. Let B be the number of selected outputs.

        Returns
        -------
        local : torch.Tensor of shape (G, q+1, B)
            Category intercepts in row zero, followed by deviation coefficients.
        shared : torch.Tensor of shape (q, B)
            Coefficients shared across categories.
        linear : torch.Tensor of shape (B,)
            One control slope per selected output.
        """
        data = self.data
        targets = data.cy[:, :, block]
        raw_mean = data.control_mean[:, block]
        pooled_targets = data.pooled_targets[:, block]
        local_zero = self.local_map @ targets
        local_raw = (self.local_map * data.c[:, None, :]) @ raw_mean
        raw_cross = (self.local_features.T * self.counts[:, None, :]) @ raw_mean
        shared_rhs = self.project_targets(self.shared_map, targets)
        shared_raw = (self.shared_map * data.c[:, None, :]).sum(dim=0) @ raw_mean
        raw_rhs = (raw_mean * pooled_targets * self.fit[:, None]).sum(dim=0)
        raw_rhs = raw_rhs - (raw_cross * local_zero).sum(dim=(0, 1))
        raw_gram = (self.counts.sum(dim=0)[:, None] * raw_mean.square()).sum(dim=0)
        raw_gram = raw_gram - (raw_cross * local_raw).sum(dim=(0, 1))
        shared_zero, shared_raw_map = torch.linalg.solve(
            self.shared_gram, torch.cat((shared_rhs, shared_raw), dim=1)
        ).chunk(2, dim=1)
        raw_diagonal = raw_gram + self.linear_reg * self.total
        denominator = raw_diagonal - (shared_raw * shared_raw_map).sum(dim=0)
        numerator = raw_rhs - (shared_raw * shared_zero).sum(dim=0)
        linear = self._linear_coefficients(numerator, denominator, raw_diagonal)
        shared = shared_zero - shared_raw_map * linear
        local = local_zero - self.coupling @ shared - local_raw * linear
        return RidgeCoefficients(local, shared, linear)

    @staticmethod
    def _linear_coefficients(numerator, denominator, scale) -> torch.Tensor:
        """Divide to get control slopes; use zero when a denominator is too small.

        Parameters
        ----------
        numerator : torch.Tensor of shape (B,)
            Target-control products after accounting for category and shared effects.
        denominator : torch.Tensor of shape (B,)
            Remaining control variation plus the ridge penalty.
        scale : torch.Tensor of shape (B,)
            Control variation plus penalty before the shared-effect adjustment.
            Sets the tolerance for floating-point rounding errors.

        Returns
        -------
        torch.Tensor of shape (B,)
            numerator / denominator where the denominator exceeds the tolerance;
            zero elsewhere.

        Raises
        ------
        FloatingPointError
            If a denominator is negative by more than the rounding tolerance.
        """
        tolerance = 100 * torch.finfo(denominator.dtype).eps * scale.abs().clamp_min(1)
        if (denominator < -tolerance).any():
            raise FloatingPointError("Negative raw-linear Schur complement")
        supported = denominator > tolerance
        safe_denominator = torch.where(supported, denominator, 1.0)
        return torch.where(supported, numerator / safe_denominator, 0.0)
