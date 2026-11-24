"""Ridge fits for linear and neural transfer models.

Notation
--------
G : number of categories
    Distinct perturbations, such as gene knockdowns or siRNAs.
D : number of environments
    Source batches or experiments.
q : number of learned features
    Output size of each control network, set by n_features_w.
q_s : number of shared features
    Equal to q for the shared model and q + r for the hybrid, where
    r <= min(D, d_w) is the numerical rank of the training control means.
    Disabling the hybrid's linear term sets r = 0.
d_w : number of input coordinates per control sample
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
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LinearRegression, Ridge

from genetic_perturbation_playground.data.neural import GroupedRidgeData


def fit_linear_slopes(stats: dict, reg: float = 1e-3) -> np.ndarray:
    """Fit diagonal ridge slopes with unpenalised category intercepts.

    Subtract category-weighted means implicitly using summary statistics.
    Each slope is the weighted control/target covariance divided by the
    weighted control variance plus N * reg, where N is total response count.

    Parameters
    ----------
    stats : dict
        Output of data.transfer.summary_stats. Uses y_mean, w_mean, n, W,
        batch_y_sum and batch_n. Controls may already be standardised.
        Inputs are unchanged.
    reg : float, default=1e-3
        Finite, nonnegative slope penalty in the supplied control units.
        Zero gives ordinary least squares.

    Returns
    -------
    numpy.ndarray of shape (J,)
        Float64 slope per output. Zero variance with reg=0 gives zero slope.
        Recover category intercepts as y_mean - slopes * w_mean.

    Raises
    ------
    ValueError
        If reg is invalid or total response weight is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from genetic_perturbation_playground.utils.ridge import fit_linear_slopes
    >>> stats = {
    ...     "n": np.array([2]),
    ...     "y_mean": np.array([[3.]]),
    ...     "w_mean": np.array([[1.]]),
    ...     "W": np.array([[0.], [2.]]),
    ...     "batch_n": np.array([1, 1]),
    ...     "batch_y_sum": np.array([[1.], [5.]]),
    ... }
    >>> fit_linear_slopes(stats, reg=1.0).tolist()
    [1.0]
    >>> fit_linear_slopes(stats, reg=0.0).tolist()
    [2.0]
    """
    if not np.isfinite(reg) or reg < 0:
        raise ValueError("reg must be finite and nonnegative")
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
    denominator = np.maximum(variance, 0.0) + total * reg
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def fit_full_linear_slopes(stats: dict, reg: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Fit factorised full-matrix slopes with sklearn linear regression.

    Remove unpenalised category intercepts in batch space and project controls
    onto an orthonormal basis before fitting. This retains the full regression
    objective without a feature-by-feature or observed-pair-by-feature matrix.

    Parameters
    ----------
    stats : dict
        Output of data.transfer.summary_stats, with optionally scaled W.
        Uses n, pert_batch_n, y_mean, batch_y_sum and W. Inputs are unchanged.
    reg : float, default=1e-3
        Finite, nonnegative ridge penalty, scaled by total response count.
        Zero uses LinearRegression to obtain minimum-norm slopes; positive
        values use Ridge. Both fit without an intercept after centring.

    Returns
    -------
    factors : tuple of numpy.ndarray
        (basis, response), both of shape (rank, J), in float64. For row
        controls w, the fitted effect is (w @ basis.T) @ response.
        Rank is at most the number of training batches minus one.
        Zero within-category variation gives factors with zero rows.

    Raises
    ------
    ValueError
        If reg is invalid or total response weight is not positive.
    """
    if not np.isfinite(reg) or reg < 0:
        raise ValueError("reg must be finite and nonnegative")
    total = np.asarray(stats["n"], dtype=np.float64).sum()
    if total <= 0:
        raise ValueError("A linear ridge fit needs positive total response weight")
    counts = np.asarray(stats["pert_batch_n"], dtype=np.float64) / total
    category_weight = counts.sum(axis=1)
    batch_weight = counts.sum(axis=0)
    controls = np.asarray(stats["W"], dtype=np.float64)
    targets = np.asarray(stats["y_mean"], dtype=np.float64)
    # Remove global target offsets before forming category residuals.
    target_origin = category_weight @ targets
    target_sums = np.asarray(stats["batch_y_sum"], dtype=np.float64) / total
    residual_sums = (
        target_sums - batch_weight[:, None] * target_origin - counts.T @ (targets - target_origin)
    )
    # H is a graph Laplacian. Sum its off-diagonal weights to form the
    # diagonal without cancellation when categories occupy one batch.
    gram = -(counts.T / category_weight) @ counts
    np.fill_diagonal(gram, 0.0)
    np.fill_diagonal(gram, -gram.sum(axis=1))
    # Perturbations never span disconnected components. Their intercepts
    # absorb a separate control origin in each component. Singleton and
    # control-only batches become exactly zero and cannot affect rank.
    _, components = connected_components(gram, directed=False)
    _, origins = np.unique(components, return_index=True)
    controls = controls - controls[origins[components]]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(np.float64).eps * len(gram) * eigenvalues.max(initial=0.0)
    keep = eigenvalues > tolerance
    roots = np.sqrt(eigenvalues[keep])
    vectors = eigenvectors[:, keep]
    if not len(roots):
        return np.empty((0, controls.shape[1])), np.empty((0, targets.shape[1]))
    design = roots[:, None] * (vectors.T @ controls)
    target = (vectors.T @ residual_sums) / roots[:, None]
    left, singular, basis = np.linalg.svd(design, full_matrices=False)
    # A control constant within each disconnected category component is
    # annihilated by H. Judge rounding against the input scale, since the
    # entire projected design can otherwise consist of numerical noise.
    input_scale = roots.max() * np.linalg.norm(controls)
    tolerance = np.finfo(np.float64).eps * max(controls.shape) * input_scale
    keep = singular > tolerance
    if not np.any(keep):
        return basis[keep], np.empty((0, targets.shape[1]))
    # Both loss and count weights were divided by N above, so alpha=reg.
    # The orthonormal basis preserves the coefficient norm in the penalty.
    design = np.diag(singular[keep])
    target = left[:, keep].T @ target
    model = (
        LinearRegression(fit_intercept=False)
        if reg == 0
        else Ridge(alpha=reg, fit_intercept=False, solver="cholesky")
    )
    response = np.atleast_2d(model.fit(design, target).coef_).T
    return basis[keep], response


class _SharedLossTerms(NamedTuple):
    """Terms needed to score one fit of the shared model.

    Attributes
    ----------
    local_sse : torch.Tensor of shape ()
        Error from the category fit alone, averaged over outputs.
    shared_gram : torch.Tensor of shape (q_s, q_s)
        Matrix for fitting the shared coefficients, including their penalty.
    score_gram : torch.Tensor of shape (q_s, q_s)
        Matrix for the squared shared prediction on the scored rows.
    fit_map : torch.Tensor of shape (G, q_s, D)
        Shared coefficients b solve shared_gram @ b = sum_x fit_map[x] @ cy[x].
    cross_map : torch.Tensor of shape (G, q_s, D)
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
    shared : torch.Tensor of shape (q_s, J) or None
        Coefficients shared across categories; None for per-category fits.
        Hybrid fits append coefficients for the projected control means after
        the q learned feature rows when the linear term is enabled.

    Examples
    --------
    Per-category fits store only local coefficients. This example has two
    categories, three features and one output.

    >>> import torch
    >>> from genetic_perturbation_playground.utils.ridge import RidgeCoefficients
    >>> coefficients = RidgeCoefficients(local=torch.ones(2, 3, 1))
    >>> coefficients.local.shape
    torch.Size([2, 3, 1])
    >>> coefficients.shared is None
    True
    """

    local: torch.Tensor
    shared: torch.Tensor | None = None


class NeuralRidgeSolver:
    """Fit coefficients and score predictions for fixed features.

    Create a solver with ``per_category`` or ``shared_deviation``, then call
    ``coefficients``, ``prediction_sse`` or ``penalty``. Fits support gradients
    through the features.

    Parameters
    ----------
    data : GroupedRidgeData
        Counts and target sums used by the fit.
    features : torch.Tensor
        Feature rows, shape (D, q) for per-category fits or (D, q_s + q) for
        shared/hybrid fits, with shared features before deviation features.
    fit : torch.Tensor of shape (D,) or None
        Boolean mask selecting training environments; None includes all of
        them. The factory methods prepare the coefficient maps for this split.

    Attributes
    ----------
    data : GroupedRidgeData
        Counts and target sums used by the fit.
    features : torch.Tensor
        Input features, shape (D, q) for per-category fits or (D, q_s + q) for
        shared/hybrid fits.
    fit : torch.Tensor of shape (D,) or None
        Boolean mask selecting training environments. None includes all of them.
    local_map : torch.Tensor
        Maps target sums to category coefficients before adding shared effects.
        Shape (G, q, D) for per-category fits, (G, q+1, D) for shared/hybrid fits.
    shared_features : torch.Tensor of shape (D, q_s) or None
        Shared features; None for per-category fits.
    local_features : torch.Tensor of shape (D, q+1)
        Intercept column followed by deviation features. Shared/hybrid fits only.
    counts : torch.Tensor of shape (G, D)
        Counts in training environments. Shared/hybrid fits only.
    total : torch.Tensor of shape ()
        Sum of training counts across all categories.
    local_reg : float or torch.Tensor of shape (q+1,)
        Penalty before multiplication by N. A scalar for per-category fits;
        intercept and deviation penalties for shared/hybrid fits.
    shared_reg : torch.Tensor of shape (q_s,)
        Penalty per shared feature before multiplication by N.
        Shared/hybrid fits only.
    coupling : torch.Tensor of shape (G, q+1, q_s)
        How category coefficients change with shared coefficients.
        Shared/hybrid fits only.
    shared_gram : torch.Tensor of shape (q_s, q_s)
        Matrix for the shared solve after accounting for category effects.
        Shared/hybrid fits only.
    shared_map : torch.Tensor of shape (G, q_s, D)
        Maps target sums to the shared solve's target term. Shared/hybrid fits only.

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
    >>> solver.penalty().item()
    4.5
    """

    def __init__(self, data: GroupedRidgeData, features: torch.Tensor, fit: torch.Tensor | None):
        self.data = data
        self.features = features
        self.fit = fit
        self.shared_features = None

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
        solver.total, solver.local_reg = counts.sum(), reg
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
        shared_reg,
    ):
        """Fit shared effects and category deviations together.

        Parameters
        ----------
        data : GroupedRidgeData
            Counts and target sums.
        embeddings : torch.Tensor of shape (D, q_s + q)
            Shared features followed by deviation features. For hybrid fits,
            the shared block includes projected control means after the
            learned shared features.
        fit : torch.Tensor of shape (D,) or None, default=None
            Boolean training mask; None includes all environments. Selected
            environments must have positive total response weight.
        reg : float
            Finite, positive intercept penalty, before multiplication by N.
        deviation_reg : float
            Finite, positive deviation penalty, before multiplication by N.
        shared_reg : torch.Tensor of shape (q_s,)
            Finite, positive penalty per shared feature, before multiplication
            by N. Its length sets the shared width. Use the same device and
            floating-point precision as embeddings.

        Returns
        -------
        NeuralRidgeSolver
            Solver ready to compute coefficients or prediction errors.

        Raises
        ------
        ValueError
            If the selected environments have no positive total response weight.

        Notes
        -----
        For shared features s, deviation features d and category intercepts a,
        fit all coefficients jointly to minimise::

            y_hat[x, e] = a[x] + s[e] @ b + d[e] @ v[x]
            penalty = N * (reg * ||a||^2 + sum_k shared_reg[k] * ||b[k]||^2
                           + deviation_reg * ||v||^2)
            loss = J * SSE + penalty

        SSE is the count-weighted sum of squared errors, averaged over outputs.
        N is the total response weight in the selected training environments.
        """
        if fit is None:
            fit = torch.ones(len(embeddings), dtype=torch.bool, device=embeddings.device)
        solver = cls(data, embeddings, fit)
        width = len(shared_reg)
        shared, deviation = embeddings.split((width, embeddings.shape[1] - width), dim=1)
        solver.shared_features = shared
        solver.local_features = torch.cat((torch.ones_like(deviation[:, :1]), deviation), dim=1)
        solver.counts = data.c * fit
        solver.total = solver.counts.sum()
        if solver.total <= 0:
            raise ValueError("Each ridge fit needs positive total response weight")
        solver._eliminate_local_effects(reg, deviation_reg, shared_reg)
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
        maps : torch.Tensor of shape (G, n_features, D)
            Maps from environments to features for each category.
        targets : torch.Tensor of shape (G, D, J)
            Weighted target sums. May contain only a slice of the outputs.

        Returns
        -------
        torch.Tensor of shape (n_features, J)
            sum_x maps[x] @ targets[x], computed without a temporary
            (G, n_features, J) tensor or a copy of targets.
        """
        weights = maps.permute(1, 0, 2).reshape(maps.shape[1], -1)
        return weights @ targets.view(-1, targets.shape[-1])

    @staticmethod
    def batched_shared_sse(data: GroupedRidgeData, terms: list[_SharedLossTerms]) -> torch.Tensor:
        """Sum prediction errors across splits with one target projection.

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

    def _eliminate_local_effects(self, reg: float, deviation_reg: float, shared_reg) -> None:
        """Express category coefficients in terms of shared coefficients.

        The category equations give

            local = local_map @ cy - coupling @ shared

        Substitution leaves a q_s-by-q_s solve. Store the coefficient maps,
        penalties and shared system on this solver.
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
        self.local_reg, self.shared_reg = penalties, shared_reg
        # local_map: (G, q+1, D); coupling: (G, q+1, q_s).
        gram = shared.T @ (self.counts.sum(dim=0)[:, None] * shared)
        gram = gram - (cross.transpose(1, 2) @ self.coupling).sum(dim=0)
        self.shared_gram = self.penalised_gram(gram, self.total, shared_reg)
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
            Category and shared coefficients for these outputs.
            Gradients can pass through the fit.
        """
        if block is None:
            block = slice(None)
        targets = self.data.cy[:, :, block]
        if self.shared_features is None:
            return RidgeCoefficients(local=self.local_map @ targets)
        shared = torch.linalg.solve(
            self.shared_gram, self.project_targets(self.shared_map, targets)
        )
        local = self.local_map @ targets - self.coupling @ shared
        return RidgeCoefficients(local, shared)

    def penalty(self) -> torch.Tensor:
        """Return the fitted ridge penalty averaged over outputs.

        Returns
        -------
        torch.Tensor of shape ()
            N * sum_k reg_k * ||coef_k||^2 / J, where k indexes coefficient
            rows and N is the total training weight. Gradients pass through
            the fitted coefficients.

        Notes
        -----
        Adding this to ``prediction_sse(self.fit)`` gives the fitted ridge
        objective divided by J. Divide by N again to obtain the penalized
        mean squared error. Shared and hybrid fits reconstruct coefficients
        for all outputs to evaluate this penalty.
        """
        if self.shared_features is None:  # ||R @ cy||^2 / J from target products; no cy needed
            R = self.local_map
            return self.total * self.local_reg * (R * (R @ self.data.target_gram)).sum()
        local, shared = self.coefficients()
        penalty = (self.local_reg * local.square().sum(dim=(0, 2))).sum()
        penalty = penalty + (self.shared_reg * shared.square().sum(dim=1)).sum()
        return self.total * penalty / self.data.cy.shape[-1]

    def prediction_sse(self, scored: torch.Tensor | None = None) -> torch.Tensor:
        """Return count-weighted squared errors averaged over outputs.

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
        return self.batched_shared_sse(self.data, [self.shared_loss_terms(scored)])

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
            Matrices needed to fit and score the shared coefficients. Shared
            feature matrices have width q_s, including projected controls.
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
