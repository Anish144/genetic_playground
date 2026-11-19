"""Tensor records shared by neural bridges and their ridge solver.

G counts perturbations, D counts environments and J counts measured outputs.
Each environment has m_e control samples with d_w input coordinates.
"""

from typing import NamedTuple

import torch


class GroupedRidgeData(NamedTuple):
    """Counts, target sums and optional controls for a neural ridge fit.

    Missing category/environment pairs have zero count. All tensors use the
    same device and precision. The record stores references without copying.

    Attributes
    ----------
    c : torch.Tensor of shape (G, D)
        Sum of response counts for each category and environment.
    cy : torch.Tensor of shape (G, D, J)
        Sum of count * target for each category and environment.
    cyy : torch.Tensor of shape (D,)
        Sum of count * squared target, divided by J, for each environment.
        Square each row's targets before adding rows from the same pair.
    target_gram : torch.Tensor of shape (G, D, D)
        Q[x] = cy[x] @ cy[x].T / J for each category x.
    w_envs : list of torch.Tensor or None
        Control samples; entry e has shape (m_e, d_w). Neural bridges need
        these samples to learn features. A standalone ridge fit may omit them.
    control_mean : torch.Tensor of shape (D, J) or None
        Mean controls used by the hybrid's linear term. May already be scaled.
    pooled_targets : torch.Tensor of shape (D, J) or None
        cy.sum(0), used by the hybrid's linear term.

    Examples
    --------
    One category, two environments, one output; the second row has twice the weight.

    >>> import torch
    >>> from genetic_perturbation_playground.data.neural import GroupedRidgeData
    >>> counts = torch.tensor([[1., 2.]], dtype=torch.float64)
    >>> targets = torch.tensor([[[2.], [4.]]], dtype=torch.float64)
    >>> cy = counts[:, :, None] * targets
    >>> data = GroupedRidgeData(
    ...     c=counts,
    ...     cy=cy,
    ...     cyy=(counts * targets.square().mean(dim=2)).sum(dim=0),
    ...     target_gram=cy @ cy.transpose(1, 2) / targets.shape[-1],
    ... )
    >>> data.cy.tolist()
    [[[2.0], [8.0]]]
    >>> data._replace(pooled_targets=cy.sum(0)).pooled_targets.shape
    torch.Size([2, 1])
    """

    c: torch.Tensor
    cy: torch.Tensor
    cyy: torch.Tensor
    target_gram: torch.Tensor
    w_envs: list[torch.Tensor] | None = None
    control_mean: torch.Tensor | None = None
    pooled_targets: torch.Tensor | None = None
