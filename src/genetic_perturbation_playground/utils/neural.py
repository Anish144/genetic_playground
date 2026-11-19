"""Neural layers, tensor conversion and checkpointed loss evaluation.

Networks use float32 weights and return float64 features for ridge fitting.
D counts environments, q counts features from one control network, and J counts
outputs. Shape prefixes marked ... are preserved by each layer.
"""

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class SwiGLU(nn.Module):
    """Apply a learned gate to a linear transformation.

    For input h, return silu(gate(h)) * value(h). The product is elementwise.

    Parameters
    ----------
    d_in : int
        Number of input coordinates.
    d_out : int
        Number of output coordinates.

    Attributes
    ----------
    gate : torch.nn.Linear
        Linear layer whose output passes through the SiLU activation.
    value : torch.nn.Linear
        Linear layer multiplied by the gate.

    Examples
    --------
    >>> import torch
    >>> from genetic_perturbation_playground.utils.neural import SwiGLU
    >>> layer = SwiGLU(3, 2)
    >>> layer(torch.ones(4, 3)).shape
    torch.Size([4, 2])
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.gate = nn.Linear(d_in, d_out)
        self.value = nn.Linear(d_in, d_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the learned gate and value layers.

        Parameters
        ----------
        h : torch.Tensor of shape (..., d_in)
            Input values.

        Returns
        -------
        torch.Tensor of shape (..., d_out)
            Transformed values.
        """
        return nn.functional.silu(self.gate(h)) * self.value(h)


class Residual(nn.Module):
    """Add a layer's output to its input: h + layer(h).

    Parameters
    ----------
    layer : torch.nn.Module
        Layer that preserves the input shape.

    Attributes
    ----------
    layer : torch.nn.Module
        Wrapped layer.

    Examples
    --------
    >>> import torch
    >>> from genetic_perturbation_playground.utils.neural import Residual
    >>> layer = Residual(torch.nn.Identity())
    >>> layer(torch.tensor([[1., 2.]])).tolist()
    [[2.0, 4.0]]
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Add the layer output to its input.

        Parameters
        ----------
        h : torch.Tensor of shape (..., d)
            Input values.

        Returns
        -------
        torch.Tensor of shape (..., d)
            Transformed values.
        """
        return h + self.layer(h)


class Float32Network(nn.Module):
    """Run a network in float32 and return float64 features for ridge fitting.

    Parameters
    ----------
    net : torch.nn.Module
        Network with float32 weights.

    Attributes
    ----------
    net : torch.nn.Module
        Wrapped network.

    Examples
    --------
    >>> import torch
    >>> from genetic_perturbation_playground.utils.neural import Float32Network
    >>> network = Float32Network(torch.nn.Linear(3, 2))
    >>> network(torch.ones(4, 3, dtype=torch.float64)).dtype
    torch.float64
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Run the wrapped network in float32.

        Parameters
        ----------
        h : torch.Tensor of shape (..., d_in)
            Input values.

        Returns
        -------
        torch.Tensor of shape (..., d_out)
            Network output converted to float64.
        """
        return self.net(h.float()).double()


def control_mlp(d_in: int, width: int, depth: int, d_out: int) -> Float32Network:
    """Build a control network with SwiGLU hidden layers.

    Hidden layers that preserve width add their input to their output.

    Parameters
    ----------
    d_in : int
        Number of input coordinates.
    width : int
        Units per hidden layer.
    depth : int
        Number of hidden layers.
    d_out : int
        Number of output features.

    Returns
    -------
    Float32Network
        Network with float32 weights and float64 inputs and outputs.
    """
    dims = [d_in] + [width] * depth
    hidden = [Residual(SwiGLU(a, b)) if a == b else SwiGLU(a, b) for a, b in pairwise(dims)]
    return Float32Network(nn.Sequential(*hidden, nn.Linear(dims[-1], d_out)))


def as_tensor(a: np.ndarray, device: str) -> torch.Tensor:
    """Convert an array to a two-dimensional float64 tensor.

    Parameters
    ----------
    a : numpy.ndarray of shape (n,) or (n, d)
        Input rows.
    device : str
        PyTorch device for the tensor.

    Returns
    -------
    torch.Tensor of shape (n, d)
        Input values on the requested device. One-dimensional input gets one column.
    """
    return torch.as_tensor(np.asarray(a, dtype=np.float64), device=device).reshape(len(a), -1)


def checkpointed_sse(score_split: Callable, data, embeddings, folds) -> torch.Tensor:
    """Sum fold errors while keeping fewer tensors for the backward pass.

    PyTorch recomputes each fold's ridge calculations when evaluating gradients.

    Parameters
    ----------
    score_split : callable
        Function taking data, embeddings, a training mask and a scoring mask.
    data : object
        Training data passed unchanged to score_split.
    embeddings : torch.Tensor
        Environment features: (D, q) for per-category fits, (D, 2q) for shared/hybrid fits.
    folds : torch.Tensor of shape (D,)
        Integer fold label for each environment, starting at zero.

    Returns
    -------
    torch.Tensor of shape ()
        Sum of squared errors across folds, weighted by counts and divided by J.
    """
    return sum(
        checkpoint(score_split, data, embeddings, folds != k, folds == k, use_reentrant=False)
        for k in range(int(folds.max()) + 1)
    )
