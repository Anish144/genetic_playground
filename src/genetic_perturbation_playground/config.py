"""Default settings for transfer models, separate from fitted model state.

Dataset presets override these defaults through Hydra. Importing this module
does not register configs or import the models.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class LinearConfig:
    """Settings for linear transfer.

    Parameters
    ----------
    slope : float or None, default=None
        None fits the slope; a number fixes it in original input units.
    standardise_w : bool, default=True
        Scale controls with training-control statistics.
    reg : float, default=1e-3
        Nonnegative slope penalty. Zero gives ordinary least squares.

    Attributes
    ----------
    slope, standardise_w, reg
        The supplied parameter values; no fitted state is stored here.

    Examples
    --------
    >>> from dataclasses import asdict
    >>> from genetic_perturbation_playground.model.linear_transfer import LinearTransfer
    >>> model = LinearTransfer(**asdict(LinearConfig(reg=0)))
    >>> model.reg
    0
    """

    slope: float | None = None
    standardise_w: bool = True
    reg: float = 1e-3


@dataclass
class KernelConfig:
    """Settings for the kernel transfer bridge.

    Parameters
    ----------
    reg : float, default=1e-6
        Ridge penalty, scaled by total training weight during fitting.
    weight_by_cells : bool, default=True
        Weight each observed pair by its cell count.
    lengthscale_scale : float, default=1.0
        Multiplier on the median-distance kernel lengthscale.

    Attributes
    ----------
    reg, weight_by_cells, lengthscale_scale
        The supplied parameter values.

    Examples
    --------
    >>> KernelConfig(reg=0.01).lengthscale_scale
    1.0
    """

    reg: float = 1e-6
    weight_by_cells: bool = True
    lengthscale_scale: float = 1.0


@dataclass
class PerXConfig:
    """Settings for a neural bridge with a separate response per perturbation.

    Parameters
    ----------
    variant : str, default="per_x"
        Model identifier. Select the corresponding config class to change it.
    weight_by_cells : bool, default=True
        Weight each observed pair by its cell count.
    n_features_w : int, default=16
        Output dimension of each control network.
    width_w : int, default=64
        Units per hidden layer.
    depth : int, default=1
        Hidden layers per network, excluding the output layer. Zero is linear.
    reg : float, default=1e-3
        Coefficient ridge penalty, scaled by total training weight.
    lr : float, default=1e-3
        AdamW learning rate.
    weight_decay : float, default=0.0
        AdamW penalty on network weights.
    n_steps : int, default=2000
        Full-batch network updates.
    cross_fit_folds : int, default=0
        Zero scores in-sample; at least two uses held-out environments.
    seed : int, default=0
        Network and fold-draw seed.
    device : str, default="cpu"
        PyTorch device. Ridge calculations require float64 support.
    log_key : str or None, default=None
        Loss key in an existing W&B run. None disables loss logging.
    gene_block_size : int, default=256
        Outputs processed together when computing final coefficients.

    Attributes
    ----------
    variant, weight_by_cells, n_features_w, width_w, depth, reg, lr
        Model, network and ridge settings supplied above.
    weight_decay, n_steps, cross_fit_folds, seed, device, log_key, gene_block_size
        Training and execution settings supplied above.

    Examples
    --------
    >>> settings = PerXConfig(width_w=32, n_steps=100)
    >>> settings.depth, settings.device
    (1, 'cpu')
    """

    variant: str = "per_x"
    weight_by_cells: bool = True
    n_features_w: int = 16
    width_w: int = 64
    depth: int = 1
    reg: float = 1e-3
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_steps: int = 2000
    cross_fit_folds: int = 0
    seed: int = 0
    device: str = "cpu"
    log_key: str | None = None
    gene_block_size: int = 256


@dataclass
class Shared2NNConfig(PerXConfig):
    """Settings for shared and per-perturbation control networks.

    Network and training parameters follow PerXConfig and apply to both networks.

    Parameters
    ----------
    variant : str, default="shared_2nn"
        Model identifier for this config class.
    deviation_reg : float or None, default=None
        Positive deviation penalty. None uses the final model's reg, including
        any tuning override. Both penalties must be finite and positive.

    Attributes
    ----------
    variant, deviation_reg
        The supplied values, alongside the inherited settings. None remains
        unset until model construction so it can follow a trial's ridge penalty.

    Examples
    --------
    >>> Shared2NNConfig(depth=1).variant
    'shared_2nn'
    """

    variant: str = "shared_2nn"
    deviation_reg: float | None = None


@dataclass
class HybridConfig(Shared2NNConfig):
    """Settings for shared neural effects with a linear control term.

    Other parameters follow Shared2NNConfig.

    Parameters
    ----------
    variant : str, default="hybrid"
        Model identifier for this config class.
    linear_reg : float, default=1e-3
        Nonnegative control-slope penalty. Zero leaves slopes unpenalised;
        infinity disables the linear term.

    Attributes
    ----------
    variant, linear_reg
        The supplied values, alongside the inherited settings.

    Examples
    --------
    >>> HybridConfig(linear_reg=0).variant
    'hybrid'
    """

    variant: str = "hybrid"
    linear_reg: float = 1e-3


_CONFIG_GROUPS = {
    "linear": (LinearConfig,),
    "kernel": (KernelConfig,),
    "neural": (PerXConfig, Shared2NNConfig, HybridConfig),
}


def register_configs() -> None:
    """Register model settings before composing a transfer experiment.

    Repeated calls register the same schemas. Importing this module alone does
    not change Hydra's config store.

    Returns
    -------
    None
    """
    from hydra.core.config_store import ConfigStore

    store = ConfigStore.instance()
    for group, configs in _CONFIG_GROUPS.items():
        for config in configs:
            name = config.variant if group == "neural" else "default"
            store.store(group=group, name=name, node=config)


def resolve_model_config(
    cfg: DictConfig, name: str, overrides: dict[str, Any] | None = None
) -> LinearConfig | KernelConfig | PerXConfig:
    """Apply model overrides and return an independent settings dataclass.

    Merge the whole experiment before resolving references such as
    ``${neural.reg}``, so they refer to the trial's settings. Reject undeclared
    fields before conversion: OmegaConf can otherwise attach extra attributes
    that dataclasses.asdict would omit.

    Parameters
    ----------
    cfg : DictConfig
        Composed experiment containing registered model config groups.
    name : {"linear", "kernel", "neural"}
        Model node to resolve.
    overrides : dict or None, default=None
        Trial or fold settings to apply without changing cfg.

    Returns
    -------
    LinearConfig, KernelConfig, PerXConfig, Shared2NNConfig or HybridConfig
        Resolved settings for one model.
    """
    from omegaconf import OmegaConf

    if name not in _CONFIG_GROUPS:
        raise ValueError(f"Unknown model config {name!r}; choose from {tuple(_CONFIG_GROUPS)}")
    schema = OmegaConf.get_type(cfg[name])
    if schema not in _CONFIG_GROUPS[name]:
        raise TypeError(f"Compose the registered {name} config group before resolving it")
    overrides = {} if overrides is None else overrides
    unknown = (set(cfg[name]) | set(overrides)) - {field.name for field in fields(schema)}
    if unknown:
        raise ValueError(f"Unknown {name} settings: {', '.join(sorted(unknown))}")
    merged = OmegaConf.merge(cfg, {name: overrides})
    config = OmegaConf.to_object(merged[name])
    if name == "neural" and config.variant != schema.variant:
        raise ValueError(f"Select the matching config class or Hydra group neural={config.variant}")
    return config
