import importlib
import inspect
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from genetic_perturbation_playground.model.batch_transfer import (
    KernelBridgeTransfer,
    NeuralBridgeTransfer,
)
from genetic_perturbation_playground.model.linear_transfer import LinearTransfer
from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralHybridEmbeddingBridge,
    NeuralPerXEmbeddingBridge,
    NeuralSharedEmbeddingBridge,
)
from genetic_perturbation_playground.utils.neural import SwiGLU
from genetic_perturbation_playground.utils.standardisation import Standardised


@pytest.fixture
def cells():
    rng = np.random.default_rng(12)
    rows, values = [], []
    for batch in range(4):
        offset = rng.normal(size=3)
        for label, effect in [("control", 0), ("a", 1), ("b", -1)]:
            rows.extend([(label, f"HUVEC-{batch}")] * 4)
            values.append(rng.normal(size=(4, 3)) + offset + effect)
    obs = pd.DataFrame(rows, columns=["perturbation", "batch"])
    obs.index = obs.index.astype(str)
    obs["well_type"] = np.where(obs.perturbation == "control", "negative_control", "treatment")
    return ad.AnnData(np.vstack(values), obs=obs)


@pytest.mark.parametrize("variant,networks", [("per_x", 1), ("shared_2nn", 2), ("hybrid", 2)])
@pytest.mark.parametrize("depth", [None, 2])
def test_neural_constructor_defaults_have_one_hidden_layer(cells, variant, networks, depth):
    overrides = {} if depth is None else {"depth": depth}
    model = NeuralBridgeTransfer(
        variant=variant, n_steps=1, width_w=4, n_features_w=2, **overrides
    ).fit(cells)
    assert sum(isinstance(layer, SwiGLU) for layer in model.bridge_.psi_.modules()) == networks * (
        1 if depth is None else depth
    )


@pytest.mark.parametrize("standardise_w", [False, True])
@pytest.mark.parametrize("variant", ["per_x", "shared_2nn", "hybrid"])
def test_typed_neural_builder_preserves_predictions(cells, standardise_w, variant):
    from genetic_perturbation_playground.config import HybridConfig, PerXConfig, Shared2NNConfig
    from genetic_perturbation_playground.model.batch_transfer import build_neural

    config_type, model_type = {
        "per_x": (PerXConfig, NeuralPerXEmbeddingBridge),
        "shared_2nn": (Shared2NNConfig, NeuralSharedEmbeddingBridge),
        "hybrid": (HybridConfig, NeuralHybridEmbeddingBridge),
    }[variant]
    settings = dict(n_steps=1, width_w=4, n_features_w=2, reg=0.01, seed=7, depth=1)
    config = config_type(weight_by_cells=False, **settings)
    configured = build_neural(config, standardise_w=standardise_w).fit(cells)
    explicit = Standardised(
        NeuralBridgeTransfer(weight_by_cells=False, variant=variant, **settings), w=standardise_w
    ).fit(cells)
    assert type(configured.model.bridge_) is model_type
    np.testing.assert_allclose(
        configured.predict(["b", "a", "b"], cells), explicit.predict(["b", "a", "b"], cells)
    )
    assert config.reg == 0.01
    if variant != "per_x":
        assert config.deviation_reg is None


@pytest.mark.parametrize("standardise_w", [False, True])
def test_typed_kernel_builder_preserves_predictions(cells, standardise_w):
    from genetic_perturbation_playground.config import KernelConfig
    from genetic_perturbation_playground.model.batch_transfer import build_kernel

    configured = build_kernel(
        KernelConfig(reg=0.01, weight_by_cells=False, lengthscale_scale=2),
        standardise_w=standardise_w,
    ).fit(cells)
    explicit = Standardised(KernelBridgeTransfer(0.01, False, 2), w=standardise_w).fit(cells)
    np.testing.assert_allclose(
        configured.predict(["b", "a"], cells), explicit.predict(["b", "a"], cells)
    )


def test_config_defaults_match_constructor_defaults():
    from genetic_perturbation_playground.config import (
        HybridConfig,
        KernelConfig,
        LinearConfig,
        PerXConfig,
        Shared2NNConfig,
    )

    for config_type, model_type in [
        (LinearConfig, LinearTransfer),
        (KernelConfig, KernelBridgeTransfer),
        (PerXConfig, NeuralPerXEmbeddingBridge),
        (Shared2NNConfig, NeuralSharedEmbeddingBridge),
        (HybridConfig, NeuralHybridEmbeddingBridge),
    ]:
        config = config_type()
        for name, parameter in inspect.signature(model_type).parameters.items():
            assert getattr(config, name) == parameter.default, (model_type, name)


def test_builder_rejects_variant_schema_mismatch():
    from genetic_perturbation_playground.config import PerXConfig
    from genetic_perturbation_playground.model.batch_transfer import build_neural

    with pytest.raises(ValueError, match="neural=hybrid"):
        build_neural(PerXConfig(variant="hybrid"))


TRANSFER_CONFIGS = [
    "train_replogle_transfer",
    "train_rxrx1_transfer",
    "tune_replogle_transfer",
    "tune_rxrx1_transfer",
    "tune_replogle_transfer_perx",
]


def compose_config(name="train_replogle_transfer", *overrides):
    from genetic_perturbation_playground.config import register_configs

    register_configs()
    register_configs()
    with initialize_config_dir(str(Path(__file__).parents[1] / "configs"), version_base=None):
        return compose(name, ["name=test", *overrides])


@pytest.mark.parametrize("name", TRANSFER_CONFIGS)
@pytest.mark.parametrize("variant", ["per_x", "shared_2nn", "hybrid"])
def test_transfer_presets_retain_model_types_and_defaults(name, variant):
    from genetic_perturbation_playground.config import (
        HybridConfig,
        KernelConfig,
        LinearConfig,
        PerXConfig,
        Shared2NNConfig,
        resolve_model_config,
    )

    config = compose_config(name, f"neural={variant}")
    neural = resolve_model_config(config, "neural")
    assert (
        type(neural)
        is {"per_x": PerXConfig, "shared_2nn": Shared2NNConfig, "hybrid": HybridConfig}[variant]
    )
    assert neural.variant == variant
    assert (neural.depth, neural.n_features_w, neural.width_w) == (
        1,
        32,
        128 if "rxrx1" in name else 64,
    )
    assert (neural.reg, neural.lr, neural.weight_decay) == (1e-6, 1e-3, 1e-2)
    assert (neural.n_steps, neural.cross_fit_folds, neural.device) == (200, 5, "cuda")
    assert resolve_model_config(config, "linear") == LinearConfig()
    assert resolve_model_config(config, "kernel") == KernelConfig()


@pytest.mark.parametrize(
    "override",
    [
        "neural=unknown",
        "neural.depth=invalid",
        "neural.depth=1.5",
        "neural.dept=1",
        "neural.linear_reg=0",
    ],
)
def test_transfer_presets_reject_invalid_model_overrides(override):
    with pytest.raises(ConfigCompositionException):
        compose_config("train_replogle_transfer", override)


@pytest.mark.parametrize("override", ["+neural.linear_reg=0", "+neural.typo=1"])
def test_resolution_rejects_undeclared_model_fields_added_with_plus(override):
    from genetic_perturbation_playground.config import resolve_model_config

    config = compose_config("train_replogle_transfer", override)
    with pytest.raises(ValueError, match="Unknown.*neural"):
        resolve_model_config(config, "neural")


def test_resolution_rejects_variant_schema_mismatch():
    from genetic_perturbation_playground.config import resolve_model_config

    config = compose_config("train_replogle_transfer", "neural.variant=hybrid")
    with pytest.raises(ValueError, match="neural=hybrid"):
        resolve_model_config(config, "neural")


def test_trial_interpolations_see_trial_values_without_changing_the_base():
    from genetic_perturbation_playground.config import resolve_model_config

    config = compose_config(
        "train_replogle_transfer",
        "neural=shared_2nn",
        "seed=19",
        "neural.deviation_reg=${neural.reg}",
    )
    before = OmegaConf.to_container(config, resolve=True)
    first = resolve_model_config(config, "neural", {"reg": 0.005, "log_key": "first"})
    second = resolve_model_config(config, "neural", {"reg": 0.02, "log_key": "second"})
    assert (first.seed, first.reg, first.deviation_reg, first.log_key) == (
        19,
        0.005,
        0.005,
        "first",
    )
    assert (second.seed, second.reg, second.deviation_reg, second.log_key) == (
        19,
        0.02,
        0.02,
        "second",
    )
    assert OmegaConf.to_container(config, resolve=True) == before


def test_root_switches_and_model_penalties_resolve_independently():
    from genetic_perturbation_playground.config import resolve_model_config

    config = compose_config(
        "train_rxrx1_transfer",
        "neural=hybrid",
        "linear.reg=0",
        "neural.linear_reg=0.002",
        "standardise_w=false",
        "weight_by_cells=false",
    )
    linear = resolve_model_config(config, "linear")
    neural = resolve_model_config(config, "neural")
    kernel = resolve_model_config(config, "kernel")
    assert linear.reg == 0
    assert neural.linear_reg == 0.002
    assert linear.standardise_w is False
    assert kernel.weight_by_cells is neural.weight_by_cells is False
    saved = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    assert saved.neural.variant == "hybrid"
    assert saved.neural.depth == 1


def test_tuning_grid_can_add_a_declared_model_field():
    config = compose_config("tune_replogle_transfer", "+neural_grid.width_w=[4]")
    assert list(config.neural_grid.width_w) == [4]


@pytest.mark.parametrize("dataset", ["replogle", "rxrx1"])
@pytest.mark.parametrize("action", ["train", "tune"])
@pytest.mark.parametrize("variant", ["per_x", "shared_2nn", "hybrid"])
def test_transfer_entry_points_run_each_neural_group(cells, monkeypatch, dataset, action, variant):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    name = f"{action}_{dataset}_transfer"
    module = importlib.import_module(name)
    config = compose_config(
        name,
        f"neural={variant}",
        "neural.device=cpu",
        "neural.n_steps=1",
        "neural.width_w=4",
        "neural.n_features_w=2",
        "neural.cross_fit_folds=0",
        "n_validation_batches=1",
    )
    if dataset == "rxrx1":
        config.wandb_mode = "disabled"
    else:
        config.top_k = 2
    if action == "train":
        fit = module.fit if dataset == "replogle" else module.fit_models
        model = fit(cells, config)["neural bridge"]
        expected = {
            "per_x": NeuralPerXEmbeddingBridge,
            "shared_2nn": NeuralSharedEmbeddingBridge,
            "hybrid": NeuralHybridEmbeddingBridge,
        }[variant]
        assert type(model.model.bridge_) is expected
        predictions = model.predict(["b", "a", "b"], cells)
        assert predictions.shape == (3, 3)
        assert np.isfinite(predictions).all()
        results = module.cross_validate(cells, config)
        scores = results[2] if dataset == "replogle" else pd.concat(results)
    else:
        config.inner_folds, config.n_configs = 2, 1
        config.kernel_grid = {"reg": [0.001], "lengthscale_scale": [1]}
        config.neural_grid = {
            "reg": [0.001],
            "weight_decay": [0.0],
            "psi_size": [2],
            "width_w": [4],
        }
        config.neural_checkpoints = [1]
        if dataset == "replogle":
            _, selected, _, _, scores = module.nested_cv(cells, config)
        else:
            [(_, selected, scores)] = list(module.nested_cv(cells, config))
        assert selected.loc[selected.bridge == "neural", "n_steps"].tolist() == [1]
    assert not scores.empty
    assert np.isfinite(scores["value"]).all()
