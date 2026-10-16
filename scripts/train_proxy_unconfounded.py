"""Train a marginal-embedding bridge on the unconfounded proxy setting and report target MSE.

For each proxy function g, number of source environments D, target theta* and seed, predictors
are scored by MSE on held-out (x, y) pairs from the target environment:
  bridge  a bridge fit on the sources, given only the target's proxies W: the kernel
          MarginalEmbeddingBridge (bridge=kernel) or NeuralMarginalEmbeddingBridge (bridge=neural)
  pooled  kernel ridge regression of y on x over all sources, ignoring the environment
  oracle  kernel ridge regression of y on x over labelled target data
  bayes   the true E[Y | x, e*]; its MSE is the irreducible error Var(Y | x, e*)

Config: configs/train_proxy_unconfounded.yaml. Usage:

    uv run python scripts/train_proxy_unconfounded.py n_seeds=3 'n_envs=[20]'
    uv run python scripts/train_proxy_unconfounded.py bridge=neural neural.n_steps=200

Per-run scores are written to the Hydra run dir as scores.csv, with one plot per proxy of
excess MSE over the oracle against the number of source environments in plots/; a summary is printed.
"""

import itertools
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from genetic_perturbation_playground.data.proxy_unconfounded import (
    conditional_mean,
    sample_environment,
    sample_environments,
)
from genetic_perturbation_playground.model.proxy_confounded import ConditionalMeanEmbedding
from genetic_perturbation_playground.model.proxy_unconfounded import MarginalEmbeddingBridge
from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralMarginalEmbeddingBridge,
)
from genetic_perturbation_playground.utils.kernels import RBF, median_heuristic
from genetic_perturbation_playground.utils.plotting import plot_excess_mse_vs_n_envs


def run(cfg: DictConfig, proxy: str, d: int, theta_star: float, seed: int) -> dict[str, float]:
    """Held-out target MSE of each predictor."""
    rng = np.random.default_rng(seed)
    sources = sample_environments(d, cfg.n, cfg.m, rng, dim_u=cfg.dim_u, proxy=proxy)
    theta = np.full(cfg.dim_u, theta_star)  # same mean in every latent coordinate
    target = sample_environment(theta, cfg.n_target_labels, cfg.m, rng, proxy=proxy)
    test = sample_environment(theta, cfg.n_test, 0, rng, proxy=proxy)

    xs, ys, ws = [e.x for e in sources], [e.y for e in sources], [e.w for e in sources]
    kx = RBF(lengthscales=median_heuristic(np.concatenate(xs)))
    kw = RBF(lengthscales=median_heuristic(np.concatenate(ws)))
    bridges = {
        "kernel": lambda: MarginalEmbeddingBridge(kx, kw, cfg.reg),
        "neural": lambda: NeuralMarginalEmbeddingBridge(reg=cfg.reg, seed=seed, **cfg.neural),
    }
    bridge = bridges[cfg.bridge]().fit(xs, ys, ws)

    def kernel_ridge(x, y):  # regression of y on x = ConditionalMeanEmbedding read out at y
        return ConditionalMeanEmbedding(kx, cfg.reg).fit(x, y).weights(test.x).T @ y

    preds = {
        "bridge": bridge.predict(test.x, target.w),
        "pooled": kernel_ridge(np.concatenate(xs), np.concatenate(ys)),
        "oracle": kernel_ridge(target.x, target.y),
        "bayes": conditional_mean(test.x, theta),
    }
    return {model: float(np.mean((p - test.y) ** 2)) for model, p in preds.items()}


def evaluate(cfg: DictConfig) -> pd.DataFrame:
    """Long table (proxy, n_envs, theta_star, seed, model, mse)."""
    rows = []
    for proxy, d in itertools.product(cfg.proxies, cfg.n_envs):
        for theta_star in cfg.theta_targets:
            for seed in range(cfg.n_seeds):
                mse = run(cfg, proxy, d, theta_star, seed)
                rows += [
                    {
                        "proxy": proxy,
                        "n_envs": d,
                        "theta_star": theta_star,
                        "seed": seed,
                        "model": k,
                        "mse": v,
                    }
                    for k, v in mse.items()
                ]
            print(
                f"{proxy} D={d} theta*={theta_star:+.1f}: bridge MSE {mse['bridge']:.4f} (last seed)"
            )
    return pd.DataFrame(rows)


@hydra.main(config_path="../configs", config_name="train_proxy_unconfounded", version_base=None)
def main(cfg: DictConfig):
    scores = evaluate(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    scores.to_csv(out_dir / "scores.csv", index=False)
    models = {
        "bridge": "domain transfer (target W only)",
        "pooled": "pooled sources (ignores environment)",
    }
    for proxy, s in scores.groupby("proxy"):
        path = out_dir / "plots" / f"excess_mse_vs_n_envs_{proxy}_{cfg.bridge}.png"
        plot_excess_mse_vs_n_envs(s, models, path)
    summary = scores.groupby(["proxy", "n_envs", "theta_star", "model"])["mse"].agg(["mean", "std"])
    print(summary.unstack("model").round(4).to_string())


if __name__ == "__main__":
    main()
