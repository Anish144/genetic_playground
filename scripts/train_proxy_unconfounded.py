"""Train the marginal-embedding bridge on the unconfounded proxy setting and report target MSE.

For each proxy function g, number of source environments D, target theta* and seed, three predictors of
E[Y | x, e*] are compared with the exact value on a grid of x:
  bridge  MarginalEmbeddingBridge on the sources, given only the target's proxies W
  pooled  kernel ridge regression of y on x over all sources, ignoring the environment
  oracle  kernel ridge regression of y on x over labelled target data

Config: configs/train_proxy_unconfounded.yaml. Usage:

    uv run python scripts/train_proxy_unconfounded.py n_seeds=3 'n_envs=[20]'

Per-run scores are written to the Hydra run dir as scores.csv; a summary is printed.
"""

import itertools
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from scipy.linalg import cho_factor, cho_solve

from genetic_perturbation_playground.data.proxy_unconfounded import (
    conditional_mean,
    sample_environment,
    sample_environments,
)
from genetic_perturbation_playground.model.proxy_unconfounded import MarginalEmbeddingBridge
from genetic_perturbation_playground.utils.kernels import RBF, Kernel


def kernel_ridge(x, y, x_new, kernel: Kernel, reg: float) -> np.ndarray:
    n = len(x)
    return kernel(x, x_new).T @ cho_solve(cho_factor(kernel(x) + reg * n * np.eye(n)), y)


def run(cfg: DictConfig, proxy: str, d: int, theta_star: float, seed: int) -> dict[str, float]:
    """MSE against the true E[Y | x, e*] for each predictor."""
    rng = np.random.default_rng(seed)
    sources = sample_environments(d, cfg.n, cfg.m, rng, proxy=proxy)
    target = sample_environment(theta_star, cfg.n_target_labels, cfg.m, rng, proxy=proxy)
    x_grid = np.linspace(-3, 3, cfg.n_grid)
    truth = conditional_mean(x_grid, theta_star)

    kx, kw = RBF(lengthscales=cfg.lengthscale_x), RBF(lengthscales=cfg.lengthscale_w)
    xs, ys = [e.x for e in sources], [e.y for e in sources]
    bridge = MarginalEmbeddingBridge(kx, kw, cfg.reg).fit(xs, ys, [e.w for e in sources])
    preds = {
        "bridge": bridge.predict(x_grid, target.w),
        "pooled": kernel_ridge(np.concatenate(xs), np.concatenate(ys), x_grid, kx, cfg.reg),
        "oracle": kernel_ridge(target.x, target.y, x_grid, kx, cfg.reg),
    }
    return {model: float(np.mean((p - truth) ** 2)) for model, p in preds.items()}


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
    scores.to_csv(Path(HydraConfig.get().runtime.output_dir) / "scores.csv", index=False)
    summary = scores.groupby(["proxy", "n_envs", "theta_star", "model"])["mse"].agg(["mean", "std"])
    print(summary.unstack("model").round(4).to_string())


if __name__ == "__main__":
    main()
