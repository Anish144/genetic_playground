"""Train the multi-domain kernel bridge on the D.3 synthetic task and report target metrics.

For each task and seed, three predictors are scored on held-out target samples:
  proposed   bridge fit on the source domains, adapted with the target W | X embedding
  no_adapt   the same bridge with a W | X embedding fit on the pooled sources instead
  oracle     bridge fit on labelled target data (Tsai et al., 2024, Appendix D.2)

Config: configs/train_proxy_confounded.yaml. Usage:

    uv run python scripts/train_proxy_confounded.py n_seeds=3 tasks=[1]

Per-run scores are written to the Hydra run dir as scores.csv; a summary is printed.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from genetic_perturbation_playground.data.proxy_confounded import ConfoundedSample, sample_task
from genetic_perturbation_playground.model.proxy_confounded import (
    ConditionalEmbeddingBridge,
    ConditionalMeanEmbedding,
)
from genetic_perturbation_playground.utils import metrics
from genetic_perturbation_playground.utils.kernels import RBF, median_heuristic

METRICS = {"auroc": metrics.auroc, "accuracy": metrics.accuracy, "brier": metrics.brier}


def run(cfg: DictConfig, task: int, seed: int) -> dict[str, np.ndarray]:
    """Target-test scores of each predictor, plus the test labels under "labels"."""
    rng = np.random.default_rng(seed)
    sources, target = sample_task(task, cfg.n_source, cfg.n_target, rng, cfg.a_w)
    _, test = sample_task(task, 0, cfg.n_test, rng, cfg.a_w)
    pooled = ConfoundedSample.concat(sources)

    kx = RBF(lengthscales=median_heuristic(pooled.x))
    kw = RBF(lengthscales=median_heuristic(pooled.w))

    def bridge(d: ConfoundedSample) -> ConditionalEmbeddingBridge:
        return ConditionalEmbeddingBridge(kx, kw, cfg.reg_cme, cfg.reg_bridge).fit(
            d.x, d.w, d.y, d.z, rng
        )

    def cme(d: ConfoundedSample) -> ConditionalMeanEmbedding:
        return ConditionalMeanEmbedding(kx, cfg.reg_cme).fit(d.x, d.w)

    source_bridge, target_cme = bridge(pooled), cme(target)
    return {
        "labels": test.y,
        "proposed": source_bridge.predict(test.x, target_cme),
        "no_adapt": source_bridge.predict(test.x, cme(pooled)),
        "oracle": bridge(target).predict(test.x, target_cme),
    }


def evaluate(cfg: DictConfig) -> pd.DataFrame:
    """Long table (task, seed, model, metric, value)."""
    rows = []
    for task in cfg.tasks:
        for seed in range(cfg.n_seeds):
            preds = run(cfg, task, seed)
            labels = preds.pop("labels")
            rows += [
                {"task": task, "seed": seed, "model": model, "metric": name, "value": f(p, labels)}
                for model, p in preds.items()
                for name, f in METRICS.items()
            ]
            print(f"task {task} seed {seed}: AUROC {metrics.auroc(preds['proposed'], labels):.4f}")
    return pd.DataFrame(rows)


@hydra.main(config_path="../configs", config_name="train_proxy_confounded", version_base=None)
def main(cfg: DictConfig):
    scores = evaluate(cfg)
    scores.to_csv(Path(HydraConfig.get().runtime.output_dir) / "scores.csv", index=False)
    summary = scores.groupby(["metric", "task", "model"])["value"].agg(["mean", "std"])
    print(summary.unstack("model").round(4).to_string())


if __name__ == "__main__":
    main()
