"""Known-relationship recall on RxRx1 HUVEC: observed means vs model averages.

Estimates per siRNA, mapped to genes and centred before cosine similarity:
  observed       mean over the plates where the siRNA was measured
  linear models  pooled (b=0), shift (b=1), transfer (fit b), full transfer
                 (fit B): fit on every plate, averaged over every plate
  neural bridge  hybrid, fit on every plate, averaged over every plate
With tune=true (the default), the penalties of transfer (fit b) and full transfer
(fit B) and the bridge settings are first chosen on per-siRNA plate folds: each
round hides some of every siRNA's plates and scores its plate-averaged prediction
against its observed average there.
Recall is the fraction of each database's gene pairs whose cosine lies in the top or
bottom `tail` of all gene pairs (configs/recall/rxrx1.yaml).

    uv run python scripts/rxrx1/recall.py name=huvec_recall
    sbatch -J huvec_recall hpc/run.sbatch scripts/rxrx1/recall.py name=huvec_recall

Rerunning with the same name resumes tuning: finished (setting, fold) trials in
<output_dir>/trials/ are skipped, so tuning restarts at the first unfinished one.
A rerun with a different config or different embeddings is refused before it
writes anything; use a new name.

Writes recall.csv, excluded_sirnas.csv, estimates.npz, selected.json, config.yaml and,
when tuning, tuning_scores.csv and linear_tuning_scores.csv to outputs/rxrx1_recall/<name>/.
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# ponytail: reach helpers in scripts/; drop once they move into experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_rxrx1_transfer import (  # noqa: E402
    select_batch_unit,
    select_cell_type,
    select_proxy_controls,
)

from genetic_perturbation_playground.config import register_configs  # noqa: E402
from genetic_perturbation_playground.data.rxrx1 import load_rxrx1  # noqa: E402
from genetic_perturbation_playground.experiments.checkpoints import run_lock  # noqa: E402
from genetic_perturbation_playground.experiments.rxrx1_recall import (  # noqa: E402
    claim_output,
    fit_bridge,
    fit_linear,
    gene_mapping,
    huvec_estimates,
    recall_table,
    tune_bridge,
    tune_linear,
    virtual_folds,
)

register_configs()


@hydra.main(config_path="../../configs", config_name="recall/rxrx1", version_base=None)
def main(cfg: DictConfig):
    out_dir = Path(to_absolute_path(cfg.output_dir))
    with run_lock(out_dir):
        run(cfg, out_dir)


def run(cfg: DictConfig, out_dir: Path):
    data = select_cell_type(load_rxrx1(**cfg.embeddings), cfg.cell_type)
    data = select_proxy_controls(select_batch_unit(data, cfg.batch_unit), cfg)
    claim_output(data, cfg, out_dir)  # before any write, so a refused rerun changes nothing
    OmegaConf.save(cfg, out_dir / "config.yaml")

    if cfg.tune:
        folds = virtual_folds(data, cfg.inner_folds, cfg.seed)
        regs, linear_scores = tune_linear(folds, list(cfg.linear_reg_grid), cfg.standardise_w)
        linear_scores.to_csv(out_dir / "linear_tuning_scores.csv", index=False)
        bridge, bridge_scores = tune_bridge(folds, cfg, out_dir / "trials")
        bridge_scores.to_csv(out_dir / "tuning_scores.csv", index=False)
    else:
        regs = {"transfer (fit b)": cfg.linear.reg, "full transfer (fit B)": cfg.linear.reg}
        bridge = {}
    # saved before the final fits, so a failed refit keeps hours of tuning
    selected = {"linear_reg": regs, "neural": bridge or "fixed settings in config.yaml"}
    (out_dir / "selected.json").write_text(json.dumps(selected, indent=2))
    models = fit_linear(data, regs, cfg.standardise_w)
    models["neural bridge"] = fit_bridge(data, cfg, bridge)

    # saved before recall, so a database error keeps the fits
    estimates = huvec_estimates(data, models)
    np.savez_compressed(
        out_dir / "estimates.npz",
        perturbations=estimates["observed"].index.to_numpy(str),
        features=data.var_names.to_numpy(str),
        **{name: frame.to_numpy() for name, frame in estimates.items()},
    )

    symbols = pd.read_csv(to_absolute_path(cfg.sirna_to_gene)).set_index("siRNA ID")
    sirna_to_gene, excluded = gene_mapping(symbols["Gene Symbol"])
    excluded = excluded[excluded.index.isin(estimates["observed"].index)]
    excluded.rename("reason").to_csv(out_dir / "excluded_sirnas.csv")
    print(f"excluded siRNAs: {excluded.value_counts().to_dict()}")
    annotations = Path(to_absolute_path(cfg.annotations_dir))
    databases = {name: pd.read_csv(annotations / f"{name}.txt") for name in cfg.databases}
    table = recall_table(estimates, sirna_to_gene, databases, cfg.tail)
    table.to_csv(out_dir / "recall.csv", index=False)
    print(table.to_string(index=False, float_format="%.3f"))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
