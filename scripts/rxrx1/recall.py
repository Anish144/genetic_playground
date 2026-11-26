"""Known-relationship recall on RxRx1 HUVEC: full observed means vs bridge averages.

Two estimates per siRNA, mapped to genes and centred before cosine similarity:
  observed  mean over the plates where the siRNA was measured
  bridge    neural bridge fit on every plate, averaged over every plate
Recall is the fraction of each database's gene pairs whose cosine lies in the top or
bottom `tail` of all gene pairs (configs/recall/rxrx1.yaml).

    uv run python scripts/rxrx1/recall.py name=huvec_recall
    sbatch -J huvec_recall hpc/run.sbatch scripts/rxrx1/recall.py name=huvec_recall

Writes recall.csv, excluded_sirnas.csv, estimates.npz and config.yaml to outputs/rxrx1_recall/<name>/.
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# ponytail: reach the select_* helpers in scripts/; drop once they move into experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_rxrx1_transfer import (  # noqa: E402
    select_batch_unit,
    select_cell_type,
    select_proxy_controls,
)

from genetic_perturbation_playground.config import register_configs  # noqa: E402
from genetic_perturbation_playground.data.rxrx1 import load_rxrx1  # noqa: E402
from genetic_perturbation_playground.experiments.rxrx1_recall import (  # noqa: E402
    gene_mapping,
    huvec_estimates,
    recall_table,
)

register_configs()


@hydra.main(config_path="../../configs", config_name="recall/rxrx1", version_base=None)
def main(cfg: DictConfig):
    data = select_cell_type(load_rxrx1(**cfg.embeddings), cfg.cell_type)
    data = select_proxy_controls(select_batch_unit(data, cfg.batch_unit), cfg)
    estimates = huvec_estimates(data, cfg)
    out_dir = Path(to_absolute_path(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    # saved before recall, so a database error keeps the bridge fit
    OmegaConf.save(cfg, out_dir / "config.yaml")
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
