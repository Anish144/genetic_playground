"""Known-relationship recall of RxRx1 perturbation maps.

Each gene has a profile, its centred mean embedding. For genes a and b, let
c(a, b) be the cosine similarity of their profiles, and let q_low and q_high
be the `tail` and 1 - `tail` quantiles of c over all distinct gene pairs.
A known pair is a hit when c(a, b) <= q_low or c(a, b) >= q_high, and

    recall = (number of hits) / (number of known pairs among the study genes).
"""

import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from genetic_perturbation_playground.config import resolve_model_config
from genetic_perturbation_playground.data.transfer import is_control
from genetic_perturbation_playground.experiments.rxrx1_sparse_recovery import observed_average
from genetic_perturbation_playground.model.batch_transfer import build_neural
from genetic_perturbation_playground.model.virtual_experiments import virtual_experiments


def huvec_estimates(data: ad.AnnData, cfg: DictConfig) -> dict[str, pd.DataFrame]:
    """Estimate each siRNA's mean embedding from observations and the bridge.

    Parameters
    ----------
    data : anndata.AnnData
        Crops of one cell type, with `perturbation`, `batch` and `well_type`
        columns in `obs` and proxy controls already selected.
    cfg : DictConfig
        Composed recall config. The bridge uses `neural` and `standardise_w`.

    Returns
    -------
    estimates : dict of str to pandas.DataFrame
        Two frames of shape (n_sirnas, n_features), indexed by sorted siRNA:
        "observed" weights each batch where the siRNA was measured equally;
        "bridge" averages the fitted bridge's predictions over every batch,
        including batches where the siRNA was never measured.
    """
    labels = sorted(data.obs.perturbation[data.obs.well_type.eq("treatment")].unique())
    model = build_neural(resolve_model_config(cfg, "neural"), standardise_w=cfg.standardise_w)
    model.fit(data)
    bridge = virtual_experiments(model, data[is_control(data)], perturbations=labels)
    return {
        "observed": observed_average(data, labels),
        "bridge": pd.DataFrame(
            bridge.X, index=pd.Index(labels, name="perturbation"), columns=data.var_names
        ),
    }


def gene_mapping(symbols: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Validate siRNA gene symbols and set aside ambiguous ones.

    Symbols a spreadsheet turned into numbers, such as "LSM 7.00" for LSM7,
    are repaired. siRNAs with no symbol or several ";"-separated targets are
    excluded, because they cannot be matched to annotated gene pairs.

    Parameters
    ----------
    symbols : pandas.Series
        Raw gene symbol, possibly missing, indexed by siRNA.

    Returns
    -------
    result : tuple of (pandas.Series, pandas.Series)
        `(mapping, excluded)`, where `mapping` holds the valid symbol of each
        kept siRNA and `excluded` holds the reason each other siRNA was
        dropped: "no gene symbol" or "multiple genes".

    Raises
    ------
    ValueError
        If a kept symbol is not letters, digits and hyphens after repair.
    """
    repaired = symbols.str.replace(r"^([A-Z]+)\s+(\d+)(?:\.0+)?$", r"\1\2", regex=True)
    reasons = pd.Series(pd.NA, index=symbols.index, dtype=object)
    reasons[symbols.isna()] = "no gene symbol"
    reasons[repaired.str.contains(";", na=False)] = "multiple genes"
    kept = repaired[reasons.isna()]
    invalid = kept[~kept.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*")]
    if len(invalid):
        raise ValueError(f"Unrecognised gene symbols: {invalid.to_dict()}")
    return kept, reasons.dropna()


def gene_profiles(means: pd.DataFrame, sirna_to_gene: pd.Series) -> tuple[pd.DataFrame, int]:
    """Average siRNA means per gene and centre each feature across genes.

    Parameters
    ----------
    means : pandas.DataFrame of shape (n_sirnas, n_features)
        Mean embedding per siRNA, indexed by siRNA.
    sirna_to_gene : pandas.Series
        Gene symbol indexed by siRNA. siRNAs absent from it are dropped.

    Returns
    -------
    result : tuple of (pandas.DataFrame, int)
        `(profiles, n_unmapped)`, where `profiles` has shape
        (n_genes, n_features), is sorted by gene and has zero column means,
        and `n_unmapped` counts the dropped siRNAs.
    """
    genes = means.index.map(sirna_to_gene)
    mapped = genes.notna()
    profiles = means[mapped].groupby(genes[mapped]).mean()
    return profiles - profiles.mean(), int((~mapped).sum())


def known_pairs(table: pd.DataFrame, genes: pd.Index) -> set[tuple[str, str]]:
    """Return the unordered pairs of distinct study genes in a pair table.

    Parameters
    ----------
    table : pandas.DataFrame
        Database pairs, one gene in each of the first two columns.
    genes : pandas.Index
        Study genes. Pairs with a gene outside it are dropped.

    Returns
    -------
    pairs : set of tuple of (str, str)
        Each pair once, with its genes in sorted order.
    """
    return {
        tuple(sorted((a, b)))
        for a, b in table.iloc[:, :2].itertuples(index=False)
        if a != b and a in genes and b in genes
    }


def tail_recall(
    profiles: pd.DataFrame, pairs: set[tuple[str, str]], tail: float = 0.05
) -> tuple[int, float]:
    """Count known pairs whose cosine similarity lies in either tail.

    Parameters
    ----------
    profiles : pandas.DataFrame of shape (n_genes, n_features)
        Gene profiles indexed by gene.
    pairs : set of tuple of (str, str)
        Nonempty set of known pairs of genes in `profiles.index`.
    tail : float, default=0.05
        Fraction of all gene pairs in each tail of the cosine distribution.

    Returns
    -------
    result : tuple of (int, float)
        `(n_hits, recall)`, where `n_hits` counts pairs with cosine at or
        beyond either tail quantile and `recall` is `n_hits / len(pairs)`.

    Raises
    ------
    ValueError
        If a cosine is not finite, as for a NaN or all-zero profile.
    """
    values = profiles.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = values / np.linalg.norm(values, axis=1, keepdims=True)
    cosine = unit @ unit.T
    if not np.isfinite(cosine).all():
        raise ValueError("Cosine similarities must be finite; check for NaN or all-zero profiles.")
    low, high = np.quantile(cosine[np.triu_indices(len(unit), k=1)], [tail, 1 - tail])
    position = {gene: i for i, gene in enumerate(profiles.index)}
    hits = sum(not low < cosine[position[a], position[b]] < high for a, b in pairs)
    return hits, hits / len(pairs)


def recall_table(
    estimates: dict[str, pd.DataFrame],
    sirna_to_gene: pd.Series,
    databases: dict[str, pd.DataFrame],
    tail: float = 0.05,
) -> pd.DataFrame:
    """Compute the recall of every estimate against every database.

    Parameters
    ----------
    estimates : dict of str to pandas.DataFrame
        Mean embedding per siRNA, of shape (n_sirnas, n_features), by
        estimator name.
    sirna_to_gene : pandas.Series
        Gene symbol indexed by siRNA, as returned by `gene_mapping`.
    databases : dict of str to pandas.DataFrame
        Pair table by database name, one gene in each of the first two columns.
    tail : float, default=0.05
        Fraction of all gene pairs in each tail of the cosine distribution.

    Returns
    -------
    table : pandas.DataFrame
        One row per (estimator, database), with columns `estimator`,
        `database`, `n_genes`, `n_unmapped`, `n_pairs`, `n_hits` and `recall`.

    Raises
    ------
    ValueError
        If a database has no pairs among the study genes; the message names it.
    """
    rows = []
    for estimator, means in estimates.items():
        profiles, n_unmapped = gene_profiles(means, sirna_to_gene)
        for database, table in databases.items():
            pairs = known_pairs(table, profiles.index)
            if not pairs:
                raise ValueError(f"{database}: no known pairs among the study genes.")
            n_hits, recall = tail_recall(profiles, pairs, tail)
            rows.append(
                {
                    "estimator": estimator,
                    "database": database,
                    "n_genes": len(profiles),
                    "n_unmapped": n_unmapped,
                    "n_pairs": len(pairs),
                    "n_hits": n_hits,
                    "recall": recall,
                }
            )
    return pd.DataFrame(rows)
