"""Known-relationship recall of RxRx1 perturbation maps.

Each gene has a profile, its centred mean embedding. For genes a and b, let
c(a, b) be the cosine similarity of their profiles, and let q_low and q_high
be the `tail` and 1 - `tail` quantiles of c over all distinct gene pairs.
A known pair is a hit when c(a, b) <= q_low or c(a, b) >= q_high, and

    recall = (number of hits) / (number of known pairs among the study genes).
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from genetic_perturbation_playground import tuning
from genetic_perturbation_playground.config import resolve_model_config
from genetic_perturbation_playground.data.transfer import is_control
from genetic_perturbation_playground.experiments.checkpoints import ensure_manifest
from genetic_perturbation_playground.experiments.rxrx1_sparse_recovery import (
    observed_average,
    reference_pairs,
    retained_order,
)
from genetic_perturbation_playground.model.batch_transfer import build_neural
from genetic_perturbation_playground.model.linear_transfer import (
    FullLinearTransfer,
    LinearTransfer,
)
from genetic_perturbation_playground.model.virtual_experiments import virtual_experiments

FITTED_LINEAR = {"transfer (fit b)": LinearTransfer, "full transfer (fit B)": FullLinearTransfer}


def huvec_estimates(data: ad.AnnData, models: dict[str, object]) -> dict[str, pd.DataFrame]:
    """Estimate each siRNA's mean embedding from observations and from models.

    Parameters
    ----------
    data : anndata.AnnData
        Crops of one cell type, with `perturbation`, `batch` and `well_type`
        columns in `obs` and proxy controls already selected.
    models : dict of str to object
        Transfer models fitted on `data`, by name.

    Returns
    -------
    estimates : dict of str to pandas.DataFrame
        Frames of shape (n_sirnas, n_features), indexed by sorted siRNA.
        "observed" weights each batch where the siRNA was measured equally.
        Each model's frame averages its predictions over every batch with
        equal weights, including batches where the siRNA was never measured.
    """
    labels = sorted(data.obs.perturbation[data.obs.well_type.eq("treatment")].unique())
    index = pd.Index(labels, name="perturbation")
    controls = data[is_control(data)]
    estimates = {"observed": observed_average(data, labels)}
    for name, model in models.items():
        average = virtual_experiments(model, controls, perturbations=labels)
        estimates[name] = pd.DataFrame(average.X, index=index, columns=data.var_names)
    return estimates


def virtual_folds(
    data: ad.AnnData, n_folds: int, seed: int
) -> list[tuple[ad.AnnData, Callable[[object], dict]]]:
    """Split each siRNA's plates into folds for virtual-experiment tuning.

    Each siRNA's measured plates are shuffled and dealt into `n_folds` folds.
    Round j hides every siRNA's fold-j treatment wells and keeps all controls,
    so every plate stays in training. A model is scored by how well its
    average prediction over each siRNA's hidden plates matches the observed
    average on the same plates.

    Parameters
    ----------
    data : anndata.AnnData
        Crops of one cell type, with `perturbation`, `batch` and `well_type`
        columns in `obs` and proxy controls already selected.
    n_folds : int
        Number of folds, at most the fewest plates of any siRNA.
    seed : int
        Seed for the plate shuffle of each siRNA.

    Returns
    -------
    folds : list of tuple of (anndata.AnnData, callable)
        One `(train, evaluate)` pair per round, as `tuning.tune` expects.
        `evaluate(model)` returns `{"mse": value}`, the squared error
        averaged over features and siRNAs.

    Raises
    ------
    ValueError
        If an siRNA has fewer plates than `n_folds`; the message names it.
    """
    order = retained_order(reference_pairs(data), seed)
    n_plates = order.groupby("perturbation").batch.size()
    if (n_plates < n_folds).any():
        names = n_plates.index[n_plates < n_folds].tolist()
        raise ValueError(f"siRNAs with fewer than {n_folds} plates: {names}")
    keys = pd.MultiIndex.from_arrays([data.obs.perturbation, data.obs.batch])
    controls = data[is_control(data)]
    folds = []
    for fold in range(n_folds):
        hidden = order[order["rank"] % n_folds == fold]
        in_hidden = keys.isin(pd.MultiIndex.from_frame(hidden[["perturbation", "batch"]]))
        observed = observed_average(data[in_hidden], sorted(hidden.perturbation.unique()))

        def evaluate(model, hidden=hidden, observed=observed):
            # one prediction per plate, then equal plate weights per siRNA
            tables = []
            for batch, group in hidden.groupby("batch"):
                labels = group.perturbation.tolist()
                plate = controls[controls.obs.batch.eq(batch).to_numpy()]
                predicted = model.predict_environment_average(labels, plate)
                tables.append(pd.DataFrame(predicted, index=labels))
            predicted = pd.concat(tables).groupby(level=0).mean().loc[observed.index]
            return {"mse": float(((predicted.to_numpy() - observed.to_numpy()) ** 2).mean())}

        folds.append((data[~in_hidden], evaluate))
    return folds


def tune_linear(
    folds: list[tuple[ad.AnnData, Callable[[object], dict]]],
    regs: list[float],
    standardise_w: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Choose the slope penalty of each fitted linear model on validation folds.

    Pooled (b=0) and shift (b=1) fix their slope, so they are scored once per
    fold as references, and only transfer (fit b) and full transfer (fit B)
    are tuned. A penalty at either end of `regs` prints a warning, since a
    wider grid might do better.

    Parameters
    ----------
    folds : list of tuple of (anndata.AnnData, callable)
        `(train, evaluate)` pairs, as returned by `virtual_folds`.
    regs : list of float
        Candidate penalties, in increasing order.
    standardise_w : bool
        Whether the models scale controls with training-control statistics.

    Returns
    -------
    result : tuple of (dict of str to float, pandas.DataFrame)
        `(selected, scores)`, where `selected` maps each tuned model to the
        penalty with the lowest mean validation mse, and `scores` has one
        row per model, penalty and fold, with columns `model`, `reg`, `fold`
        and `mse`; `reg` is NaN for the two reference models.
    """
    rows = []
    for fold, (train, evaluate) in enumerate(folds):
        for name, slope in [("pooled (b=0)", 0.0), ("shift (b=1)", 1.0)]:
            model = LinearTransfer(slope=slope, standardise_w=standardise_w).fit(train)
            rows.append({"model": name, "reg": np.nan, "fold": fold, **evaluate(model)})
        for name, model_class in FITTED_LINEAR.items():
            for reg in regs:
                model = model_class(standardise_w=standardise_w, reg=reg).fit(train)
                rows.append({"model": name, "reg": reg, "fold": fold, **evaluate(model)})
    scores = pd.DataFrame(rows)
    means = scores.groupby(["model", "reg"]).mse.mean()
    selected = {}
    for name in FITTED_LINEAR:
        best = means[name].idxmin()
        selected[name] = float(best)
        edge = " (edge of the grid)" if best in (min(regs), max(regs)) else ""
        print(f"{name}: reg {best:g}, mse {means[name][best]:.4g}{edge}", flush=True)
    return selected, scores


def fit_linear(data: ad.AnnData, regs: dict[str, float], standardise_w: bool) -> dict[str, object]:
    """Fit the four linear models, each fitted model with its own penalty.

    Parameters
    ----------
    data : anndata.AnnData
        Training crops with proxy controls.
    regs : dict of str to float
        Slope penalty of "transfer (fit b)" and "full transfer (fit B)".
    standardise_w : bool
        Whether the models scale controls with training-control statistics.

    Returns
    -------
    models : dict of str to object
        Fitted pooled (b=0), shift (b=1), transfer (fit b) and full
        transfer (fit B), in that order.
    """
    models = {
        "pooled (b=0)": LinearTransfer(slope=0.0, standardise_w=standardise_w),
        "shift (b=1)": LinearTransfer(slope=1.0, standardise_w=standardise_w),
    }
    for name, model_class in FITTED_LINEAR.items():
        models[name] = model_class(standardise_w=standardise_w, reg=regs[name])
    return {name: model.fit(data) for name, model in models.items()}


def claim_output(data: ad.AnnData, cfg: DictConfig, directory: Path) -> None:
    """Claim `directory` for this run, or confirm it belongs to the same run.

    Call before writing anything, so a refused rerun leaves the old results
    untouched. The run is identified by the whole config and a hash of the
    embeddings, plates and treatments, so changed embeddings on the same
    plates are refused rather than resumed.

    Parameters
    ----------
    data : anndata.AnnData
        The selected crops the run fits on.
    cfg : DictConfig
        Composed recall config.
    directory : pathlib.Path
        Output directory of the run.

    Raises
    ------
    ValueError
        If `directory` holds results from a different config or data.
    """
    digest = hashlib.sha256(np.ascontiguousarray(data.X).tobytes())
    digest.update(data.obs[["perturbation", "batch"]].to_csv().encode())
    digest.update("\n".join(data.var_names).encode())
    identity = {"config": OmegaConf.to_container(cfg, resolve=True), "data": digest.hexdigest()}
    ensure_manifest(directory, {"schema_version": 1, "identity": identity}, resume=True)


def tune_bridge(
    folds: list[tuple[ad.AnnData, Callable[[object], dict]]],
    cfg: DictConfig,
    directory: Path,
) -> tuple[dict, pd.DataFrame]:
    """Choose the neural bridge's settings on validation folds, resumably.

    Each finished (setting, fold) trial is saved in `directory`, and a rerun
    skips it, so tuning restarts at the first unfinished trial; a trial cut
    off midway reruns. Use `claim_output` first so the trials belong to this
    config and data.

    Parameters
    ----------
    folds : list of tuple of (anndata.AnnData, callable)
        `(train, evaluate)` pairs, as returned by `virtual_folds`.
    cfg : DictConfig
        Composed recall config, with the search settings `tuning.tune` reads.
    directory : pathlib.Path
        Trial storage.

    Returns
    -------
    result : tuple of (dict, pandas.DataFrame)
        `(settings, scores)`, where `settings` holds the winning neural
        settings and `scores` has one row per setting, checkpoint and fold.
    """
    candidates = tuning.candidate_configs(cfg)
    scores, selected, _ = tuning.tune_resumable(
        folds, cfg, directory=directory, candidates=candidates
    )
    return selected["neural"], scores


def fit_bridge(data: ad.AnnData, cfg: DictConfig, settings: dict) -> object:
    """Fit the neural bridge on all of `data`.

    Parameters
    ----------
    data : anndata.AnnData
        Training crops with proxy controls.
    cfg : DictConfig
        Composed recall config; `neural` and `standardise_w` give the
        settings that `settings` does not.
    settings : dict
        Tuned settings, as returned by `tune_bridge`. Empty uses the fixed
        settings in `cfg.neural`.

    Returns
    -------
    model : object
        The fitted bridge.
    """
    if settings:
        return tuning.make_neural(settings, cfg).fit(data)
    config = resolve_model_config(cfg, "neural")
    return build_neural(config, standardise_w=cfg.standardise_w).fit(data)


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
