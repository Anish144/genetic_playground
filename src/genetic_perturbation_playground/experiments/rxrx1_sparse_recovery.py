"""Retain sparse responses and evaluate equal-batch RxRx1 averages.

Only treatment rows enter response means. Controls remain available in all
splits; each perturbation defines its own original reference population.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterator
from functools import partial

import anndata as ad
import numpy as np
import pandas as pd

from genetic_perturbation_playground.data.transfer import (
    as_dense,
    batch_labels,
    is_control,
    perturbation_labels,
)
from genetic_perturbation_playground.model.virtual_experiments import virtual_experiments
from genetic_perturbation_playground.utils.metrics import mse


def reference_pairs(data: ad.AnnData) -> pd.DataFrame:
    """Return original treatment/batch memberships and crop counts.

    Parameters
    ----------
    data : anndata.AnnData
        Selected cell type and batch labels, before response removal.

    Returns
    -------
    pandas.DataFrame
        Sorted perturbation, batch and n_crops columns; controls are excluded.
    """
    responses = data.obs.loc[data.obs.well_type.eq("treatment"), ["perturbation", "batch"]]
    if responses.isna().any().any():
        raise ValueError("Treatment perturbation and batch labels must be nonmissing.")
    return (
        responses.astype(str)
        .groupby(["perturbation", "batch"], sort=True)
        .size()
        .rename("n_crops")
        .reset_index()
    )


def eligibility_table(pairs: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Record coverage and exclusions for every perturbation and requested k.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Original reference_pairs table.
    k_values : list of int
        Numbers of response batches to retain.

    Returns
    -------
    pandas.DataFrame
        Coverage counts, eligibility and an explicit exclusion reason.
    """
    counts = pairs.groupby("perturbation").batch.nunique().rename("n_reference_batches")
    tables = []
    for k in k_values:
        table = counts.reset_index().assign(k=k)
        table["eligible"] = table.n_reference_batches >= k
        table["n_retained_batches"] = np.where(table.eligible, k, 0)
        table["n_removed_batches"] = table.n_reference_batches - table.n_retained_batches
        table["reason"] = np.where(table.eligible, "", "fewer_than_k_batches")
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def retained_order(pairs: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign stable random batch ranks independently per perturbation.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Original reference pairs, in any row order.
    seed : int
        Nonnegative seed shared across k values.

    Returns
    -------
    pandas.DataFrame
        Sorted original pairs with rank; the first k ranks define retention.
    """
    tables = []
    for label, group in pairs.groupby("perturbation", sort=True):
        group = group.sort_values("batch").copy()
        digest = int.from_bytes(hashlib.sha256(str(label).encode()).digest(), "little")
        rng = np.random.default_rng(np.random.SeedSequence([seed, digest]))
        group["rank"] = rng.permutation(len(group))
        tables.append(group)
    return pd.concat(tables, ignore_index=True)


def _retained_pairs(order, k):
    eligible = order.groupby("perturbation").batch.transform("size") >= k
    return order[eligible & order["rank"].lt(k)]


def _pair_mask(data, pairs):
    keys = pd.MultiIndex.from_arrays([perturbation_labels(data), batch_labels(data)])
    return keys.isin(pd.MultiIndex.from_frame(pairs[["perturbation", "batch"]]))


def retained_data(
    data: ad.AnnData, order: pd.DataFrame, k: int, *, held_out_rank: int | None = None
) -> ad.AnnData:
    """Select retained response pairs and all controls without mutating input.

    Parameters
    ----------
    data : anndata.AnnData
        Original responses and selected proxy controls.
    order : pandas.DataFrame
        Original pairs with seeded ranks.
    k : int
        Retained batches per eligible perturbation.
    held_out_rank : int or None, default=None
        Exclude this retained rank for an inner fitting split.

    Returns
    -------
    anndata.AnnData
        A view containing controls and only permitted response rows.
    """
    pairs = _retained_pairs(order, k)
    if held_out_rank is not None:
        pairs = pairs[pairs["rank"] != held_out_rank]
    return data[is_control(data) | _pair_mask(data, pairs)]


def inner_folds(data: ad.AnnData, order: pd.DataFrame, k: int) -> Iterator[tuple]:
    """Yield k retained-pair validation folds with all controls available.

    Parameters
    ----------
    data : anndata.AnnData
        Original selected data. Removed responses never enter an evaluator.
    order : pandas.DataFrame
        Original pairs with seeded ranks.
    k : int
        Each fit keeps k-1 pairs per perturbation; one pair is validated.

    Yields
    ------
    tuple
        Training AnnData and a callable returning equally weighted pair MSE.
    """
    pairs = _retained_pairs(order, k)
    for rank in range(k):
        val_pairs = pairs[pairs["rank"] == rank]
        val = data[is_control(data) | _pair_mask(data, val_pairs)]
        yield retained_data(data, order, k, held_out_rank=rank), partial(_validation_mse, val=val)


def _validation_mse(model, val):
    targets = _pair_means(val)
    controls = val[is_control(val)]
    control_batches = batch_labels(controls)
    errors = []
    for batch, means in targets.groupby(level="batch", sort=True):
        labels = means.index.get_level_values("perturbation").tolist()
        prediction = model.predict_environment_average(labels, controls[control_batches == batch])
        errors.extend(mse(prediction, means.to_numpy()))
    if not errors or not np.isfinite(errors).all():
        raise ValueError("Validation requires nonempty, finite pair scores.")
    return {"mse": float(np.mean(errors))}


def _pair_means(data):
    responses = data.obs.well_type.eq("treatment").to_numpy()
    values = as_dense(data.X[responses]).astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Selected treatment responses must be finite.")
    index = pd.MultiIndex.from_arrays(
        [perturbation_labels(data)[responses], batch_labels(data)[responses]],
        names=["perturbation", "batch"],
    )
    return pd.DataFrame(values, index=index, columns=data.var_names).groupby(level=[0, 1]).mean()


def observed_average(data: ad.AnnData, perturbations: list[str]) -> pd.DataFrame:
    """Average treatment crop means equally over their observed batches.

    Parameters
    ----------
    data : anndata.AnnData
        Responses in original embedding units; controls are ignored.
    perturbations : list of str
        Nonempty, unique labels in requested result order.

    Returns
    -------
    pandas.DataFrame
        Perturbations by features, with equal batch weights.
    """
    labels = pd.Index(perturbations, name="perturbation")
    if labels.empty or not labels.is_unique:
        raise ValueError("perturbations must be nonempty and unique.")
    selected = data[np.isin(perturbation_labels(data), labels)]
    means = _pair_means(selected).groupby(level="perturbation").mean()
    if not labels.isin(means.index).all():
        raise ValueError("Observed responses are missing requested perturbations.")
    return means.loc[labels]


def virtual_average(model: object, controls: ad.AnnData, pairs: pd.DataFrame) -> pd.DataFrame:
    """Average predictions over each perturbation's original reference batches.

    Parameters
    ----------
    model : object
        Fitted transfer model supporting control-only predictions.
    controls : anndata.AnnData
        All selected proxy controls in original feature units.
    pairs : pandas.DataFrame
        Original reference pairs for eligible perturbations only.

    Returns
    -------
    pandas.DataFrame
        Sorted perturbation rows and original feature columns.
    """
    groups = defaultdict(list)
    for label, group in pairs.groupby("perturbation", sort=True):
        groups[tuple(sorted(group.batch.unique()))].append(label)
    tables = []
    batches = batch_labels(controls)
    for reference, labels in groups.items():
        result = virtual_experiments(
            model, controls[np.isin(batches, reference)], perturbations=labels
        )
        tables.append(
            pd.DataFrame(
                result.X, index=pd.Index(labels, name="perturbation"), columns=controls.var_names
            )
        )
    return pd.concat(tables).sort_index()


def score_estimates(
    full: pd.DataFrame,
    retained: pd.DataFrame,
    virtual: dict[str, pd.DataFrame],
    coverage: pd.DataFrame,
    *,
    seed: int,
    k: int,
) -> pd.DataFrame:
    """Score aligned estimates against the full mean in embedding coordinates.

    Parameters
    ----------
    full, retained : pandas.DataFrame
        Full and retained averages, identically labeled perturbations/features.
    virtual : dict of str to pandas.DataFrame
        Named virtual averages with the same labels.
    coverage : pandas.DataFrame
        Eligibility rows for this k, including batch counts.
    seed, k : int
        Experiment identifiers included in every result row.

    Returns
    -------
    pandas.DataFrame
        One MSE per seed, k, perturbation and model.
    """
    tables = []
    if not np.isfinite(full.to_numpy()).all():
        raise ValueError("Full means must be finite.")
    for name, estimate in {"retained_mean": retained, **virtual}.items():
        if not estimate.index.equals(full.index) or not estimate.columns.equals(full.columns):
            raise ValueError("Estimate labels and features must align with full means.")
        values = mse(estimate.to_numpy(), full.to_numpy())
        if not np.isfinite(values).all():
            raise ValueError("Estimate MSE must be finite.")
        tables.append(pd.DataFrame({"perturbation": full.index, "model": name, "value": values}))
    counts = ["perturbation", "n_reference_batches", "n_retained_batches", "n_removed_batches"]
    return (
        pd.concat(tables, ignore_index=True)
        .merge(coverage[counts], on="perturbation", validate="many_to_one")
        .assign(seed=seed, k=k, metric="mse")
    )
