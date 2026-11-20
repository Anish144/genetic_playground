"""Average perturbation responses over a common reference population.

Each reference batch defines an environment through its control samples.
Outputs remain in the target units of the fitted model; Standardised restores
original units. No model is fitted or changed here.
"""

from collections.abc import Mapping

import anndata as ad
import numpy as np
import pandas as pd

from genetic_perturbation_playground.data.transfer import (
    batch_labels,
    label_positions,
    observed_batch_means,
    perturbation_labels,
    reference_weights,
    validate_features,
)


def virtual_experiments(
    model: object,
    reference: ad.AnnData,
    *,
    perturbations: list[str] | None = None,
    environment_weights: Mapping[str, float] | None = None,
    observed: ad.AnnData | None = None,
) -> ad.AnnData:
    """Compute bridge averages and optionally retain observed pair means.

    For perturbation x, environment e and output g, define

        prediction[x,e,g] = mean_j h_g(x, control[e,j])
        bridge[x,g] = sum_e p[e] * prediction[x,e,g]
        observed_fill[x,g] = bridge[x,g]
            + sum_{e observed for x} p[e] * (observed_mean[x,e,g] - prediction[x,e,g])

    Parameters
    ----------
    model : object
        Fitted transfer model with perturbations_, feature_names_in_ and
        predict_environment_average. Standardised models return original units.
    reference : anndata.AnnData
        Controls only, with perturbation and batch labels in obs and features
        in training order. These controls define the reference environments.
    perturbations : list of str or None, default=None
        Nonempty, unique fitted labels in result order. None uses all fitted labels.
    environment_weights : mapping of str to float or None, default=None
        Probability of every reference batch. None weights environments equally,
        not by sample counts. Probabilities must be finite, nonnegative and sum
        to one within absolute tolerance 1e-8.
    observed : anndata.AnnData or None, default=None
        Measured responses in original target units for a Standardised model,
        otherwise in fitted target units. Columns must match training. Controls
        and rows outside requested perturbations/reference batches are ignored.
        Missing pairs have no rows; selected values must be finite. Empty data
        is allowed. Supply only responses available to the estimator, excluding
        those reserved for evaluation.

    Returns
    -------
    anndata.AnnData
        Requested perturbations by outputs. X contains bridge averages; var is
        copied from reference. uns['environment_weights'] records probabilities.
        When observed is supplied, layers['observed_fill'] contains the corrected
        estimate and obs['observed_weight'] its observed probability coverage.
        Invalid labels, features, controls or observations raise ValueError.

    Notes
    -----
    The causal interpretation requires the bridge assumptions. Equal environment
    weights describe a chosen reference population, not necessarily the sampled
    population. Observed means must estimate the intended within-environment
    intervention responses; the correction is not automatically doubly robust.

    Examples
    --------
    >>> from genetic_perturbation_playground.model.linear_transfer import LinearTransfer
    >>> obs = pd.DataFrame(
    ...     {'perturbation': ['control', 'a', 'control', 'a'],
    ...      'batch': ['b0', 'b0', 'b1', 'b1']}, index=['0', '1', '2', '3'])
    >>> train = ad.AnnData(np.array([[0.], [1.], [2.], [3.]]), obs=obs)
    >>> model = LinearTransfer(slope=1, standardise_w=False).fit(train)
    >>> controls = train[[0, 2]]
    >>> measured = train[[1]].copy()
    >>> measured.X[:] = 9.0
    >>> result = virtual_experiments(model, controls, observed=measured)
    >>> result.X.tolist()
    [[2.0]]
    >>> result.layers['observed_fill'].tolist()
    [[6.0]]
    >>> result.obs['observed_weight'].tolist()
    [0.5]
    """
    weights = reference_weights(
        reference, feature_names=model.feature_names_in_, environment_weights=environment_weights
    )
    labels = list(model.perturbations_ if perturbations is None else perturbations)
    index = pd.Index(labels, name="perturbation")
    if not len(index) or not index.is_unique:
        raise ValueError("perturbations must be nonempty and unique.")
    label_positions(model.perturbations_, labels)
    if observed is not None:
        _validate_observed(observed, model.feature_names_in_)
    prediction = model.predict_environment_average(
        labels, reference, environment_weights=weights.to_dict()
    )
    result = ad.AnnData(
        prediction,
        obs=pd.DataFrame(index=index),
        var=reference.var.copy(),
        uns={"environment_weights": weights.to_dict()},
    )
    if observed is None:
        return result

    filled = np.array(prediction, dtype=np.result_type(prediction.dtype, float), copy=True)
    coverage = np.zeros(len(labels))
    batches = batch_labels(reference)
    observed_rows = _observed_rows_by_batch(observed, labels)
    for batch, probability in weights.items():
        if batch not in observed_rows:
            continue
        means = observed_batch_means(observed[observed_rows[batch]], labels, batch)
        if probability == 0 or means.empty:
            continue
        positions = index.get_indexer(means.index)
        predicted = model.predict_environment_average(
            means.index.tolist(), reference[batches == batch]
        )
        filled[positions] += probability * (means.to_numpy() - predicted)
        coverage[positions] += probability
    result.layers["observed_fill"] = filled
    result.obs["observed_weight"] = coverage
    return result


def _validate_observed(observed: ad.AnnData, feature_names: np.ndarray) -> None:
    """Check response schema; finite-value checks apply only to selected pairs."""
    validate_features(observed, feature_names)
    for column in ("perturbation", "batch"):
        if column not in observed.obs:
            raise ValueError(f"observed.obs must contain {column} labels.")
    responses = ~observed.obs["perturbation"].eq("control").fillna(False)
    if observed.obs.loc[responses, ["perturbation", "batch"]].isna().any().any():
        raise ValueError("observed responses must have nonmissing perturbation and batch labels.")


def _observed_rows_by_batch(observed: ad.AnnData, perturbations: list[str]) -> dict:
    """Group requested row positions once, without reading the response matrix."""
    keep = pd.Index(perturbation_labels(observed)).isin(perturbations)
    batches = pd.Series(batch_labels(observed))
    selected = batches[keep]
    return {
        batch: rows.to_numpy()
        for batch, rows in selected.groupby(selected, sort=False).groups.items()
    }
