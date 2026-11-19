"""Check prediction inputs without using held-out response values."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from genetic_perturbation_playground.data.transfer import batch_weights, matched_control_mean


@pytest.fixture
def test_cells():
    obs = pd.DataFrame(
        {
            "perturbation": ["control", "a", "c", "a", "control", "a", "c", "control"],
            "batch": ["b1", "b1", "b1", "b0", "b0", "b0", "b0", "b2"],
        },
        index=[str(i) for i in range(8)],
    )
    values = np.full((8, 2), np.nan)
    values[[0, 4, 7]] = [[10, 20], [2, 4], [90, 80]]
    return ad.AnnData(values, obs=obs)


def test_batch_weights_need_only_labels_and_include_control_only_batches(test_cells):
    test_cells.X = None
    np.testing.assert_allclose(
        batch_weights(test_cells, ["c", "a", "c"]),
        [[1 / 2, 1 / 2, 0], [2 / 3, 1 / 3, 0], [1 / 2, 1 / 2, 0]],
    )


@pytest.mark.parametrize("sparse_input", [False, True])
def test_matched_controls_follow_response_counts_with_unavailable_targets(test_cells, sparse_input):
    if sparse_input:
        test_cells.X = sparse.csr_matrix(test_cells.X)
    np.testing.assert_allclose(
        matched_control_mean(test_cells, ["c", "a", "c"]),
        [[6, 12], [14 / 3, 28 / 3], [6, 12]],
    )


@pytest.mark.parametrize("prepare", [batch_weights, matched_control_mean])
def test_prediction_inputs_require_controls_in_each_response_batch(test_cells, prepare):
    without_controls = test_cells[test_cells.obs.index != "4"]
    with pytest.raises(ValueError, match="Every batch with perturbed cells needs control cells"):
        prepare(without_controls, ["a"])


@pytest.mark.parametrize("label", ["b", "z"])
@pytest.mark.parametrize("prepare", [batch_weights, matched_control_mean])
def test_prediction_inputs_reject_missing_perturbations(test_cells, label, prepare):
    with pytest.raises(ValueError, match="Unknown labels"):
        prepare(test_cells, [label])
