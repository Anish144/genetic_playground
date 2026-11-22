"""Small control references for transfer prediction tests."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from genetic_perturbation_playground.model.batch_transfer import (
    KernelBridgeTransfer,
    NeuralBridgeTransfer,
)
from genetic_perturbation_playground.model.linear_transfer import FullLinearTransfer, LinearTransfer
from genetic_perturbation_playground.utils.standardisation import Standardised


@pytest.fixture
def virtual_train():
    rows, values = [], []
    for i in range(3):
        controls = np.array([[0, 2], [2, 4]], dtype=float) + 4 * i
        rows.extend([("control", f"b{i}")] * 2 + [("a", f"b{i}"), ("b", f"b{i}")])
        values.extend([*controls, controls.mean(0) + [10, 20], controls.mean(0) + [30, 40]])
    obs = pd.DataFrame(rows, columns=["perturbation", "batch"], index=list(map(str, range(12))))
    return ad.AnnData(np.array(values), obs=obs, var=pd.DataFrame(index=["g0", "g1"]))


@pytest.fixture
def virtual_reference():
    obs = pd.DataFrame(
        {"perturbation": ["control"] * 3, "batch": ["e0", "e0", "e1"]}, index=["r0", "r1", "r2"]
    )
    var = pd.DataFrame({"description": ["first", "second"]}, index=["g0", "g1"])
    return ad.AnnData(np.array([[0, 2], [2, 4], [5, 9]], dtype=float), obs=obs, var=var)


@pytest.fixture(
    params=["linear", "full_linear", "zero", "shift", "kernel", "per_x", "shared_2nn", "hybrid"]
)
def adapter_kind(request):
    return request.param


@pytest.fixture(params=[None, True, False], ids=["raw", "scaled", "centered"])
def transfer_model(virtual_train, adapter_kind, request):
    if adapter_kind in ("linear", "zero", "shift"):
        model = LinearTransfer(slope={"linear": None, "zero": 0.0, "shift": 1.0}[adapter_kind])
    elif adapter_kind == "full_linear":
        model = FullLinearTransfer()
    elif adapter_kind == "kernel":
        model = KernelBridgeTransfer(reg=0.1)
    else:
        model = NeuralBridgeTransfer(
            variant=adapter_kind, n_steps=2, n_features_w=2, width_w=4, reg=0.1
        )
    if request.param is not None:
        model = Standardised(model, w=request.param)
    return model.fit(virtual_train)
