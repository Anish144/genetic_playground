"""Small control references for transfer prediction tests."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest


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
