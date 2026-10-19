import anndata as ad
import numpy as np
import pandas as pd

from genetic_perturbation_playground.model.batch_transfer import KernelBridgeTransfer
from genetic_perturbation_playground.model.proxy_unconfounded import MarginalEmbeddingBridge
from genetic_perturbation_playground.utils.kernels import RBF


def _cells(rng, n_batches=4, perts=("a", "b", "c"), n_genes=5):
    obs = [("control", f"b{e}") for e in range(n_batches) for _ in range(rng.integers(5, 10))]
    obs += [
        (p, f"b{e}")
        for e in range(n_batches)
        for p in perts
        if rng.random() < 0.8
        for _ in range(rng.integers(1, 6))
    ]
    obs = pd.DataFrame(obs, columns=["perturbation", "batch"])
    return ad.AnnData(rng.normal(size=(len(obs), n_genes)), obs=obs)


def test_matches_cell_level_marginal_embedding_bridge():
    rng = np.random.default_rng(0)
    data = _cells(rng)
    train, test = data[data.obs["batch"] != "b3"], data[data.obs["batch"] == "b3"]
    model = KernelBridgeTransfer(reg=1e-2).fit(train)

    # the same estimator fit on cells: one-hot x with a delta kernel, w = control cells
    perts = model.perturbations_
    xs, ys, ws = [], [], []
    for e in sorted(set(train.obs["batch"])):
        b = train[train.obs["batch"] == e]
        ctrl = b.obs["perturbation"] == "control"
        xs.append(np.eye(len(perts))[np.searchsorted(perts, b.obs["perturbation"][~ctrl])])
        ys.append(b.X[~ctrl.to_numpy()])
        ws.append(b.X[ctrl.to_numpy()])
    query = sorted(set(test.obs["perturbation"]) - {"control"})
    x_query = np.eye(len(perts))[np.searchsorted(perts, query)]
    w_test = test.X[(test.obs["perturbation"] == "control").to_numpy()]
    bridge = MarginalEmbeddingBridge(RBF(lengthscales=1e-2), model.kernel_w_, reg=1e-2)
    expected = np.stack(
        [bridge.fit(xs, [y[:, g] for y in ys], ws).predict(x_query, w_test) for g in range(5)], 1
    )
    np.testing.assert_allclose(model.predict(query, test), expected, rtol=1e-8, atol=1e-10)
