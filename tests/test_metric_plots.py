import numpy as np
import pandas as pd

from genetic_perturbation_playground.utils import plotting


def test_training_batch_report_keeps_held_out_batches_separate(tmp_path):
    scores = pd.DataFrame(
        {
            "perturbation": ["a", "a", "b", "c", "d"],
            "batch": ["b1", "b2", "b1", "b2", "b2"],
            "n_training_batches": [1, 1, 1, 3, 3],
            "metric": "mse",
            "model": "bridge",
            "value": [1.0, 3.0, 6.0, 0.0, np.nan],
        }
    )
    plotting.report_by_training_batches(scores, tmp_path)
    for batch, values, counts in [
        ("b1", {"a": 1.0, "b": 6.0}, {"a": 1, "b": 1}),
        ("b2", {"a": 3.0, "c": 0.0}, {"a": 1, "c": 3}),
    ]:
        out = tmp_path / f"batch_{batch}"
        points = pd.read_csv(out / "per_perturbation.csv").set_index("perturbation")
        assert points["value"].to_dict() == values
        assert points["n_training_batches"].to_dict() == counts
        assert set(points["batch"]) == {batch}
        assert (out / "mse.png").stat().st_size > 0
