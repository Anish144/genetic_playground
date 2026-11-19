"""Compare the ridge solvers with saved pre-refactor calculations on synthetic data.

Usage:
    python scripts/benchmark_ridge_solver.py --baseline /tmp/before.py --output outputs/ridge_benchmark.json

Benchmarks include the forward/backward training objective and final coefficient
reconstruction. Data preparation is outside timing. Old/new measurements alternate;
medians and individual samples are recorded. CPU allocation traffic is measured
separately from timing; it does not measure peak live memory or RSS.
The linear comparison sets reg=0 to match the previous least-squares calculation.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Timer

from genetic_perturbation_playground.model import proxy_unconfounded_neural as current
from genetic_perturbation_playground.utils.ridge import LinearRidgeSolver

MODELS = (
    "NeuralPerXEmbeddingBridge",
    "NeuralSharedEmbeddingBridge",
    "NeuralHybridEmbeddingBridge",
)
CASES = ((32, 8, 64, 4), (128, 24, 384, 8))  # categories, environments, outputs, embedding size


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("ridge_benchmark_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def environments(groups, envs, genes):
    rng = np.random.default_rng(19)
    categories = [np.flatnonzero(rng.random(groups) < 0.75) for _ in range(envs)]
    xs = [np.eye(groups)[labels] for labels in categories]
    ys = [rng.normal(size=(len(labels), genes)) for labels in categories]
    ws = [rng.normal(size=(24, genes)) for _ in range(envs)]
    counts = [rng.integers(1, 20, size=len(labels)) for labels in categories]
    return xs, ys, ws, counts


def setup(module, name, samples, features):
    model = getattr(module, name)(
        n_features_w=features,
        width_w=16,
        depth=1,
        reg=0.003,
        cross_fit_folds=4,
        seed=17,
    )
    data = model._to_tensors(*samples)
    torch.manual_seed(model.seed)
    model._build_networks(data)
    folds = torch.arange(len(samples[2])) % 4
    model._env_folds = lambda _: folds
    model._solve_A(data)
    return model, data


def training_step(model, data):
    for parameter in model.psi_.parameters():
        parameter.grad = None
    loss = model._training_loss(data)
    loss.backward()
    return loss.detach()


def parity(old, new):
    old_model, old_data = old
    new_model, new_data = new
    old_loss, new_loss = training_step(*old), training_step(*new)
    torch.testing.assert_close(old_loss, new_loss, rtol=1e-10, atol=1e-10)
    max_gradient_error = 0.0
    for before, after in zip(old_model.psi_.parameters(), new_model.psi_.parameters(), strict=True):
        torch.testing.assert_close(before.grad, after.grad, rtol=1e-4, atol=1e-6)
        max_gradient_error = max(max_gradient_error, (before.grad - after.grad).abs().max().item())
    old_model._solve_A(old_data)
    new_model._solve_A(new_data)
    names = (
        ["A_"]
        if hasattr(old_model, "A_")
        else [
            "intercept_",
            "shared_coef_",
            "deviation_coef_",
            "linear_coef_",
        ]
    )
    for name in names:
        if hasattr(old_model, name):
            torch.testing.assert_close(
                getattr(old_model, name), getattr(new_model, name), rtol=1e-10, atol=1e-10
            )
    return {
        "loss_abs_error": abs(old_loss.item() - new_loss.item()),
        "max_gradient_abs_error": max_gradient_error,
        "coefficients_match": True,
    }


def timings(functions, repeats, seconds, threads):
    samples = {name: [] for name in functions}
    for repeat in range(repeats):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            measured = Timer("fn()", globals={"fn": functions[name]}, num_threads=threads)
            samples[name].append(measured.blocked_autorange(min_run_time=seconds).median * 1000)
    before, after = (statistics.median(samples[name]) for name in ("before", "after"))
    return {
        "before_ms": before,
        "after_ms": after,
        "change_pct": 100 * (after / before - 1),
        "samples_ms": samples,
    }


def tensor_memory(fn):
    # Count operator-local allocations rather than double-counting nested operator totals.
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as trace:
        fn()
    allocated = sum(max(event.self_cpu_memory_usage, 0) for event in trace.events())
    return {"allocated_bytes": allocated}


def old_linear_slope(stats):
    """Exact pre-refactor analytic formula, kept as the linear benchmark reference."""
    y, w, n = stats["y_mean"], stats["w_mean"], stats["n"][:, None]
    num = (stats["W"] * stats["batch_y_sum"]).sum(axis=0) - (n * y * w).sum(axis=0)
    den = (stats["batch_n"][:, None] * stats["W"] ** 2).sum(axis=0) - (n * w**2).sum(axis=0)
    return num / den


def linear_statistics(samples):
    xs, ys, ws, counts = samples
    groups, genes = xs[0].shape[1], ys[0].shape[1]
    totals = np.zeros(groups)
    targets = np.zeros((groups, genes))
    proxies = np.zeros_like(targets)
    control_means = np.stack([w.mean(axis=0) for w in ws])
    batch_targets = []
    for x, y, count, control in zip(xs, ys, counts, control_means, strict=True):
        labels = x.argmax(axis=1)
        np.add.at(totals, labels, count)
        np.add.at(targets, labels, count[:, None] * y)
        np.add.at(proxies, labels, count[:, None] * control)
        batch_targets.append((count[:, None] * y).sum(axis=0))
    assert (totals > 0).all()
    return {
        "n": totals,
        "y_mean": targets / totals[:, None],
        "w_mean": proxies / totals[:, None],
        "W": control_means,
        "batch_n": np.array([c.sum() for c in counts]),
        "batch_y_sum": np.stack(batch_targets),
    }


def numpy_peak(fn):
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=0.15)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    baseline = load_baseline(args.baseline)
    report = {
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "threads": args.threads,
        "repeats": args.repeats,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "cases": [],
    }
    for groups, envs, genes, features in CASES:
        samples = environments(groups, envs, genes)
        shape = {"categories": groups, "environments": envs, "outputs": genes, "features": features}
        for name in MODELS:
            old, new = (setup(module, name, samples, features) for module in (baseline, current))
            row = {"model": name, "shape": shape, "parity": parity(old, new)}
            for phase in ("training", "reconstruction"):
                functions = {
                    key: (lambda pair=pair: training_step(*pair))
                    if phase == "training"
                    else (lambda pair=pair: pair[0]._solve_A(pair[1]))
                    for key, pair in (("before", old), ("after", new))
                }
                for fn in functions.values():
                    for _ in range(3):
                        fn()
                row[phase] = timings(functions, args.repeats, args.seconds, args.threads)
                row[phase]["memory"] = {key: tensor_memory(fn) for key, fn in functions.items()}
                timing = row[phase]
                print(
                    f"{groups}/{envs}/{genes} {name} {phase}: "
                    f"{timing['before_ms']:.3f} -> {timing['after_ms']:.3f} ms "
                    f"({timing['change_pct']:+.1f}%)",
                    flush=True,
                )
            report["cases"].append(row)
            del old, new, functions
            gc.collect()
        stats = linear_statistics(samples)
        linear_solver = LinearRidgeSolver(reg=0.0)
        np.testing.assert_allclose(
            old_linear_slope(stats), linear_solver.solve(stats), rtol=1e-10, atol=1e-12
        )
        functions = {
            "before": lambda stats=stats: old_linear_slope(stats),
            "after": lambda stats=stats, solver=linear_solver: solver.solve(stats),
        }
        row = {
            "model": "linear",
            "reg": 0.0,
            "shape": shape,
            "solve": timings(functions, args.repeats, args.seconds, args.threads),
            "peak_numpy_bytes": {name: numpy_peak(fn) for name, fn in functions.items()},
            "coefficients_match": True,
        }
        report["cases"].append(row)
        print(f"{groups}/{envs}/{genes} linear: {row['solve']['change_pct']:+.1f}%", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
