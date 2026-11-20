import importlib
import multiprocessing
from pathlib import Path

import pytest


@pytest.fixture
def state():
    return importlib.import_module("genetic_perturbation_playground.experiments.checkpoints")


def test_settings_encode_infinity_without_allowing_nonfinite_scores(state, tmp_path):
    settings = {"neural": {"linear_reg": float("inf")}, "grid": [None, 0.0, float("inf")]}
    encoded = state.encode_settings(settings)
    state.write_json(tmp_path / "settings.json", encoded)
    assert state.decode_settings(state.read_json(tmp_path / "settings.json")) == settings
    assert state.fingerprint_settings(settings) == state.fingerprint_settings(encoded)
    with pytest.raises(ValueError):
        state.write_json(tmp_path / "scores.json", {"mse": float("inf")})
    with pytest.raises(ValueError):
        state.fingerprint_settings({"penalty": float("nan")})


def test_atomic_write_preserves_previous_file_on_failure(state, tmp_path):
    path = tmp_path / "result.json"
    state.write_json(path, {"mse": 2.0})

    def fail(stream):
        stream.write(b"partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        state.atomic_write(path, fail)
    assert state.read_json(path) == {"mse": 2.0}
    assert list(tmp_path.iterdir()) == [path]


def test_completion_requires_all_matching_artifacts(state, tmp_path):
    artifact = tmp_path / "scores.csv"
    artifact.write_text("mse\n2.0\n")
    marker = tmp_path / "complete.json"
    identity = {"seed": 0, "k": 2}
    assert not state.is_complete(marker, identity=identity)
    state.write_completion(marker, [artifact], identity=identity)
    assert state.is_complete(marker, identity=identity)
    with pytest.raises(ValueError, match="identity"):
        state.is_complete(marker, identity={"seed": 1})
    artifact.write_text("truncated")
    with pytest.raises(ValueError, match="scores.csv"):
        state.is_complete(marker, identity=identity)
    artifact.unlink()
    with pytest.raises(ValueError, match="scores.csv"):
        state.is_complete(marker, identity=identity)


@pytest.mark.parametrize(
    "changed", [{"input": "changed"}, {"variant": "hybrid"}, {"k_values": [2, 3]}, {"grid": [1, 3]}]
)
def test_manifest_rejects_incompatible_resume(state, tmp_path, changed):
    identity = {"input": "abc", "variant": "per_x", "k_values": [2], "grid": [1, 2]}
    expected = {"schema_version": 1, "identity": identity}
    state.ensure_manifest(tmp_path, expected, resume=True)
    assert state.ensure_manifest(tmp_path, expected, resume=True) == expected
    with pytest.raises(ValueError, match="incompatible"):
        state.ensure_manifest(tmp_path, expected | {"identity": identity | changed}, resume=True)
    with pytest.raises(ValueError, match="resume"):
        state.ensure_manifest(tmp_path, expected, resume=False)


def test_manifest_preserves_provenance_and_allows_only_operational_files(state, tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / ".lock").touch()
    original = {"schema_version": 1, "identity": {"data": "abc"}, "provenance": {"path": "old"}}
    state.ensure_manifest(tmp_path, original, resume=False)
    result = state.ensure_manifest(
        tmp_path, original | {"provenance": {"path": "moved"}}, resume=True
    )
    assert result == original
    another = tmp_path / "other"
    another.mkdir()
    (another / "scores.csv").write_text("old results")
    with pytest.raises(ValueError, match="nonempty"):
        state.ensure_manifest(another, original, resume=True)


@pytest.mark.parametrize(
    "content", ['{"mse": NaN}', '{"mse": Infinity}', '{"mse": 1e9999}', "{", "[]"]
)
def test_read_json_rejects_malformed_committed_files(state, tmp_path, content):
    path = tmp_path / "trial.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="trial.json"):
        state.read_json(path)


def test_write_json_rejects_nonfinite_values(state, tmp_path):
    with pytest.raises(ValueError):
        state.write_json(tmp_path / "bad.json", {"mse": float("nan")})
    assert not (tmp_path / "bad.json").exists()


def test_manifest_can_resume_after_kill_during_first_atomic_write(state, tmp_path):
    (tmp_path / ".manifest.json.interrupted.tmp").write_bytes(b'{"schema_')
    expected = {"schema_version": 1, "identity": {"data": "abc"}}
    assert state.ensure_manifest(tmp_path, expected, resume=True) == expected


def _try_lock(directory, queue):
    from genetic_perturbation_playground.experiments.checkpoints import run_lock

    try:
        with run_lock(Path(directory)):
            queue.put("acquired")
    except RuntimeError:
        queue.put("busy")


def test_run_lock_blocks_second_process_and_releases_on_exit(state, tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with state.run_lock(tmp_path):
        process = context.Process(target=_try_lock, args=(str(tmp_path), queue))
        process.start()
        assert queue.get(timeout=20) == "busy"
        process.join(20)
        assert process.exitcode == 0
    process = context.Process(target=_try_lock, args=(str(tmp_path), queue))
    process.start()
    assert queue.get(timeout=20) == "acquired"
    process.join(20)
    assert process.exitcode == 0


def test_fingerprints_are_content_based_and_settings_order_independent(state, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert state.fingerprint_file(a) == state.fingerprint_file(b)
    b.write_bytes(b"different")
    assert state.fingerprint_file(a) != state.fingerprint_file(b)
    assert state.fingerprint_settings({"a": 1, "b": 2}) == state.fingerprint_settings(
        {"b": 2, "a": 1}
    )
