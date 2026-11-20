"""Atomic experiment files and explicit completion checks for local/POSIX runs."""

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


def atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """Replace a file only after its complete contents have been flushed.

    Parameters
    ----------
    path : pathlib.Path
        Destination; its parent is created if necessary.
    writer : callable
        Write bytes to the supplied stream. Failure preserves an old destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(payload):
    return json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def write_json(path: Path, payload: dict) -> None:
    """Atomically save a JSON object, rejecting nonfinite numerical values."""
    content = _json_bytes(payload)
    atomic_write(path, lambda stream: stream.write(content))


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path: Path) -> dict:
    """Read a committed JSON object or report its path and recovery action."""
    try:
        result = json.loads(path.read_text(), parse_constant=_invalid_constant)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Invalid checkpoint {path}; restore it or use a fresh output directory."
        ) from error


def fingerprint_file(path: Path) -> str:
    """Return a streaming SHA-256 digest of a file's contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint_settings(settings: dict) -> str:
    """Hash canonical finite JSON, independent of dictionary insertion order."""
    return hashlib.sha256(_json_bytes(settings)).hexdigest()


def ensure_manifest(directory: Path, expected: dict, *, resume: bool) -> dict:
    """Create an immutable run manifest or require matching scientific identity.

    Parameters
    ----------
    directory : pathlib.Path
        Locked output directory. Only logs and the lock may predate the manifest.
    expected : dict
        schema_version and identity; optional provenance is retained from creation.
    resume : bool
        Permit an existing compatible run. False never overwrites results.

    Returns
    -------
    dict
        The original manifest, or the newly created one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        if not resume:
            raise ValueError(f"Results exist in {directory}; use resume=true or a fresh directory.")
        original = read_json(path)
        if any(original.get(key) != expected[key] for key in ("schema_version", "identity")):
            raise ValueError(f"Run manifest {path} is incompatible; use a fresh output directory.")
        return original
    existing = [p for p in directory.iterdir() if p.name not in {"logs", ".lock"}]
    if existing:
        raise ValueError(
            f"Cannot initialize nonempty output directory {directory} without a manifest."
        )
    write_json(path, expected)
    return expected


def write_completion(path: Path, artifacts: list[Path], *, identity: dict) -> None:
    """Commit a stage only after all its artifacts exist, with content checksums."""
    if not artifacts:
        raise ValueError("Completion requires at least one artifact.")
    checksums = {str(item.relative_to(path.parent)): fingerprint_file(item) for item in artifacts}
    write_json(path, {"schema_version": 1, "identity": identity, "artifacts": checksums})


def is_complete(path: Path, *, identity: dict) -> bool:
    """Check a stage identity and every artifact before skipping its work.

    Parameters
    ----------
    path : pathlib.Path
        Completion manifest, written last by its stage.
    identity : dict
        Expected seed/k or other stage identity.

    Returns
    -------
    bool
        False only when the manifest is absent. Damaged committed state raises.
    """
    if not path.exists():
        return False
    record = read_json(path)
    if record.get("schema_version") != 1 or record.get("identity") != identity:
        raise ValueError(f"Completion identity mismatch at {path}; use a fresh output directory.")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"Invalid artifact list in {path}; restore the checkpoint.")
    for name, expected in artifacts.items():
        artifact = path.parent / name
        if not artifact.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError(f"Invalid artifact path in {path}: {name}")
        if not artifact.is_file() or fingerprint_file(artifact) != expected:
            raise ValueError(
                f"Damaged checkpoint artifact {artifact}; restore it or use a fresh directory."
            )
    return True


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    """Hold one OS-released writer lock for a local/POSIX output directory."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing to {directory}.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
