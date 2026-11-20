"""Atomic experiment files and explicit completion checks for local/POSIX runs."""

import fcntl
import hashlib
import json
import math
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
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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
    """Atomically save a finite JSON object.

    Parameters
    ----------
    path : pathlib.Path
        Destination, replaced only after successful serialization and flushing.
    payload : dict
        JSON-compatible values; NaN and infinity are rejected.
    """
    content = _json_bytes(payload)
    atomic_write(path, lambda stream: stream.write(content))


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return parsed


def read_json(path: Path) -> dict:
    """Read a committed JSON object, rejecting malformed or nonfinite values.

    Parameters
    ----------
    path : pathlib.Path
        File containing a JSON object, not a list or scalar.

    Returns
    -------
    dict
        Parsed checkpoint contents.

    Raises
    ------
    ValueError
        If the file cannot be read or contains invalid checkpoint JSON.
    """
    try:
        result = json.loads(
            path.read_text(), parse_constant=_invalid_constant, parse_float=_finite_float
        )
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Invalid checkpoint {path}; restore it or use a fresh output directory."
        ) from error


def fingerprint_file(path: Path) -> str:
    """Hash file contents without loading the entire file into memory.

    Parameters
    ----------
    path : pathlib.Path
        Readable file; names and modification times do not affect the hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint_settings(settings: dict) -> str:
    """Hash settings with canonical keys and explicit positive infinity.

    Parameters
    ----------
    settings : dict
        JSON-compatible settings, allowing positive infinity for model penalties.
        List ordering remains significant; NaN and negative infinity are rejected.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest, independent of dictionary insertion order.
    """
    return hashlib.sha256(_json_bytes(encode_settings(settings))).hexdigest()


def encode_settings(settings: dict) -> dict:
    """Represent positive infinity in settings using a reversible JSON tag.

    Parameters
    ----------
    settings : dict
        Nested settings with dictionary/list values. Positive infinity is
        permitted for penalties such as the hybrid model's linear_reg.

    Returns
    -------
    dict
        A copy using {"__float__": "+inf"} for positive infinity. Other values
        are unchanged; write_json still rejects NaN and negative infinity.
        The tag is reserved for this encoding. Use decode_settings to restore it.
    """
    return _map_settings(settings, encode=True)


def decode_settings(settings: dict) -> dict:
    """Restore tagged positive infinity in saved configuration values.

    Parameters
    ----------
    settings : dict
        Configuration returned by read_json after encode_settings was used.

    Returns
    -------
    dict
        A copy with tagged values restored to float('inf'). Do not use this
        decoder for metrics, which must remain finite.
    """
    return _map_settings(settings, encode=False)


def _map_settings(value, *, encode):
    if encode and isinstance(value, float) and value == math.inf:
        return {"__float__": "+inf"}
    if isinstance(value, dict):
        if not encode and value == {"__float__": "+inf"}:
            return math.inf
        return {key: _map_settings(item, encode=encode) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_map_settings(item, encode=encode) for item in value]
    return value


def ensure_manifest(directory: Path, expected: dict, *, resume: bool) -> dict:
    """Create an immutable run manifest or require matching scientific identity.

    Parameters
    ----------
    directory : pathlib.Path
        Locked output directory. Only logs, the lock and abandoned temporary
        manifest writes may predate the manifest.
    expected : dict
        schema_version and identity; optional provenance is retained from creation.
        Positive infinity in configuration is encoded with encode_settings.
    resume : bool
        Permit an existing compatible run. False never overwrites results.

    Returns
    -------
    dict
        The original manifest, or the newly created one, in its JSON-safe encoded
        form. Use decode_settings when numerical configuration values are needed.
    """
    expected = encode_settings(expected)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        if not resume:
            raise ValueError(f"Results exist in {directory}; use resume=true or a fresh directory.")
        original = read_json(path)
        if any(original.get(key) != expected[key] for key in ("schema_version", "identity")):
            raise ValueError(f"Run manifest {path} is incompatible; use a fresh output directory.")
        return original
    existing = [
        p
        for p in directory.iterdir()
        if p.name not in {"logs", ".lock"}
        and not (p.name.startswith(".manifest.json.") and p.name.endswith(".tmp"))
    ]
    if existing:
        raise ValueError(
            f"Cannot initialize nonempty output directory {directory} without a manifest."
        )
    write_json(path, expected)
    return expected


def write_completion(path: Path, artifacts: list[Path], *, identity: dict) -> None:
    """Commit a stage after hashing all its required artifacts.

    Parameters
    ----------
    path : pathlib.Path
        Completion manifest to write atomically after the stage outputs.
    artifacts : list of pathlib.Path
        Nonempty list of existing files beneath the manifest's parent directory.
    identity : dict
        JSON-compatible run and stage identifiers checked when resuming.
    """
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
        True for matching committed state; False if the manifest is absent.

    Raises
    ------
    ValueError
        For a mismatched identity, malformed manifest, missing artifact or
        changed artifact checksum. File existence alone is insufficient.
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
    """Hold an exclusive writer lock for a local/POSIX output directory.

    Parameters
    ----------
    directory : pathlib.Path
        Output directory, created if needed. Use this function in a with block.

    Yields
    ------
    None
        The lock is held during the block and released on exit or process death.

    Raises
    ------
    RuntimeError
        If another process already holds the directory's writer lock.
    """
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
