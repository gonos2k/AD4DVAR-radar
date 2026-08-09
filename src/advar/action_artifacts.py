"""Allocation-free resource checks for durable action artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import zipfile

from torch import Tensor


def expanded_tensor_bytes(tensors: Iterable[Tensor | None]) -> int:
    """Return retained Tensor bytes without materializing host copies."""

    return sum(
        value.numel() * value.element_size()
        for value in tensors
        if value is not None
    )


def validate_artifact_directory(
    source: Path,
    *,
    expected_members: frozenset[str],
    maximum_members: int,
    maximum_file_bytes: int,
) -> None:
    """Reject unsafe or oversized durable members before reading content."""

    members = tuple(source.iterdir())
    if len(members) > maximum_members or {
        item.name for item in members
    } != expected_members:
        raise ValueError("durable intervention artifact members are invalid")
    if any(
        not item.is_file()
        or item.is_symlink()
        or item.stat().st_size > maximum_file_bytes
        for item in members
    ):
        raise ValueError("durable intervention artifact member is unsafe")


def preflight_npz_archive(
    path: Path,
    *,
    expected_members: frozenset[str],
    maximum_members: int,
    maximum_expanded_bytes: int,
) -> None:
    """Inspect a NumPy ZIP central directory before any array allocation."""

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as error:
        raise ValueError("durable intervention tensor archive is invalid") from error
    names = frozenset(
        item.filename.removesuffix(".npy")
        for item in infos
        if item.filename.endswith(".npy")
    )
    if (
        len(infos) > maximum_members
        or any(item.is_dir() for item in infos)
        or len(names) != len(infos)
        or names != expected_members
    ):
        raise ValueError("durable intervention tensor archive is invalid")
    if sum(item.file_size for item in infos) > maximum_expanded_bytes:
        raise ValueError("durable intervention tensor archive is too large")
