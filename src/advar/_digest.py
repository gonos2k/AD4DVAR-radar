from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

import torch
from torch import Tensor


def json_digest(value: Any) -> str:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataclass_digest(value: Any) -> str:
    return json_digest(asdict(value))


def tensor_digest(value: Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    metadata = json.dumps(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(metadata)
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_sha256_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
