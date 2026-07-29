from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

import torch
from torch import Tensor


def dataclass_digest(value: Any) -> str:
    text = json.dumps(
        asdict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
