from __future__ import annotations

import unittest
from unittest.mock import patch

from advar._runtime import numerical_runtime_manifest


class NumericalRuntimeManifestTest(unittest.TestCase):
    def test_runtime_digest_changes_with_device_model(self) -> None:
        with patch(
            "advar._runtime._execution_device_identity",
            return_value={"type": "mps", "index": 0, "model": "Apple M2"},
        ):
            first = numerical_runtime_manifest("mps")
        with patch(
            "advar._runtime._execution_device_identity",
            return_value={"type": "mps", "index": 0, "model": "Apple M4"},
        ):
            second = numerical_runtime_manifest("mps")

        self.assertEqual(first.compatibility_digest, second.compatibility_digest)
        self.assertNotEqual(first.exact_digest, second.exact_digest)

    def test_runtime_digest_changes_with_matmul_precision(self) -> None:
        with patch(
            "advar._runtime.torch.get_float32_matmul_precision",
            return_value="highest",
        ):
            first = numerical_runtime_manifest("cpu")
        with patch(
            "advar._runtime.torch.get_float32_matmul_precision",
            return_value="medium",
        ):
            second = numerical_runtime_manifest("cpu")

        self.assertNotEqual(first.compatibility_digest, second.compatibility_digest)
        self.assertNotEqual(first.exact_digest, second.exact_digest)


if __name__ == "__main__":
    unittest.main()
