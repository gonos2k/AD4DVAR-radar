from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import advar.runtime_closure as runtime_closure_module


class _FakeDistribution:
    def __init__(self, root: Path, name: str, version: str) -> None:
        self._root = root
        self.metadata = {"Name": name}
        self.version = version
        self.files = (Path("advar/__init__.py"),)

    def locate_file(self, path: Path) -> Path:
        return self._root / path


class RuntimeClosureTests(unittest.TestCase):
    def test_runtime_file_snapshot_rejects_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / "target.py"
            link = base / "shadow.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "opened safely"):
                runtime_closure_module._runtime_file_snapshot(
                    link,
                    deployable=False,
                )

    def test_active_import_path_identity_is_relocatable(self) -> None:
        retained: list[list[dict[str, object]]] = []
        for name in ("first", "second"):
            with tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary) / name
                stdlib = base / "stdlib"
                site = base / "site-packages"
                dynamic = stdlib / "lib-dynload"
                for path in (stdlib, site, dynamic):
                    path.mkdir(parents=True, exist_ok=True)
                with patch.object(
                    runtime_closure_module.sys,
                    "path",
                    [str(stdlib), str(dynamic), str(site)],
                ):
                    retained.append(
                        runtime_closure_module.active_import_path_snapshot(
                            import_roots=(site,),
                            stdlib_root=stdlib,
                            deployable=False,
                        )
                    )
        self.assertEqual(retained[0], retained[1])

    def test_active_import_path_rejects_an_ambient_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stdlib = base / "stdlib"
            site = base / "site-packages"
            extra = base / "shadow-root"
            for path in (stdlib, site, extra):
                path.mkdir()
            with patch.object(
                runtime_closure_module.sys,
                "path",
                [str(stdlib), str(site), str(extra)],
            ), self.assertRaisesRegex(ValueError, "unexpected import root"):
                runtime_closure_module.active_import_path_snapshot(
                    import_roots=(site,),
                    stdlib_root=stdlib,
                    deployable=False,
                )

    def test_product_runtime_snapshot_rejects_shadow_and_writable_deployable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "site-packages"
            source = root / "advar/__init__.py"
            source.parent.mkdir(parents=True)
            source.write_text("", encoding="utf-8")
            distribution = _FakeDistribution(
                root,
                "advar-radar-nowcast",
                "0.92.0",
            )
            interpreter = {
                "contract": "advar-python-interpreter-closure-v1",
                "interpreter_closure_digest": "7" * 64,
            }
            with (
                patch.object(
                    runtime_closure_module,
                    "_runtime_import_roots",
                    return_value=(root,),
                ),
                patch.object(
                    runtime_closure_module.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
                patch.object(
                    runtime_closure_module.importlib.metadata,
                    "distributions",
                    return_value=(distribution,),
                ),
                patch.object(
                    runtime_closure_module,
                    "_interpreter_closure_snapshot",
                    return_value=interpreter,
                ),
            ):
                snapshot = runtime_closure_module.snapshot_current_runtime(
                    runtime_mode="candidate-smoke"
                )
                runtime_closure_module.validate_current_runtime_closure(
                    expected_runtime_tree_digest=str(
                        snapshot["runtime_tree_digest"]
                    ),
                    expected_interpreter_closure_digest="7" * 64,
                    runtime_mode="candidate-smoke",
                )

                shadow = root / "advar/shadow.py"
                shadow.write_text("raise SystemExit", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unowned file"):
                    runtime_closure_module.snapshot_current_runtime(
                        runtime_mode="candidate-smoke"
                    )
                shadow.unlink()

                with self.assertRaisesRegex(ValueError, "root-owned"):
                    runtime_closure_module.snapshot_current_runtime(
                        runtime_mode="deployable"
                    )


if __name__ == "__main__":
    unittest.main()
