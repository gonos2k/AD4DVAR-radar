"""Reproduce inherited async observer records using the pinned pre-fix source."""

import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BASE = "918e1c004182199617282cb14293d7bdcdd9955d"


def run():
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary) / "advar"
        shutil.copytree("src/advar", package)
        source_hashes = {}
        for name in ("transport.py", "matrix_free.py"):
            source = subprocess.check_output(["git", "show", f"{BASE}:src/advar/{name}"])
            (package / name).write_bytes(source)
            source_hashes[name] = hashlib.sha256(source).hexdigest()
        sys.path.insert(0, temporary)
        import torch
        import advar
        from advar import transport
        from advar.matrix_free import observe_pcg_calls, pcg

        assert Path(advar.__file__).resolve().is_relative_to(Path(temporary).resolve())
        stage_records = []
        pcg_records = []

        def stage(marker):
            q = torch.full((4, 5), marker, dtype=torch.float64)
            qx = torch.full((4, 6), 0.1, dtype=torch.float64)
            qy = torch.full((5, 5), -0.1, dtype=torch.float64)
            edges = (
                torch.full((4,), marker, dtype=torch.float64),
                torch.full((4,), marker, dtype=torch.float64),
                torch.full((5,), marker, dtype=torch.float64),
                torch.full((5,), marker, dtype=torch.float64),
            )
            transport._euler_minmod(q, qx, qy, edges, q.new_tensor(0.05), q.new_tensor(1.0))

        def solve(marker):
            rhs = torch.tensor([marker, 2 * marker], dtype=torch.float64)
            diagonal = torch.tensor([2.0, 3.0], dtype=torch.float64)
            pcg(lambda value: diagonal * value, rhs, rtol=1e-13)

        def observer(original):
            def recorded(operator, rhs, **kwargs):
                pcg_records.append(float(rhs[0]))
                return original(operator, rhs, **kwargs)
            return recorded

        async def scenario():
            release = asyncio.Event()

            async def child():
                await release.wait()
                stage(2.0)
                solve(2.0)

            with transport.observe_minmod_stages(
                lambda q, *_: stage_records.append(float(q[0, 0]))
            ), observe_pcg_calls(observer):
                stage(1.0)
                solve(1.0)
                task = asyncio.create_task(child())
            release.set()
            await task

        asyncio.run(scenario())
        return {
            "base_commit": BASE,
            "baseline_source_sha256": source_hashes,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "stage_records": stage_records,
            "pcg_records": pcg_records,
            "late_callback_observed": stage_records == [1.0, 2.0] and pcg_records == [1.0, 2.0],
        }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
