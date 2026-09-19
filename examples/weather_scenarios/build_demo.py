"""Build the standalone weather-scenarios HTML demo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios import build_dataset, dumps_dataset  # noqa: E402


MARKER = "__DEMO_DATA__"


def render_template(template: str, payload: dict[str, object]) -> str:
    if template.count(MARKER) != 1:
        raise ValueError(f"template must contain exactly one {MARKER} marker")
    # The payload is normally placed inside a script element; escape HTML-open
    # characters while retaining valid JSON and Unicode in the page.
    encoded = dumps_dataset(payload).replace("<", "\\u003c")
    return template.replace(MARKER, encoded)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("template.html"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("index.html"),
        help="HTML output path; omitted only with --dry-run",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = build_dataset()
    if args.json_output is not None:
        _atomic_write(args.json_output, dumps_dataset(payload))
    if args.dry_run:
        print(dumps_dataset(payload))
        return 0
    _atomic_write(args.output, render_template(args.template.read_text(encoding="utf-8"), payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
