"""Exercise the local homepage HTTP routes; run while the lab server is up."""
from __future__ import annotations

from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json


BASE = "http://127.0.0.1:8766"


def request(path: str, body: bytes | None = None) -> dict[str, object]:
    method = "POST" if body is not None else "GET"
    query = Request(
        BASE + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        response = urlopen(query, timeout=20)
    except HTTPError as error:
        response = error
    with response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
        summary: dict[str, object] = {
            "method": method,
            "path": path,
            "status": response.status,
            "content_type": content_type,
            "bytes": len(payload),
        }
        if "application/json" in content_type:
            result = json.loads(payload)
            summary["error"] = result.get("error")
            if "case" in result:
                summary["lead_minutes"] = result["case"]["lead_minutes"]
                summary["scored_pixels"] = result["metrics"]["scored_pixels"]
        return summary


def main() -> None:
    scenarios = [
        ("/", None, 200),
        ("/app.js", None, 200),
        ("/styles.css", None, 200),
        ("/api/default", None, 200),
        ("/api/run", b"{}", 200),
        ("/api/run", b'{"use_background":false,"lead_minutes":180}', 200),
        ("/api/run", b'{"lead_minutes":25}', 400),
        ("/api/run", b'{"lead_minutes":60,"reference":{"lead_minutes":30}}', 400),
        ("/api/run", b"{", 400),
        ("/api/run", b"[]", 400),
        ("/api/run", b" " * 16385, 400),
        ("/api/run", b'{"use_background":1}', 400),
        ("/missing", None, 404),
        ("/missing", b"{}", 404),
    ]
    checks = [request(path, body) for path, body, _ in scenarios]
    for check, (_, _, expected) in zip(checks, scenarios):
        if check["status"] != expected:
            raise AssertionError(f"unexpected HTTP status: {check}")
    print(json.dumps({"base": BASE, "checks": checks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
