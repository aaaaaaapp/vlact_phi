#!/usr/bin/env python3
"""Serve the mentor brief as a read-only landing page, without directory lists."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = (REPO_ROOT / "reports").resolve()
DOC_ROOT = (REPO_ROOT / "docs").resolve()
LANDING_PAGE = "/reports/physical_action_token_mentor_brief_20261014.html"
REPORT_SUFFIXES = {".html", ".js", ".css", ".mp4"}
LINKED_DOCS = {
    "PHYSICAL_ACTION_TOKEN_PLAN.md",
    "ROBOTWIN_PURE_PI_EVAL_RECOVERY_20261013.md",
}


class ReportHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        if request_path == "/":
            request_path = LANDING_PAGE
        relative = PurePosixPath(request_path.lstrip("/"))
        denied = str(REPO_ROOT / "__reports_not_found__")
        if ".." in relative.parts or len(relative.parts) < 2:
            return denied
        area = relative.parts[0]
        if area == "reports":
            allowed_root = REPORT_ROOT
            allowed = relative.suffix.lower() in REPORT_SUFFIXES
        elif area == "docs":
            allowed_root = DOC_ROOT
            allowed = len(relative.parts) == 2 and relative.name in LINKED_DOCS
        else:
            return denied
        target = (REPO_ROOT / Path(*relative.parts)).resolve()
        if not allowed or not target.is_relative_to(allowed_root) or not target.is_file():
            return denied
        return str(target)

    def list_directory(self, path: str):
        self.send_error(403, "Directory listing is disabled")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), ReportHandler)
    print(f"Open http://{args.bind}:{args.port}/ for the mentor brief", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
