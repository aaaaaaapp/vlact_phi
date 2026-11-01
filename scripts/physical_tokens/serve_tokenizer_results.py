#!/usr/bin/env python3
"""Serve the tokenizer results page and only its explicitly linked files."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
REPORT = "docs/ROBOTWIN_TOKENIZER_RESULTS_SNAPSHOT.html"
ALLOWED: dict[str, tuple[Path, str]] = {
    "/" + REPORT: (ROOT / REPORT, "text/html; charset=utf-8"),
    "/docs/ROBOTWIN_AR_GENERATION_TRAINING.html": (
        ROOT / "docs/ROBOTWIN_AR_GENERATION_TRAINING.html",
        "text/html; charset=utf-8",
    ),
    "/docs/ROBOTWIN_DUAL_AR_POSTTRAIN_STATUS.html": (
        ROOT / "docs/ROBOTWIN_DUAL_AR_POSTTRAIN_STATUS.html",
        "text/html; charset=utf-8",
    ),
    "/docs/ROBOTWIN_DUAL_AR_POSTTRAIN.md": (
        ROOT / "docs/ROBOTWIN_DUAL_AR_POSTTRAIN.md",
        "text/plain; charset=utf-8",
    ),
    "/docs/ROBOTWIN_ACTION_TOKEN_LENGTH_8_VS_32.md": (
        ROOT / "docs/ROBOTWIN_ACTION_TOKEN_LENGTH_8_VS_32.md",
        "text/plain; charset=utf-8",
    ),
    "/results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps500/heldout_diagnostic.json": (
        ROOT / "results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps500/heldout_diagnostic.json",
        "application/json; charset=utf-8",
    ),
    "/docs/ROBOTWIN_EMA_FSQ_IFSQ_TOKENIZER_STUDY.md": (
        ROOT / "docs/ROBOTWIN_EMA_FSQ_IFSQ_TOKENIZER_STUDY.md",
        "text/plain; charset=utf-8",
    ),
}
for relative in (
    "artifacts/robotwin_full50_study/action_tokenizers/ema_8x525_decay095_dead100_bs128_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/action_tokenizers/ema_32x525_decay095_dead100_bs128_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/action_tokenizers/ema_8x525_decay095_dead100_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/action_tokenizers/ema_16x525_decay095_dead100_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/action_tokenizers/fsq_8x525_levels3557_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/action_tokenizers/ifsq_8x525_levels3557_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_k256_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_k256_steps20000/heldout_step20000_schema3.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_k1024_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_k2048_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_k16384_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_k32768_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_t64_k1024_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_t64_k2048_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_t128_k1024_steps20000/metrics.jsonl",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_t128_k2048_steps20000/metrics.jsonl",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_t128_k1024_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_t128_k2048_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_global_k1024_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ema_global_k2048_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/fsq_levels4444_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/fsq_levels444442_k2048_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/fsq_official_levels84444_k2048_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/fsq_symmetric_levels84444_k2048_steps20000/heldout_step20000.json",
    "artifacts/robotwin_full50_study/physics_tokenizers/ifsq_levels3517_steps20000/heldout_step20000.json",
    "artifacts/physical_language_tokenizers/quantizer_study_ema_256x256_steps8000/heldout_evaluation.json",
    "artifacts/physical_language_tokenizers/quantizer_study_fsq_256x256_steps8000/heldout_evaluation.json",
    "artifacts/physical_language_tokenizers/quantizer_study_fsq_256x256_levels4444_steps8000/heldout_evaluation.json",
):
    content_type = (
        "text/plain; charset=utf-8"
        if relative.endswith(".jsonl")
        else "application/json; charset=utf-8"
    )
    ALLOWED["/" + relative] = (ROOT / relative, content_type)


class ReportHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._serve(send_body=True)

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def _serve(self, *, send_body: bool) -> None:
        request_path = urlsplit(self.path).path
        if request_path == "/":
            self.send_response(302)
            self.send_header("Location", "/" + REPORT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        resource = ALLOWED.get(request_path)
        if resource is None:
            self.send_error(404, "Not found")
            return
        path, content_type = resource
        if not path.is_file():
            self.send_error(404, "Report source is not available")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        if send_body:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ReportHandler)
    print(f"Serving tokenizer results on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
