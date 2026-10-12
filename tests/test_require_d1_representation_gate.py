import json
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/physical_tokens/require_d1_representation_gate.py"
)


def test_gate_guard_accepts_pass_and_rejects_fail(tmp_path: Path) -> None:
    report = tmp_path / "comparison.json"
    report.write_text(json.dumps({"gate": {"passed": True, "checks": {"a": True}}}))
    subprocess.run([sys.executable, str(SCRIPT), str(report)], check=True)

    report.write_text(json.dumps({"gate": {"passed": False, "checks": {"a": False}}}))
    failed = subprocess.run(
        [sys.executable, str(SCRIPT), str(report)],
        text=True,
        capture_output=True,
    )
    assert failed.returncode != 0
    assert "failed checks" in failed.stderr
