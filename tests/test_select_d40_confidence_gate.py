import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_d40_confidence_gate import build_report
from test_select_d38_calibrated_scale import make_report, metrics


def add_gates(report, values):
    report["hard_confidence_gated_sweeps"] = {"thresholds": values}
    return report


def threshold(coverage, scales):
    return {"active_fraction": coverage, "scales": scales}


def test_selects_confidence_gate_then_confirms_same_candidate():
    calibration = add_gates(
        make_report([5, 20], {}),
        {
            "0.08": threshold(0.7, {"0.25": metrics(0.02, 0.021)}),
            "0.12": threshold(0.3, {"0.5": metrics(0.03, 0.025)}),
        },
    )
    confirmation = add_gates(
        make_report([25, 39, 49], {}),
        {
            "0.08": threshold(0.6, {"0.25": metrics(-1.0, -1.0)}),
            "0.12": threshold(0.2, {"0.5": metrics(0.011, 0.012)}),
        },
    )
    report = build_report(calibration, confirmation)
    assert report["selected_confidence_threshold"] == 0.12
    assert report["selected_scale"] == 0.5
    assert report["independent_confidence_gate_passed"] is True
    assert report["dit_authorized"] is False
