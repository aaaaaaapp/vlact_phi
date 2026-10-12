import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_d39_soft_conditioning import build_report
from test_select_d38_calibrated_scale import make_report, metrics


def add_soft(report, values):
    report["soft_conditioned_sweeps"] = {"temperatures": values}
    return report


def test_selects_soft_temperature_and_scale_then_confirms_it():
    calibration = add_soft(
        make_report([5, 20], {}),
        {
            "0.5": {"scales": {"0.125": metrics(0.02, 0.021)}},
            "1": {"scales": {"0.25": metrics(0.03, 0.025)}},
            "2": {"scales": {"0.5": metrics(0.01, 0.04)}},
        },
    )
    confirmation = add_soft(
        make_report([25, 39, 49], {}),
        {
            "0.5": {"scales": {"0.125": metrics(-1.0, -1.0)}},
            "1": {"scales": {"0.25": metrics(0.011, 0.012)}},
            "2": {"scales": {"0.5": metrics(1.0, 1.0)}},
        },
    )
    report = build_report(calibration, confirmation)
    assert report["selected_temperature"] == 1.0
    assert report["selected_scale"] == 0.25
    assert report["independent_soft_gate_passed"] is True
    assert report["dit_authorized"] is False
