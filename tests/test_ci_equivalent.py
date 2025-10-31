"""
CI-equivalent tests converted from .github/workflows/test-and-sweep.yml.

This test suite replaces GitHub Actions by running the same checks locally:
- Environment sanity and imports
- Small parameter sweep using scripts/run_sweep.py and a minimal JSON config
- Basic assertions on outputs and scaling

To run only this suite:
    pytest -q tests/test_ci_equivalent.py

To enable slower end-to-end sweep, set EFQS_RUN_SLOW=1.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def test_imports_sanity():
    """Quick import sanity for core modules."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import efqs  # noqa: F401
    from efqs import gravitational_coupling, em_sources  # noqa: F401


@pytest.mark.slow
@pytest.mark.parametrize("grid_points,steps", [(15, 50)])
def test_small_parameter_sweep(tmp_path: Path, grid_points: int, steps: int):
    """
    Run a small parameter sweep equivalent to the old CI job and assert outputs.

    This mirrors .github/workflows/test-and-sweep.yml's nightly-sweep job.
    """
    if os.environ.get("EFQS_RUN_SLOW") not in {"1", "true", "True"}:
        pytest.skip("Set EFQS_RUN_SLOW=1 to run end-to-end sweep test")

    sweep_ci = {
        "base_config": {
            "enable_gravity": True,
            "observer_distance_m": 10.0,
            "box_size_m": 0.05,
            "grid_points": grid_points,
            "dt_s": 1e-15,
            "steps": steps,
            "use_tt_projection": True,
            "include_pair_losses": False,
            "source": {
                "type": "interfering_pulses",
                "E0_Vpm": 1e15,
                "wavelength_m": 800e-9,
                "frequency_Hz": 3.75e14
            }
        },
        "sweep_parameters": {
            "source.E0_Vpm": [5e14, 1e15],
            "include_pair_losses": [False, True]
        }
    }

    sweep_path = tmp_path / "sweep_ci.json"
    out_path = tmp_path / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_ci, f)

    # Run the sweep script
    cmd = [sys.executable, str(SCRIPTS / "run_sweep.py"), "--sweep-config", str(sweep_path), "--output", str(out_path)]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    assert proc.returncode == 0, f"Sweep failed with code {proc.returncode}:\n{proc.stdout}"

    # Validate output
    assert out_path.exists(), "Sweep output JSON not found"
    with open(out_path, "r") as f:
        data = json.load(f)

    # Expect 2x2 = 4 configurations
    assert isinstance(data, list) and len(data) == 4

    # Check required fields and reasonable ranges
    required_keys = {"h_rms", "h_max", "P_avg", "P_max", "source.E0_Vpm", "include_pair_losses"}
    for entry in data:
        assert required_keys.issubset(entry.keys())
        assert np.isfinite(entry["h_rms"]) and entry["h_rms"] >= 0
        assert np.isfinite(entry["P_avg"]) and entry["P_avg"] >= 0

    # Simple scaling sanity: h_rms should not decrease when E0 increases (for same pair-loss setting)
    group_false = sorted([e for e in data if not e["include_pair_losses"]], key=lambda e: e["source.E0_Vpm"])  # pair losses off
    if len(group_false) == 2:
        assert group_false[1]["h_rms"] >= group_false[0]["h_rms"]


def test_cli_help_runs():
    """scripts/bayesian_analysis.py should print help and exit 0 with --help."""
    cmd = [sys.executable, str(SCRIPTS / "bayesian_analysis.py"), "--help"]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0
    assert "Bayesian analysis of parameter sweep results" in proc.stdout
