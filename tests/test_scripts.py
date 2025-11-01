"""
Tests for command-line scripts in the scripts/ directory.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def test_cli_help_runs():
    """scripts/bayesian_analysis.py should print help and exit 0 with --help."""
    cmd = [sys.executable, str(SCRIPTS / "bayesian_analysis.py"), "--help"]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0
    assert "Bayesian analysis of parameter sweep results" in proc.stdout
