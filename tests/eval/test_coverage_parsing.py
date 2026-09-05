"""Regression guard for the coverage TOTAL-row parser.

``pytest --cov`` emits a variable number of numeric columns before the final
percentage depending on whether branch coverage is enabled:

    non-branch:  TOTAL   591    67   85%
    branch:      TOTAL   591    67   172    39   85%

The parser must extract the trailing percentage in both cases instead of
assuming exactly two numeric columns (Stmts, Miss).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from factory.eval.hygiene import eval_coverage
from tests.eval.test_interpreter_resolution import _make_run_result, make_python_project, make_venv


def _coverage_score(tmp_path: Path, coverage_output: str) -> dict:
    make_python_project(tmp_path)
    make_venv(tmp_path)

    with patch("factory.eval.languages.base.subprocess.run") as mock_run:
        mock_run.return_value = _make_run_result(
            stdout=f"7 passed\n{coverage_output}\n", returncode=0
        )
        return eval_coverage(tmp_path)


class TestCoverageTotalParsing:
    def test_branch_coverage_total_row_is_parsed(self, tmp_path):
        result = _coverage_score(tmp_path, "TOTAL                           591     67    172     39    85%")

        assert result["score"] == 0.85
        assert "85" in result["details"]

    def test_non_branch_coverage_total_row_is_parsed(self, tmp_path):
        result = _coverage_score(tmp_path, "TOTAL    591    67    85%")

        assert result["score"] == 0.85
        assert "85" in result["details"]
