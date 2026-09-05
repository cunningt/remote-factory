"""Tests for project-interpreter resolution in the Python evaluator.

Regression guard for the defect where hygiene evals shelled out with
``sys.executable`` (the FACTORY's venv) while ``cwd`` was the target project.
pytest died during collection, nothing matched ``N passed``, and the tests /
coverage dimensions silently reported "no test suite detected" (0.5) for
projects that had a large, green suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from factory.eval.hygiene import eval_coverage, eval_lint, eval_tests, eval_type_check
from factory.eval.languages.interpreter import resolve_python_interpreter


def _make_run_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    class _Result:
        def __init__(self, rc, out, err):
            self.returncode = rc
            self.stdout = out
            self.stderr = err
    return _Result(returncode, stdout, stderr)


def make_venv(project_path: Path, dir_name: str = ".venv") -> Path:
    """Materialise a minimal, executable project-local venv interpreter."""
    if os.name == "nt":  # pragma: no cover - CI runs POSIX
        python = project_path / dir_name / "Scripts" / "python.exe"
    else:
        python = project_path / dir_name / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    return python


def make_python_project(project_path: Path, *, with_tests: bool = True) -> Path:
    (project_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    pkg = project_path / "demo"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    if with_tests:
        tests = project_path / "tests"
        tests.mkdir(exist_ok=True)
        (tests / "test_demo.py").write_text("def test_ok(): pass\n")
    return project_path


# ── Case 1: project-local .venv is found and used ─────────────────


class TestProjectLocalVenv:
    def test_resolution_prefers_project_venv(self, tmp_path):
        make_python_project(tmp_path)
        venv_python = make_venv(tmp_path)

        resolution = resolve_python_interpreter(tmp_path)

        assert resolution.resolved
        assert resolution.source == "venv"
        assert resolution.argv == (str(venv_python),)
        assert ".venv" in resolution.reason

    def test_plain_venv_dir_also_accepted(self, tmp_path):
        make_python_project(tmp_path)
        venv_python = make_venv(tmp_path, dir_name="venv")

        resolution = resolve_python_interpreter(tmp_path)

        assert resolution.source == "venv"
        assert resolution.argv == (str(venv_python),)

    def test_dot_venv_wins_over_venv(self, tmp_path):
        make_python_project(tmp_path)
        make_venv(tmp_path, dir_name="venv")
        preferred = make_venv(tmp_path, dir_name=".venv")

        assert resolve_python_interpreter(tmp_path).argv == (str(preferred),)

    def test_venv_preferred_over_uv_even_when_uv_available(self, tmp_path):
        make_python_project(tmp_path)
        (tmp_path / "uv.lock").write_text("")
        venv_python = make_venv(tmp_path)

        with patch("factory.eval.languages.interpreter.shutil.which", return_value="/bin/uv"):
            resolution = resolve_python_interpreter(tmp_path)

        assert resolution.source == "venv"
        assert resolution.argv == (str(venv_python),)

    def test_pytest_is_invoked_with_the_project_venv(self, tmp_path):
        make_python_project(tmp_path)
        venv_python = make_venv(tmp_path)

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = _make_run_result(
                stdout="7 passed\nTOTAL    100    10    90%\n", returncode=0
            )
            result = eval_tests(tmp_path)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == str(venv_python)
        assert cmd[0] != sys.executable
        assert cmd[1:3] == ["-m", "pytest"]
        assert result["score"] == 1.0
        assert "7 passed" in result["details"]

    def test_lint_and_type_check_use_the_project_venv(self, tmp_path):
        make_python_project(tmp_path)
        venv_python = make_venv(tmp_path)

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = _make_run_result(returncode=0)
            eval_lint(tmp_path)
            assert mock_run.call_args[0][0][:4] == [str(venv_python), "-m", "ruff", "check"]
            eval_type_check(tmp_path)
            assert mock_run.call_args[0][0][:3] == [str(venv_python), "-m", "mypy"]


# ── Case 2: uv-managed project with no .venv resolves via `uv run` ─


class TestUvManagedProject:
    def test_uv_lock_resolves_via_uv_run(self, tmp_path):
        make_python_project(tmp_path)
        (tmp_path / "uv.lock").write_text("")

        with patch("factory.eval.languages.interpreter.shutil.which", return_value="/bin/uv"):
            resolution = resolve_python_interpreter(tmp_path)

        assert resolution.resolved
        assert resolution.source == "uv"
        assert resolution.argv == ("/bin/uv", "run", "--project", str(tmp_path), "python")

    def test_pyproject_alone_is_a_uv_marker(self, tmp_path):
        make_python_project(tmp_path)  # writes a [project] pyproject.toml

        with patch("factory.eval.languages.interpreter.shutil.which", return_value="/bin/uv"):
            resolution = resolve_python_interpreter(tmp_path)

        assert resolution.source == "uv"

    def test_pytest_runs_through_uv(self, tmp_path):
        make_python_project(tmp_path)
        (tmp_path / "uv.lock").write_text("")

        with patch("factory.eval.languages.interpreter.shutil.which", return_value="/bin/uv"), \
                patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = _make_run_result(
                stdout="12 passed\nTOTAL   200   20   90%\n", returncode=0
            )
            tests_result = eval_tests(tmp_path)
            cov_result = eval_coverage(tmp_path)

        cmd = mock_run.call_args[0][0]
        assert cmd[:5] == ["/bin/uv", "run", "--project", str(tmp_path), "python"]
        assert cmd[5:7] == ["-m", "pytest"]
        assert tests_result["score"] == 1.0
        assert cov_result["score"] == 0.9


# ── Case 3: unresolvable interpreter -> NEUTRAL with a named reason ─


class TestUnresolvableInterpreter:
    def test_resolution_reports_uv_missing_from_path(self, tmp_path):
        make_python_project(tmp_path)
        (tmp_path / "uv.lock").write_text("")

        with patch("factory.eval.languages.interpreter.shutil.which", return_value=None):
            resolution = resolve_python_interpreter(tmp_path)

        assert not resolution.resolved
        assert resolution.argv is None
        assert "interpreter not resolved" in resolution.reason
        assert "uv not on PATH" in resolution.reason

    def test_resolution_reports_no_markers(self, tmp_path):
        (tmp_path / "setup.py").write_text("")
        (tmp_path / "tests").mkdir()

        with patch("factory.eval.languages.interpreter.shutil.which", return_value=None):
            resolution = resolve_python_interpreter(tmp_path)

        assert not resolution.resolved
        assert "no uv project markers" in resolution.reason

    def test_tests_dimension_is_neutral_not_zero(self, tmp_path):
        make_python_project(tmp_path)

        with patch("factory.eval.languages.interpreter.shutil.which", return_value=None):
            result = eval_tests(tmp_path)

        assert result["score"] == 0.5, "unmeasurable must abstain, never score 0.0"
        assert result["passed"] is True
        assert "interpreter not resolved" in result["details"]
        assert "uv not on PATH" in result["details"]

    def test_coverage_dimension_is_neutral_not_zero(self, tmp_path):
        make_python_project(tmp_path)

        with patch("factory.eval.languages.interpreter.shutil.which", return_value=None):
            result = eval_coverage(tmp_path)

        assert result["score"] == 0.5
        assert "interpreter not resolved" in result["details"]

    @pytest.mark.parametrize("evaluate", [eval_lint, eval_type_check])
    def test_lint_and_type_check_are_neutral_not_zero(self, tmp_path, evaluate):
        make_python_project(tmp_path)

        with patch("factory.eval.languages.interpreter.shutil.which", return_value=None):
            result = evaluate(tmp_path)

        assert result["score"] == 0.5
        assert "interpreter not resolved" in result["details"]

    def test_no_subprocess_is_spawned_when_unresolvable(self, tmp_path):
        make_python_project(tmp_path)

        with patch("factory.eval.languages.interpreter.shutil.which", return_value=None), \
                patch("factory.eval.languages.base.subprocess.run") as mock_run:
            eval_tests(tmp_path)

        mock_run.assert_not_called()

    def test_missing_tool_in_project_interpreter_is_neutral_not_zero(self, tmp_path):
        """ruff absent from the project's venv is an abstention, not 0.0."""
        make_python_project(tmp_path)
        make_venv(tmp_path)

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = _make_run_result(
                stderr="No module named ruff\n", returncode=1
            )
            result = eval_lint(tmp_path)

        assert result["score"] == 0.5
        assert "ruff not installed" in result["details"]

    def test_suite_present_but_uncollectable_is_neutral_not_zero(self, tmp_path):
        make_python_project(tmp_path)
        make_venv(tmp_path)

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = _make_run_result(
                stdout="!!!! Interrupted: 3 errors during collection !!!!\n", returncode=2
            )
            result = eval_tests(tmp_path)

        assert result["score"] == 0.5
        assert "test suite present but pytest exited 2" in result["details"]


# ── Case 4: project genuinely has NO test suite -> NEUTRAL, not 0.0 ─
#
# This is the regression guard for the constraint that "no suite exists" and
# "I could not run the suite that exists" are DIFFERENT FACTS. Collapsing
# either into 0.0 converts an honest abstention into a false accusation.


class TestGenuinelyNoTestSuite:
    def test_no_suite_with_resolvable_interpreter_is_neutral(self, tmp_path):
        make_python_project(tmp_path, with_tests=False)
        make_venv(tmp_path)

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            # pytest exit code 5 == "no tests were collected"
            mock_run.return_value = _make_run_result(stdout="no tests ran\n", returncode=5)
            result = eval_tests(tmp_path)

        assert result["score"] == 0.5
        assert result["passed"] is True

    def test_no_suite_and_no_interpreter_is_neutral(self, tmp_path):
        make_python_project(tmp_path, with_tests=False)

        with patch("factory.eval.languages.interpreter.shutil.which", return_value=None):
            result = eval_tests(tmp_path)

        assert result["score"] == 0.5
        assert result["passed"] is True

    def test_empty_project_is_neutral(self, tmp_path):
        result = eval_tests(tmp_path)
        assert result["score"] == 0.5
        assert result["passed"] is True

    def test_no_suite_and_could_not_run_are_distinguishable(self, tmp_path):
        """The two facts must not share a details string."""
        no_suite = tmp_path / "no_suite"
        no_suite.mkdir()
        make_python_project(no_suite, with_tests=False)
        make_venv(no_suite)

        unmeasurable = tmp_path / "unmeasurable"
        unmeasurable.mkdir()
        make_python_project(unmeasurable, with_tests=True)

        with patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = _make_run_result(stdout="no tests ran\n", returncode=5)
            no_suite_result = eval_tests(no_suite)
        with patch("factory.eval.languages.interpreter.shutil.which", return_value=None):
            unmeasurable_result = eval_tests(unmeasurable)

        assert no_suite_result["score"] == unmeasurable_result["score"] == 0.5
        assert no_suite_result["details"] != unmeasurable_result["details"]
        assert "no test suite detected" in no_suite_result["details"]
        assert "no test suite detected" not in unmeasurable_result["details"]
        assert "interpreter not resolved" in unmeasurable_result["details"]


# ── Aggregation: abstentions never drag down a real measurement ────


class TestAbstentionAggregation:
    def test_abstaining_sub_project_does_not_lower_the_score(self, tmp_path):
        """A monorepo where one sub-project is unmeasurable keeps the other's score."""
        measured = tmp_path / "measured"
        measured.mkdir()
        make_python_project(measured)
        make_venv(measured)

        abstaining = tmp_path / "abstaining"
        abstaining.mkdir()
        make_python_project(abstaining)

        real_which = __import__("shutil").which

        def _which(name, *args, **kwargs):
            return None if name == "uv" else real_which(name, *args, **kwargs)

        with patch("factory.eval.languages.interpreter.shutil.which", side_effect=_which), \
                patch("factory.eval.languages.base.subprocess.run") as mock_run:
            mock_run.return_value = _make_run_result(stdout="4 passed\n", returncode=0)
            result = eval_tests(tmp_path)

        assert result["score"] == 1.0
        assert "4 passed" in result["details"]
        assert "Not measured:" in result["details"]
        assert "interpreter not resolved" in result["details"]
