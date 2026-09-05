"""Python language evaluator."""

from __future__ import annotations

import re
from pathlib import Path

from factory.eval.languages.base import EvalFragment, _run_cmd
from factory.eval.languages.interpreter import resolve_python_interpreter

# pytest exit code 5 means "no tests were collected" — the run itself succeeded,
# the project simply has nothing to run.
_PYTEST_NO_TESTS_COLLECTED = 5

_MISSING_MODULE_RE = re.compile(r"No module named '?([A-Za-z0-9_.]+)'?")


def _missing_module(output: str) -> str | None:
    """Return the name of a module the project interpreter is missing, if any."""
    match = _MISSING_MODULE_RE.search(output)
    return match.group(1) if match else None


def _has_test_suite(project_path: Path) -> bool:
    """Best-effort check for whether a test suite exists at all.

    Deliberately shallow (project root + immediate subdirectories) so it stays
    cheap on large trees. A false negative degrades to the pre-existing
    behaviour — "no test suite detected" — so the failure mode is safe.
    """
    def _looks_like_tests(directory: Path) -> bool:
        if any((directory / name).is_dir() for name in ("tests", "test")):
            return True
        return any(directory.glob("test_*.py")) or any(directory.glob("*_test.py"))

    if _looks_like_tests(project_path):
        return True
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__"}
    for child in project_path.iterdir():
        if child.is_dir() and child.name not in skip and not child.name.startswith("."):
            if _looks_like_tests(child):
                return True
    return False


def _abstain(project_path: Path, reason: str) -> EvalFragment:
    """Build an abstention fragment: neutral 0.5, carrying the reason."""
    return EvalFragment(
        passed=0,
        failed=0,
        score=0.5,
        details=f"{project_path.name}: {reason}",
        neutral=True,
    )


class PythonEvaluator:
    @property
    def name(self) -> str:
        return "python"

    def _detect_cov_target(self, project_path: Path) -> str:
        src_dirs = [
            c.name for c in sorted(project_path.iterdir())
            if c.is_dir() and (c / "__init__.py").exists()
        ]
        return src_dirs[0] if src_dirs else "."

    def detect(self, project_path: Path) -> bool:
        return (
            (project_path / "pyproject.toml").exists()
            or (project_path / "setup.py").exists()
        )

    def run_tests_with_coverage(
        self, project_path: Path, timeout: int = 300,
    ) -> tuple[EvalFragment | None, EvalFragment | None]:
        interp = resolve_python_interpreter(project_path)
        if not interp.resolved:
            if not _has_test_suite(project_path):
                # No suite AND no interpreter — report the simpler, truer fact.
                return None, None
            frag = _abstain(project_path, interp.reason)
            return frag, _abstain(project_path, interp.reason)

        cov_target = self._detect_cov_target(project_path)
        rc, stdout, stderr = _run_cmd(
            interp.command(
                "-m", "pytest",
                f"--cov={cov_target}", "--cov-report=term",
                "-v", "--tb=no", "-q",
            ),
            project_path,
            timeout=timeout,
        )
        output = stdout + stderr

        # Parse test results
        test_frag: EvalFragment | None = None
        p_match = re.search(r"(\d+)\s+passed", output)
        f_match = re.search(r"(\d+)\s+failed", output)
        p = int(p_match.group(1)) if p_match else 0
        f = int(f_match.group(1)) if f_match else 0
        if p + f > 0:
            total = p + f
            test_frag = EvalFragment(
                passed=p,
                failed=f,
                score=p / total,
                details=f"{project_path.name}: {p} passed, {f} failed",
            )

        if test_frag is None:
            # Nothing parsed. Distinguish "there is no suite" (honest 'not
            # detected') from "there is a suite and I could not run it"
            # (abstention naming the reason). Collapsing either into 0.0 would
            # be a false accusation.
            missing = _missing_module(output)
            if missing:
                reason = f"{missing} not installed in the project interpreter ({interp.reason})"
                return _abstain(project_path, reason), _abstain(project_path, reason)
            if rc not in (0, _PYTEST_NO_TESTS_COLLECTED) and _has_test_suite(project_path):
                reason = (
                    f"test suite present but pytest exited {rc} without reporting results "
                    f"(via {interp.reason})"
                )
                return _abstain(project_path, reason), _abstain(project_path, reason)
            return None, None

        # Parse coverage only if tests were collected
        cov_frag: EvalFragment | None = None
        total_match = re.search(r"TOTAL\s+(?:\d+\s+)+(\d+)%", output)
        if total_match:
            pct = int(total_match.group(1))
            cov_frag = EvalFragment(
                passed=0,
                failed=0,
                score=pct / 100.0,
                coverage_pct=pct,
                details=f"{project_path.name}: {pct}%",
            )

        return test_frag, cov_frag

    def run_tests(self, project_path: Path, timeout: int = 300) -> EvalFragment | None:
        """Prefer run_tests_with_coverage() to avoid a redundant pytest invocation."""
        return self.run_tests_with_coverage(project_path, timeout=timeout)[0]

    def run_lint(self, project_path: Path) -> EvalFragment | None:
        interp = resolve_python_interpreter(project_path)
        if not interp.resolved:
            return _abstain(project_path, interp.reason)
        rc, stdout, stderr = _run_cmd(interp.command("-m", "ruff", "check", "."), project_path)
        output = stdout + stderr
        if rc == 0:
            return EvalFragment(passed=1, failed=0, score=1.0, details=f"{project_path.name}: clean")
        missing = _missing_module(output)
        if missing:
            return _abstain(
                project_path,
                f"{missing} not installed in the project interpreter ({interp.reason})",
            )
        err_match = re.search(r"Found\s+(\d+)\s+error", output)
        count = int(err_match.group(1)) if err_match else 1
        return EvalFragment(passed=0, failed=count, score=0.0, details=f"{project_path.name}: {count} errors")

    def run_type_check(self, project_path: Path) -> EvalFragment | None:
        interp = resolve_python_interpreter(project_path)
        if not interp.resolved:
            return _abstain(project_path, interp.reason)
        src_dirs = []
        for child in sorted(project_path.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                src_dirs.append(child.name)
        target = src_dirs[0] if src_dirs else "."
        rc, stdout, stderr = _run_cmd(interp.command("-m", "mypy", target), project_path)
        output = stdout + stderr
        if rc == 0:
            return EvalFragment(passed=1, failed=0, score=1.0, details=f"{project_path.name}: clean")
        missing = _missing_module(output)
        if missing:
            return _abstain(
                project_path,
                f"{missing} not installed in the project interpreter ({interp.reason})",
            )
        err_match = re.search(r"Found\s+(\d+)\s+error", output)
        count = int(err_match.group(1)) if err_match else 1
        return EvalFragment(
            passed=0, failed=count, score=0.0,
            details=f"{project_path.name}: {count} errors",
        )

    def run_coverage(self, project_path: Path, timeout: int = 300) -> EvalFragment | None:
        """Prefer run_tests_with_coverage() to avoid a redundant pytest invocation."""
        return self.run_tests_with_coverage(project_path, timeout=timeout)[1]


def register_evaluator() -> PythonEvaluator:
    return PythonEvaluator()
