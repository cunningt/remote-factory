"""Resolve the Python interpreter that owns a *target* project's dependencies.

The factory evaluates other projects. Anything the Python evaluator shells out
to (``pytest``, ``ruff``, ``mypy``) has to run under the interpreter that has
the *target project's* dependencies installed. ``sys.executable`` is the
factory's own virtualenv: it knows nothing about the target's imports, so
pytest dies during collection, no ``N passed`` is ever printed, and the hygiene
layer cannot tell that apart from "this project has no tests".

Resolution order, and the justification for it:

1. **Project-local virtualenv** — ``<project>/.venv`` then ``<project>/venv``,
   using ``bin/python`` on POSIX and ``Scripts/python.exe`` on Windows.
   First because it is the cheapest correct answer: a materialised venv is a
   direct exec with no resolver, no lockfile parsing and no network. It is also
   the layout both ``uv`` and ``python -m venv`` produce, so it already covers
   uv-managed projects that have been synced at least once — which is why it is
   preferred over #2 even when uv markers are present.

2. **uv-managed project with no materialised venv** — ``uv run --project
   <project> python -m ...``. Requires a uv marker (``uv.lock``, or a
   ``pyproject.toml`` declaring ``[project]`` / ``[tool.uv]``) *and* the ``uv``
   binary on PATH. Second because uv re-resolves the environment on every
   invocation, which is slower than #1, and because it may materialise
   ``<project>/.venv`` as a side effect of measuring.

3. **Unresolvable** — return an unresolved marker carrying a human-readable
   reason. Callers MUST surface that as a NEUTRAL (0.5) score naming the
   reason. Falling back to ``sys.executable`` here would reintroduce the
   original bug, and scoring 0.0 would turn "I could not measure this" into
   "this project is broken" — a false accusation, and a worse defect than the
   one this module exists to fix.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger()

# Candidate venv directory names, in preference order.
VENV_DIR_NAMES = (".venv", "venv")

# Interpreter path inside a venv, POSIX layout first then Windows.
_VENV_PYTHON_RELPATHS = ("bin/python", "bin/python3", "Scripts/python.exe")


@dataclass(frozen=True)
class InterpreterResolution:
    """Where a project's Python interpreter lives, or why it could not be found.

    ``argv`` is a command *prefix*: append ``-m <tool> ...`` to build a full
    command. It is ``None`` when the interpreter is unresolvable, in which case
    ``reason`` explains what was looked for and what was missing.
    """

    argv: tuple[str, ...] | None
    source: str  # "venv" | "uv" | "unresolved"
    reason: str

    @property
    def resolved(self) -> bool:
        return self.argv is not None

    def command(self, *args: str) -> list[str]:
        """Build a full argv by appending ``args`` to the interpreter prefix."""
        if self.argv is None:
            raise ValueError(f"interpreter is unresolved: {self.reason}")
        return [*self.argv, *args]


def _venv_python(project_path: Path) -> Path | None:
    """Return the interpreter inside a project-local venv, if one exists."""
    for dir_name in VENV_DIR_NAMES:
        for rel in _VENV_PYTHON_RELPATHS:
            candidate = project_path / dir_name / rel
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def _has_uv_markers(project_path: Path) -> bool:
    """True if the project looks uv-managed (uv.lock, or a [project]/[tool.uv] pyproject)."""
    if (project_path / "uv.lock").is_file():
        return True
    pyproject = project_path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "[project]" in text or "[tool.uv]" in text


def resolve_python_interpreter(project_path: Path) -> InterpreterResolution:
    """Resolve the interpreter that owns ``project_path``'s dependencies.

    Never falls back to ``sys.executable`` — see the module docstring.
    """
    venv_python = _venv_python(project_path)
    if venv_python is not None:
        rel = venv_python.relative_to(project_path)
        return InterpreterResolution(
            argv=(str(venv_python),),
            source="venv",
            reason=f"project venv: {rel}",
        )

    if _has_uv_markers(project_path):
        uv_bin = shutil.which("uv")
        if uv_bin:
            return InterpreterResolution(
                argv=(uv_bin, "run", "--project", str(project_path), "python"),
                source="uv",
                reason="uv run (uv-managed project, no materialised venv)",
            )
        return InterpreterResolution(
            argv=None,
            source="unresolved",
            reason="interpreter not resolved: no .venv and uv not on PATH",
        )

    return InterpreterResolution(
        argv=None,
        source="unresolved",
        reason="interpreter not resolved: no .venv and no uv project markers",
    )
