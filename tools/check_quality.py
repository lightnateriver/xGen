"""Run lint, syntax and self-contained CPU regression checks."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/test_architecture.py",
    "tests/test_pipeline_build.py",
    "tests/test_batch_inference.py",
    "tests/test_runtime_boundaries.py",
    "x_diffusers/tests/test_h3_workflow_selection.py",
    "x_diffusers/tests/test_h3_request_runtime_cpu.py",
    "x_comfyui/tests/test_nputurbo_prompt_scan.py",
)


def main():
    if importlib.util.find_spec("ruff") is None:
        print(
            "Install requirements-quality.txt in a development environment before running the quality gate.",
            file=sys.stderr,
        )
        return 2
    lint = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "x_base",
            "x_diffusers",
            "x_comfyui",
            "server",
            "example",
            "tools",
            "tests",
        ],
        cwd=ROOT,
    )
    if lint.returncode:
        return lint.returncode
    # Include new source files, but ignore generated/untracked deployment assets.
    sources = set()
    for name in (
        "x_base",
        "x_diffusers",
        "x_comfyui",
        "server",
        "example",
        "tools",
        "tests",
    ):
        sources.update((ROOT / name).rglob("*.py"))
    for path in sorted(sources):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            print(f"{path}:{exc.lineno}: {exc.msg}", file=sys.stderr)
            return 1
    print(f"Syntax: {len(sources)} Python files passed", flush=True)
    environment = dict(os.environ)
    paths = [
        str(ROOT / name) for name in ("", "tools", "x_base", "x_diffusers", "x_comfyui")
    ]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    failures = []
    for name in TESTS:
        try:
            result = subprocess.run(
                [sys.executable, name],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            failures.append(name)
            print(f"FAIL {name}: exceeded 60 seconds", flush=True)
            continue
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {name}", flush=True)
        if result.returncode:
            failures.append(name)
            print(result.stdout + result.stderr, flush=True)
        else:
            summary = next(
                (
                    line
                    for line in result.stderr.splitlines()
                    if line.startswith("Ran ")
                ),
                "",
            )
            if summary:
                print(f"  {summary}", flush=True)
    print(
        f"CPU suites: {len(TESTS) - len(failures)} passed, {len(failures)} failed",
        flush=True,
    )
    print(
        "This gate does not run model-weight, numerical device or performance acceptance tests.",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
