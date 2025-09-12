#!/usr/bin/env python3
"""
Type checking baseline script that allows existing pyright errors but fails on new ones.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Set, Tuple


# Configuration for pyright
BASELINE_FILE = "scripts/type-baseline/pyright-baseline.json"
PYRIGHT_CMD = ["uv", "run", "pyright", "src", "tests", "--outputjson"]


def run_pyright() -> Dict[str, Any]:
    """Run pyright and return JSON output."""
    try:
        result = subprocess.run(
            PYRIGHT_CMD,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Pyright outputs JSON directly
        return (
            json.loads(result.stdout) if result.stdout else {"generalDiagnostics": []}
        )

    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
        print(f"Error running pyright: {e}")
        return {"generalDiagnostics": []}


def normalize_diagnostic(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize pyright diagnostic for comparison."""
    project_root = Path(__file__).parent.parent.parent

    file_path = diag.get("file", "")
    line = diag.get("range", {}).get("start", {}).get("line", 0)
    character = diag.get("range", {}).get("start", {}).get("character", 0)
    message = diag.get("message", "")
    rule = diag.get("rule", "pyright")
    severity = diag.get("severity", "error")

    # Make file path relative to project root for cross-environment compatibility
    try:
        relative_path = Path(file_path).relative_to(project_root.absolute())
        normalized_file = str(relative_path)
    except (ValueError, TypeError):
        # Fallback to original path if relative conversion fails
        normalized_file = file_path

    return {
        "file": normalized_file,
        "line": line,
        "character": character,
        "message": message,
        "rule": rule,
        "severity": severity,
    }


def load_baseline() -> Tuple[Set[str], Set[str]]:
    """Load baseline warnings and errors."""
    baseline_path = Path(__file__).parent.parent.parent / BASELINE_FILE
    if not baseline_path.exists():
        print(f"Baseline file {baseline_path} not found. Creating new baseline...")
        return set(), set()

    try:
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        baseline_warnings: Set[str] = set()
        baseline_errors: Set[str] = set()

        diagnostics = baseline_data.get("generalDiagnostics", [])

        for diag in diagnostics:
            key = f"{diag['file']}:{diag['line']}:{diag['character']}:{diag['rule']}:{diag['message']}"

            if diag.get("severity") == "warning":
                baseline_warnings.add(key)
            elif diag.get("severity") == "error":
                baseline_errors.add(key)

        return baseline_warnings, baseline_errors
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading baseline: {e}")
        return set(), set()


def save_baseline(current_data: Dict[str, Any]) -> None:
    """Save current pyright diagnostics as new baseline."""

    diagnostics = current_data.get("generalDiagnostics", [])

    normalized_diagnostics: list[Dict[str, Any]] = []
    for diag in diagnostics:
        normalized_diag = normalize_diagnostic(diag)
        normalized_diagnostics.append(normalized_diag)

    # Save in pyright format
    baseline_data = {"generalDiagnostics": normalized_diagnostics}

    baseline_path = Path(__file__).parent.parent.parent / BASELINE_FILE
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"Baseline saved to {baseline_path}")


def check_pyright(create_baseline: bool = False) -> bool:
    """Check pyright type errors."""
    print("🔍 Checking pyright...")

    current_data = run_pyright()

    if create_baseline:
        save_baseline(current_data)
        return True

    # Load baseline
    baseline_warnings, baseline_errors = load_baseline()

    # Get current warnings and errors
    current_warnings: Set[str] = set()
    current_errors: Set[str] = set()
    new_warnings: list[Dict[str, Any]] = []
    new_errors: list[Dict[str, Any]] = []

    diagnostics = current_data.get("generalDiagnostics", [])

    for diag in diagnostics:
        normalized = normalize_diagnostic(diag)
        key = f"{normalized['file']}:{normalized['line']}:{normalized['character']}:{normalized['rule']}:{normalized['message']}"

        if normalized.get("severity") == "error":
            current_errors.add(key)
            if key not in baseline_errors:
                new_errors.append(normalized)
        elif normalized.get("severity") == "warning":
            current_warnings.add(key)
            if key not in baseline_warnings:
                new_warnings.append(normalized)

    # Calculate fixed issues
    fixed_errors = len(baseline_errors) - len(current_errors)
    fixed_warnings = len(baseline_warnings) - len(current_warnings)

    # Report results
    print(f"  Errors: {len(current_errors)} (baseline: {len(baseline_errors)})")
    print(f"  Warnings: {len(current_warnings)} (baseline: {len(baseline_warnings)})")
    print(f"  New errors: {len(new_errors)}")
    print(f"  New warnings: {len(new_warnings)}")
    if fixed_errors > 0:
        print(f"  Fixed errors: {fixed_errors}")
    if fixed_warnings > 0:
        print(f"  Fixed warnings: {fixed_warnings}")

    if new_errors:
        print(f"  ❌ FAILED: {len(new_errors)} new errors found:")
        for error in new_errors:
            print(
                f"    {error['file']}:{error['line']} - {error['message']} ({error['rule']})"
            )
        return False

    if new_warnings:
        print(f"  ❌ FAILED: {len(new_warnings)} new warnings found:")
        for warning in new_warnings:
            print(
                f"    {warning['file']}:{warning['line']} - {warning['message']} ({warning['rule']})"
            )
        return False

    # Check if baseline needs updating due to fixed issues
    if fixed_errors > 0 or fixed_warnings > 0:
        print(f"  🎉 GREAT: {fixed_errors + fixed_warnings} type issues fixed!")
        print("  ❌ FAILED: Baseline needs updating")
        print(
            "  ➡️  Run: uv run python scripts/type-baseline/check-type-baseline.py --create-baseline"
        )
        print("  ➡️  Then commit the updated baseline file")
        return False

    print("  ✅ PASSED: No new type issues found")
    return True


def main():
    create_baseline = "--create-baseline" in sys.argv

    # Check pyright
    passed = check_pyright(create_baseline)
    return 0 if (create_baseline or passed) else 1


if __name__ == "__main__":
    sys.exit(main())
