#!/usr/bin/env python3
"""Verify that a T8 result directory is byte-identical to the frozen baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "baselines" / "t8_20260913_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []

    def check_hash(relative: str, expected: str) -> None:
        path = root / relative
        if not path.exists():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"sha256 mismatch: {relative}")

    for relative, expected in manifest.get("code_sha256", {}).items():
        check_hash(relative, expected)
    for relative, expected in manifest.get("static_model_data", {}).items():
        check_hash(relative, expected)
    check_hash(manifest["input_data"]["path"], manifest["input_data"]["sha256"])
    result_dir = root / manifest["result_dir"]
    for filename, expected in manifest.get("artifact_sha256", {}).items():
        check_hash(str(Path(manifest["result_dir"]) / filename), expected)

    def read_result(filename):
        return json.loads((result_dir / filename).read_text(encoding="utf-8"))

    try:
        calibration = read_result("calibration_model.json")
        lookup = read_result("dynamics_likelihood_lookup_diagnostics.json")
        convergence = read_result("lookup_convergence_t8.json")
        mlr = read_result("mlr_model.json")
        diagnostics = read_result("mlr_diagnostics_t8.json")
        monotonicity = read_result("mlr_monotonicity.json")
        checks = {
            "calibration schema": calibration.get("schema") == "t8a-metallicity-posterior-v1",
            "lookup schema": lookup.get("schema") == "t8c-rice-lookup-v1",
            "MLR schema": mlr.get("schema") == "t8b-monotone-tensor-spline-mlr-v1",
            "retained systems": calibration.get("n_systems") == 14876,
            "lookup convergence": convergence.get("passed") is True,
            "MLR divergences": diagnostics.get("num_divergences") == 0,
            "x monotonicity": monotonicity.get("magnitude_direction_violation_fraction") == 0.0,
            "Z monotonicity": monotonicity.get("metallicity_direction_violation_fraction") == 0.0,
        }
        failures.extend(f"diagnostic failed: {name}" for name, passed in checks.items() if not passed)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        failures.append(f"diagnostic read failed: {exc}")

    if failures:
        print("T8 baseline verification FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"T8 baseline verified: {manifest['baseline_id']}")
    print(f"Result directory: {result_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
