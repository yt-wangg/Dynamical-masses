#!/usr/bin/env bash
# Run the MLR stage of the main model: fixed p(u~) shape with the u/sigma_u > 3
# truncation normalization (default). Extra arguments are passed to the runner,
# e.g. --data, --output-dir, --metallicity-posterior, --dynamics-lookup.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -u "$repo_root/src/examples/run_hierarchical_metallicity_test.py" --stage mlr "$@"
