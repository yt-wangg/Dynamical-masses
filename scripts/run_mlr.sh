#!/usr/bin/env bash
# Run the default EM-style alternating-MAP MLR workflow.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$repo_root/src/examples/run_em_mlr_pilot.py" "$@"
