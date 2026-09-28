#!/usr/bin/env bash
# Full reproduction pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

export USE_TF=0
export USE_FLAX=0
export PYTHONPATH=.

echo "=== Phase 1-7: end-to-end main run ==="
python scripts/run_all.py "$@"

echo "=== Cross-arch + multi-seed battery + pressure tests ==="
python scripts/run_battery.py --models gpt2 gpt2-medium --seeds 0 1 --pressure

echo "=== Causal mediation (per-layer patching curve) ==="
python scripts/run_mediation.py --model gpt2 --n-examples 12

echo "=== Figures ==="
python scripts/make_concept_figure.py
python scripts/make_figures.py
python scripts/make_battery_figures.py

echo "=== Summary ==="
python scripts/summarize_results.py

echo "=== Paper PDF ==="
bash scripts/build_paper.sh

echo "Done."
