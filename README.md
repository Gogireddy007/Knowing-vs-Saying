# The Dissociation Test

Decomposing LLM-agent failures into **world-model fidelity degradation** and
**deployment failures** via the fidelity-behavior gap

    G(t) = F(t) - B(t)

where `F` is what a linear probe can recover from the model's hidden state,
and `B` is what the model actually emits.

## Layout

| Path | Purpose |
|------|---------|
| `cart_state/`     | Synthetic Cart-State environment (Phase 1) |
| `probing/`        | Hidden-state extraction + linear probes (Phase 2–3) |
| `evaluation/`     | Behavioral JSON-generation accuracy (Phase 4) |
| `analysis/`       | Gap computation, null calibration, statistics (Phase 5) |
| `minigrid_val/`   | MiniGrid validation (Phase 6) |
| `interventions/`  | Activation patching, steering, constrained decoding (Phase 7) |
| `scripts/`        | Top-level end-to-end runners |
| `paper/`          | LaTeX manuscript + figures + tables |
| `results/`        | Numerical results, figures, tables produced by runs |
| `tests/`          | Unit tests (pytest) |

## Quick start

    python -m pip install -r requirements.txt
    python scripts/run_all.py --quick      # ~5 min smoke pipeline
    python scripts/run_all.py --full       # full ladder

The full pipeline produces every figure and table referenced in `paper/main.tex`.

## License

Code and Cart-State data: MIT.
# Knowing-vs.-Saying
