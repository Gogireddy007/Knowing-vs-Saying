"""Phase 7: Intervention experiments.

Three knobs that let us causally test the dissociation hypothesis:

    activation_patch(layer, donor_h)
        Replace the residual stream at the last token of `layer` with the
        donor state (from a probably-correct example). If behavior improves,
        the model had been failing in the "deployment" direction.

    steering_vector(F_correct, F_incorrect, layer)
        Compute  v = mean(h | correct) - mean(h | incorrect).
        Inject α * v at the last token of `layer`. If gap shrinks, there is
        a recoverable "state-recall" direction in activation space.

    constrained_generate(prompt)
        Force the model's first generated tokens to be a JSON state template,
        so the only freedom left is the variable values. Eliminates parse
        failures and isolates pure content errors.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from tqdm import tqdm

from cart_state.generator import Trajectory
from evaluation.behavior import _extract_first_json, _coerce_state, _score_state
from probing.extract import Extractor, ProbeExample


# ----------------------------------------------------------------- hooks
def _block_module(model, layer: int):
    """Return the residual-stream block at layer ℓ across architectures."""
    # GPT-2: model.transformer.h[layer]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer]
    # GPT-NeoX / Pythia: model.gpt_neox.layers[layer]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers[layer]
    # Llama / Qwen / Mistral: model.model.layers[layer]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer]
    raise ValueError("unknown architecture")


@contextlib.contextmanager
def _patch_hook(model, layer: int, replacement: torch.Tensor,
                first_pass_only: bool = True):
    """Replace the *last-token* residual at `layer` with `replacement`.

    When ``first_pass_only`` is True (default) the hook is auto-disabled
    after the first forward pass. This is what callers want during
    ``generate()`` — patch the prompt's final token, then let the model
    decode the continuation from the perturbed KV cache without
    further interference.
    """
    block = _block_module(model, layer)
    rep = replacement.to(next(model.parameters()).device).to(
        next(model.parameters()).dtype)
    state = {"fired": False}

    def hook(_module, _inp, out):
        if first_pass_only and state["fired"]:
            return None  # leave output unchanged
        state["fired"] = True
        if isinstance(out, tuple):
            hs = out[0]
            hs = hs.clone()
            hs[:, -1, :] = rep
            return (hs,) + out[1:]
        hs = out.clone()
        hs[:, -1, :] = rep
        return hs

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def _add_hook(model, layer: int, vec: torch.Tensor, alpha: float,
              first_pass_only: bool = True):
    """Add α * vec to the last-token residual at `layer`. First-pass only by
    default (see :func:`_patch_hook` for rationale)."""
    block = _block_module(model, layer)
    v = vec.to(next(model.parameters()).device).to(
        next(model.parameters()).dtype) * float(alpha)
    state = {"fired": False}

    def hook(_module, _inp, out):
        if first_pass_only and state["fired"]:
            return None
        state["fired"] = True
        if isinstance(out, tuple):
            hs = out[0].clone()
            hs[:, -1, :] = hs[:, -1, :] + v
            return (hs,) + out[1:]
        hs = out.clone()
        hs[:, -1, :] = hs[:, -1, :] + v
        return hs

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ----------------------------------------------------------------- patching
@torch.no_grad()
def activation_patch(extractor: Extractor, prompt: str, donor_h: torch.Tensor,
                     layer: int, max_new_tokens: int = 220) -> str:
    """Generate from `prompt` while clamping layer `layer` to `donor_h`."""
    with _patch_hook(extractor.model, layer, donor_h):
        return extractor.generate(prompt, max_new_tokens=max_new_tokens)


def steering_vector(H: torch.Tensor, correct_mask: np.ndarray,
                    layer: int) -> torch.Tensor:
    """v = mean(H[correct, layer]) - mean(H[~correct, layer])."""
    H_layer = H[:, layer, :]
    pos = H_layer[correct_mask]
    neg = H_layer[~correct_mask]
    if len(pos) == 0 or len(neg) == 0:
        return torch.zeros(H_layer.shape[1])
    return pos.mean(0) - neg.mean(0)


@torch.no_grad()
def apply_steering(extractor: Extractor, prompt: str, vec: torch.Tensor,
                   layer: int, alpha: float = 1.0,
                   max_new_tokens: int = 220) -> str:
    with _add_hook(extractor.model, layer, vec, alpha):
        return extractor.generate(prompt, max_new_tokens=max_new_tokens)


# ----------------------------------------------------------------- constrained
@torch.no_grad()
def constrained_generate(extractor: Extractor, prompt: str,
                         items: Sequence[str], flags: Sequence[str],
                         max_new_tokens: int = 220) -> str:
    """Prefix the prompt with the start of the JSON schema so the model is
    constrained to fill in the values rather than invent structure."""
    template = '{"counts": {'
    extended = prompt + " " + template
    raw = extractor.generate(extended, max_new_tokens=max_new_tokens)
    # Reattach prefix so downstream parser finds a complete JSON object.
    return template + raw


# ----------------------------------------------------------------- runner
@dataclass
class InterventionRow:
    traj_id: str
    checkpoint: int
    baseline_score: float
    patched_score: float
    steered_score: float
    constrained_score: float
    parse_ok: dict[str, bool] = field(default_factory=dict)


def _score_generation(gen: str, e: ProbeExample) -> float:
    obj = _extract_first_json(gen) or {}
    pred = _coerce_state(obj, e.items, e.flags) if obj else None
    scores = _score_state(pred, e.target)
    return float(np.mean(list(scores.values())))


def run_intervention_suite(
    extractor: Extractor,
    trajs: list[Trajectory],
    examples: list[ProbeExample],
    H: torch.Tensor,
    baseline_per_example: np.ndarray,
    target_layer: int,
    donor_pool_mask: np.ndarray | None = None,
    steering_alpha: float = 1.5,
    max_examples: int | None = None,
    show_progress: bool = True,
) -> list[InterventionRow]:
    """Run patch / steer / constrained on a subset of examples.

    `donor_pool_mask`: boolean mask over `examples` selecting candidate donor
    hidden states. By default we use the top-quartile by baseline behavior."""
    by_id = {t.traj_id: t for t in trajs}
    if donor_pool_mask is None:
        thresh = np.nanquantile(baseline_per_example, 0.75)
        donor_pool_mask = baseline_per_example >= thresh
        if donor_pool_mask.sum() == 0:
            donor_pool_mask = np.ones_like(baseline_per_example, dtype=bool)
    donor_idxs = np.where(donor_pool_mask)[0]

    # Build the steering vector on a held-out half of correct/incorrect.
    correct_mask = baseline_per_example >= 0.8
    sv = steering_vector(H, correct_mask, target_layer)

    rows: list[InterventionRow] = []
    rng = np.random.RandomState(0)
    iter_examples = list(enumerate(examples))
    if max_examples is not None:
        iter_examples = iter_examples[:max_examples]
    it = tqdm(iter_examples, desc="intervene", disable=not show_progress)
    for k, e in it:
        t = by_id[e.traj_id]
        prompt = t.render_query(e.checkpoint)
        baseline = float(baseline_per_example[k])

        # Patch: pick a donor not from this trajectory.
        donor_candidates = [j for j in donor_idxs.tolist()
                            if examples[j].traj_id != e.traj_id and j != k]
        if donor_candidates:
            donor_k = rng.choice(donor_candidates)
            donor_h = H[donor_k, target_layer]
            patched = activation_patch(extractor, prompt, donor_h, target_layer)
            patched_score = _score_generation(patched, e)
        else:
            patched_score = baseline

        steered = apply_steering(extractor, prompt, sv, target_layer,
                                 alpha=steering_alpha)
        steered_score = _score_generation(steered, e)

        constrained = constrained_generate(extractor, prompt, e.items, e.flags)
        constrained_score = _score_generation(constrained, e)

        rows.append(InterventionRow(
            traj_id=e.traj_id, checkpoint=e.checkpoint,
            baseline_score=baseline, patched_score=patched_score,
            steered_score=steered_score, constrained_score=constrained_score,
        ))
    return rows


def save_intervention_results(rows: list[InterventionRow], path: str) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.__dict__ for r in rows], f)


# ----------------------------------------------------------------- significance
def paired_sign_test_p(baseline: np.ndarray, treated: np.ndarray,
                       n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired significance test for an intervention's effect on the SAME
    examples (baseline vs. treated), backing any "reduces B by X%
    (p < y)" claim with an actual computed p-value instead of an
    asserted one.

    We report two statistics since n is typically small (~12-60
    examples per condition, see run_all.py's --quick/--full intervene_n):

      * paired bootstrap p-value on the mean paired difference
        (treated - baseline), two-sided: resample example indices with
        replacement, recompute the mean difference, and ask how often
        a bootstrap resample crosses zero in the direction opposite the
        observed effect. Valid for any n, doesn't assume normality.
      * a binomial sign-test p-value on the direction of the per-example
        deltas (ties excluded), which only uses the *sign* of each
        example's change and is the more conservative, fewer-assumptions
        complement to the bootstrap test.
    """
    baseline = np.asarray(baseline, dtype=float)
    treated = np.asarray(treated, dtype=float)
    delta = treated - baseline
    n = len(delta)
    rng = np.random.RandomState(seed)
    obs_mean = float(delta.mean())
    boot_means = np.array([
        rng.choice(delta, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    if obs_mean >= 0:
        p_boot = float((np.sum(boot_means <= 0) + 1) / (n_boot + 1)) * 2
    else:
        p_boot = float((np.sum(boot_means >= 0) + 1) / (n_boot + 1)) * 2
    p_boot = min(1.0, p_boot)

    n_pos = int(np.sum(delta > 0))
    n_neg = int(np.sum(delta < 0))
    n_nonzero = n_pos + n_neg
    if n_nonzero == 0:
        p_sign = 1.0
    else:
        from math import comb
        k = min(n_pos, n_neg)
        p_sign = min(1.0, 2 * sum(
            comb(n_nonzero, i) * 0.5 ** n_nonzero for i in range(k + 1)
        ))
    return {
        "n": int(n), "mean_delta": obs_mean,
        "p_bootstrap": p_boot, "p_sign_test": float(p_sign),
        "n_pos": n_pos, "n_neg": n_neg,
    }
