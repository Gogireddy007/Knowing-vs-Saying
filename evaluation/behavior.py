"""Phase 4: Behavioral accuracy.

Given the same prompt the probes saw, ask the model to *generate* the cart
state as JSON. Parse the generation (robust to extra prose), score each
variable, and classify error modes.

Behavioral correctness per variable is exact-match (with a 1-cent tolerance
on prices/total).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from cart_state.generator import Trajectory
from cart_state.state import CartState
from probing.extract import Extractor, ProbeExample


# ----------------------------------------------------------------- parser
_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.S)


def _extract_first_json(text: str) -> dict | None:
    """Best-effort: find the first balanced { ... } substring and json-parse it."""
    # Walk character by character keeping a depth counter, since cart JSON
    # contains nested objects.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start:i + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        return None
    return None


def parse_state_from_generation(text: str,
                                items: list[str], flags: list[str]
                                ) -> tuple[CartState | None, dict]:
    """Return (parsed state or None, telemetry dict)."""
    tele: dict = {"raw_len": len(text), "parsed": False,
                  "fallback_regex": False}
    obj = _extract_first_json(text)
    if obj is None:
        # Regex fallback: try to recover counts dict.
        m = re.search(r'"counts"\s*:\s*\{([^}]*)\}', text)
        tele["fallback_regex"] = m is not None
        if m is None:
            return None, tele
        # Build a minimal state from what we can find.
        body = m.group(1)
        counts = {}
        for k, v in re.findall(r'"([^"]+)"\s*:\s*(-?\d+)', body):
            try:
                counts[k] = int(v)
            except Exception:
                pass
        partial = {"counts": counts, "prices": {}, "flags": {}, "discount": 1.0}
        return _coerce_state(partial, items, flags), tele

    tele["parsed"] = True
    return _coerce_state(obj, items, flags), tele


def _coerce_state(d: dict, items: list[str], flags: list[str]) -> CartState:
    counts = {it: int(d.get("counts", {}).get(it, 0)) for it in items}
    prices = {it: float(d.get("prices", {}).get(it, 1.0)) for it in items}
    flgs = {fl: bool(d.get("flags", {}).get(fl, False)) for fl in flags}
    disc = float(d.get("discount", 1.0))
    return CartState(counts=counts, prices=prices, flags=flgs, discount=disc)


# ----------------------------------------------------------------- scoring
def _score_state(pred: CartState | None, gt: dict) -> dict[str, float]:
    """Return per-variable correctness (1.0 / 0.0) including:
       counts:<item>, prices:<item>, flag:<f>, total."""
    out: dict[str, float] = {}
    items = list(gt["counts"].keys())
    flags = list(gt["flags"].keys())
    for it in items:
        gt_c = int(gt["counts"][it])
        pr_c = pred.counts.get(it, 0) if pred else 0
        out[f"counts:{it}"] = 1.0 if pr_c == gt_c else 0.0
        gt_p = float(gt["prices"][it])
        pr_p = pred.prices.get(it, 0.0) if pred else 0.0
        out[f"prices:{it}"] = 1.0 if abs(pr_p - gt_p) <= 0.01 else 0.0
    for fl in flags:
        gt_f = bool(gt["flags"][fl])
        pr_f = pred.flags.get(fl, False) if pred else False
        out[f"flag:{fl}"] = 1.0 if pr_f == gt_f else 0.0
    pr_total = pred.total if pred else 0.0
    gt_total = float(gt["total"])
    out["total"] = 1.0 if abs(pr_total - gt_total) <= 0.01 else 0.0
    return out


def _classify_error(pred: CartState | None, gt: dict) -> list[str]:
    if pred is None:
        return ["parse_failure"]
    errs: list[str] = []
    # Flag hallucination: predicted True where GT False.
    for fl, v in gt["flags"].items():
        if pred.flags.get(fl, False) and not v:
            errs.append("flag_hallucination")
            break
    # Count off-by-one.
    for it, v in gt["counts"].items():
        if abs(int(pred.counts.get(it, 0)) - int(v)) == 1:
            errs.append("count_off_by_one")
            break
    # Price drift.
    for it, v in gt["prices"].items():
        d = abs(float(pred.prices.get(it, 0.0)) - float(v))
        if 0.01 < d <= 1.0:
            errs.append("price_drift")
            break
    # Total miscalculation given correct components.
    pr_total_from_pred = pred.total
    if abs(pr_total_from_pred - float(gt["total"])) > 0.01:
        errs.append("total_miscalc")
    return errs


# ----------------------------------------------------------------- main
@dataclass
class BehaviorResult:
    per_example_overall: list[float]                   # mean correctness per example
    per_variable: dict[str, list[float]]               # var -> correctness per example (NaN if N/A)
    error_tags: list[list[str]]                         # per example
    raw_generations: list[str] = field(default_factory=list)


def evaluate_behavior(
    trajectories: list[Trajectory],
    examples: list[ProbeExample],
    model_name: str = "gpt2-medium",
    device: torch.device | None = None,
    max_new_tokens: int = 220,
    show_progress: bool = True,
    extractor: Extractor | None = None,
    keep_raw: bool = True,
) -> BehaviorResult:
    extractor = extractor or Extractor(model_name=model_name, device=device)
    traj_by_id = {t.traj_id: t for t in trajectories}

    overall = []
    pvar: dict[str, list[float]] = {}
    tags: list[list[str]] = []
    raws: list[str] = []

    it = tqdm(examples, desc="behavior", disable=not show_progress)
    for e in it:
        t = traj_by_id[e.traj_id]
        prompt = t.render_query(e.checkpoint)
        gen = extractor.generate(prompt, max_new_tokens=max_new_tokens)
        if keep_raw:
            raws.append(gen)
        pred, _ = parse_state_from_generation(gen, e.items, e.flags)
        scores = _score_state(pred, e.target)
        overall.append(float(np.mean(list(scores.values()))))
        for k, v in scores.items():
            pvar.setdefault(k, [float("nan")] * len(examples))
        # Fill this example's slots.
        idx = len(overall) - 1
        for k, v in scores.items():
            pvar[k][idx] = v
        tags.append(_classify_error(pred, e.target))

    # Pad lists in pvar to N (variables that never appeared for some examples).
    N = len(examples)
    for k in pvar:
        if len(pvar[k]) < N:
            pvar[k] = pvar[k] + [float("nan")] * (N - len(pvar[k]))

    return BehaviorResult(
        per_example_overall=overall,
        per_variable=pvar,
        error_tags=tags,
        raw_generations=raws,
    )


def behavior_per_example(res: BehaviorResult) -> np.ndarray:
    return np.asarray(res.per_example_overall)


def save_behavior(res: BehaviorResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({
            "per_example_overall": res.per_example_overall,
            "per_variable": res.per_variable,
            "error_tags": res.error_tags,
            "raw_generations": res.raw_generations,
        }, f)


def load_behavior(path: str | Path) -> BehaviorResult:
    with open(path) as f:
        d = json.load(f)
    return BehaviorResult(**d)
