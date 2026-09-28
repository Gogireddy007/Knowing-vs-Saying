"""Multi-token pooled representations for the dissociation test.

The original fidelity F is computed from the *last prompt token* only.
The model's autoregressive generation, by contrast, attends over the
entire prefix. Comparing a single-vector probe to a full-context
read-out is not commensurable: it conflates "the state is not encoded"
with "the state is not encoded *in the last token*".

This module adds context-aware probe inputs:

    last  : phi_L(h_t)                 (original; single vector)
    mean  : (1/T) sum_i phi_i(h_t)     (mean-pool over prompt tokens)
    attn  : softmax(phi W) . phi       (learned-free attention pool:
                                        a fixed query reads the prefix)

If F rises to meet B as the probe is allowed to read more of the
prefix, the negative gap was a probe-locality artefact and the state
is recovered *compositionally over the prefix* rather than stored in
one vector. That is the distributed-readout result.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cart_state.generator import Trajectory
from probing.extract import Extractor, ProbeExample


@torch.no_grad()
def encode_pooled(extractor: Extractor, prompt: str,
                  max_length: int = 1024) -> dict[str, torch.Tensor]:
    """Return three per-layer representations, each [L+1, d_model]:
        'last' : last-token hidden state
        'mean' : mean over all prompt tokens
        'attn' : magnitude-weighted pool (||phi_i|| as the attention
                 weight) over all prompt tokens -- a parameter-free
                 stand-in for an attention read-out.
    """
    ids = extractor.tok(extractor._wrap(prompt), return_tensors="pt",
                        truncation=True,
                        max_length=max_length).to(extractor.device)
    out = extractor.model(**ids, output_hidden_states=True, use_cache=False)
    last, mean, attn = [], [], []
    for h in out.hidden_states:                # each [1, T, d]
        H = h[0].float().cpu()                 # [T, d]
        last.append(H[-1])
        mean.append(H.mean(0))
        w = torch.softmax(H.norm(dim=-1), dim=0)   # [T]
        attn.append((w[:, None] * H).sum(0))
    return {
        "last": torch.stack(last, 0),
        "mean": torch.stack(mean, 0),
        "attn": torch.stack(attn, 0),
    }


def extract_pooled(trajectories: list[Trajectory], out_dir: str | Path,
                   model_name: str = "gpt2",
                   max_examples: int | None = None,
                   show_progress: bool = True
                   ) -> tuple[dict[str, torch.Tensor], list[ProbeExample]]:
    """Extract last/mean/attn pooled hidden states for every checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = Extractor(model_name=model_name)

    examples: list[ProbeExample] = []
    pairs: list[tuple[int, int]] = []
    for ti, t in enumerate(trajectories):
        for s in t.checkpoints:
            examples.append(ProbeExample(
                traj_id=t.traj_id, rung=t.rung, checkpoint=s,
                length=len(t.instructions), target=t.states[s].to_dict(),
                items=list(t.items), flags=list(t.flags)))
            pairs.append((ti, s))
            if max_examples and len(examples) >= max_examples:
                break
        if max_examples and len(examples) >= max_examples:
            break

    nL = ex.n_layers + 1
    banks = {k: torch.empty(len(examples), nL, ex.d_model)
             for k in ("last", "mean", "attn")}
    it = tqdm(pairs, desc=f"pooled[{model_name}]", disable=not show_progress)
    for k, (ti, s) in enumerate(it):
        rep = encode_pooled(ex, trajectories[ti].render_query(s))
        for name in banks:
            banks[name][k] = rep[name]

    for name, bank in banks.items():
        torch.save(bank, out_dir / f"hidden_{name}.pt")
    with open(out_dir / "examples.json", "w") as f:
        json.dump([e.to_dict() for e in examples], f)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"model": model_name, "n_layers_plus_embed": nL,
                   "d_model": ex.d_model}, f)
    return banks, examples


def load_pooled(in_dir: str | Path
                ) -> tuple[dict[str, torch.Tensor], list[ProbeExample], dict]:
    in_dir = Path(in_dir)
    banks = {name: torch.load(in_dir / f"hidden_{name}.pt",
                              map_location="cpu", weights_only=True)
             for name in ("last", "mean", "attn")}
    with open(in_dir / "examples.json") as f:
        examples = [ProbeExample(**d) for d in json.load(f)]
    with open(in_dir / "config.json") as f:
        cfg = json.load(f)
    return banks, examples, cfg
