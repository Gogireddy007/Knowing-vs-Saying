"""Hidden-state extraction pipeline.

For each (trajectory, checkpoint) pair we:
    1.  Render the prompt up to and including the checkpoint instruction,
        ending with ``Current cart state as JSON:``.
    2.  Run the model once with ``output_hidden_states=True``.
    3.  Save the **last-token** hidden state at every layer.

The result is a single torch tensor of shape
    [N_examples, N_layers + 1, d_model]
where layer 0 is the embedding output and layers 1..L are transformer blocks.

We also save a parallel metadata index so downstream code can join hidden
states back to trajectories and timesteps without re-tokenising.
"""
from __future__ import annotations

import json
import os

# Disable TensorFlow / Flax backends in transformers — they trigger a
# segfault on macOS Apple-Silicon + Python 3.13 when TF tries to preload
# its native libs. We never use them.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from cart_state.generator import Trajectory


# ----------------------------------------------------------------- device
def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------- record
@dataclass
class ProbeExample:
    traj_id: str
    rung: str
    checkpoint: int           # index into trajectory.instructions
    length: int               # full trajectory length L
    target: dict              # ground-truth state.to_dict()
    items: list[str]
    flags: list[str]

    def to_dict(self) -> dict:
        return self.__dict__


# ----------------------------------------------------------------- extractor
class Extractor:
    def __init__(self, model_name: str = "gpt2-medium",
                 device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32,
                 chat: bool | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or best_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        self.model_name = model_name
        # Instruction-tuned models are evaluated through their chat
        # template so the comparison is fair (the base prompt is wrapped
        # as a user turn with a generation prompt appended). Auto-detect
        # unless overridden.
        if chat is None:
            chat = (self.tok.chat_template is not None
                    and "instruct" in model_name.lower())
        self.chat = chat

    def _wrap(self, prompt: str) -> str:
        if not self.chat:
            return prompt
        msgs = [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def encode(self, prompt: str, max_length: int = 1024) -> torch.Tensor:
        """Return last-token hidden states per layer: tensor [L+1, d_model]."""
        ids = self.tok(self._wrap(prompt), return_tensors="pt",
                       truncation=True, max_length=max_length).to(self.device)
        out = self.model(**ids, output_hidden_states=True, use_cache=False)
        # out.hidden_states: tuple of length L+1, each [1, T, d]
        last = torch.stack([h[0, -1, :].float().cpu()
                            for h in out.hidden_states], dim=0)
        return last  # [L+1, d]

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 220,
                 max_length: int = 1024) -> str:
        ids = self.tok(self._wrap(prompt), return_tensors="pt",
                       truncation=True, max_length=max_length).to(self.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tok.eos_token_id,
        )
        # Strip the prompt back off.
        gen = out[0, ids["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


# ----------------------------------------------------------------- main API
def extract_hidden_states(
    trajectories: list[Trajectory],
    out_dir: str | Path,
    model_name: str = "gpt2-medium",
    device: torch.device | None = None,
    every_step: bool = False,
    max_examples: int | None = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, list[ProbeExample]]:
    """Extract hidden states for every (traj, checkpoint) pair.

    If ``every_step`` is True, extract at *every* timestep instead of just the
    four pre-baked checkpoints — useful for fine-grained gap curves.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = Extractor(model_name=model_name, device=device)

    examples: list[ProbeExample] = []
    pairs: list[tuple[int, int]] = []
    for ti, t in enumerate(trajectories):
        steps = range(len(t.instructions)) if every_step else t.checkpoints
        for s in steps:
            examples.append(ProbeExample(
                traj_id=t.traj_id, rung=t.rung, checkpoint=s,
                length=len(t.instructions),
                target=t.states[s].to_dict(),
                items=list(t.items), flags=list(t.flags),
            ))
            pairs.append((ti, s))
            if max_examples is not None and len(examples) >= max_examples:
                break
        if max_examples is not None and len(examples) >= max_examples:
            break

    H = torch.empty(len(examples), extractor.n_layers + 1, extractor.d_model,
                    dtype=torch.float32)
    it = tqdm(pairs, desc="extract", disable=not show_progress)
    for k, (ti, s) in enumerate(it):
        prompt = trajectories[ti].render_query(s)
        H[k] = extractor.encode(prompt)

    torch.save(H, out_dir / "hidden_states.pt")
    with open(out_dir / "examples.json", "w") as f:
        json.dump([e.to_dict() for e in examples], f)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"model": model_name,
                   "n_layers_plus_embed": extractor.n_layers + 1,
                   "d_model": extractor.d_model,
                   "every_step": every_step}, f)
    return H, examples


def load_hidden_states(
    in_dir: str | Path,
) -> tuple[torch.Tensor, list[ProbeExample], dict]:
    in_dir = Path(in_dir)
    H = torch.load(in_dir / "hidden_states.pt", map_location="cpu",
                   weights_only=True)
    with open(in_dir / "examples.json") as f:
        examples = [ProbeExample(**d) for d in json.load(f)]
    with open(in_dir / "config.json") as f:
        cfg = json.load(f)
    return H, examples, cfg


# ----------------------------------------------------------------- CLI
def main() -> None:
    import argparse
    from cart_state.generator import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cart_state.json")
    ap.add_argument("--out", default="results/cache/hidden")
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--every-step", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    trajs = load_dataset(args.data)
    if args.limit:
        trajs = trajs[:args.limit]
    H, ex = extract_hidden_states(trajs, args.out, model_name=args.model,
                                  every_step=args.every_step)
    print(f"extracted {H.shape} from {len(trajs)} trajectories")


if __name__ == "__main__":
    main()
