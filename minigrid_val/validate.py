"""Phase 6: MiniGrid validation.

We run a fixed-seed deterministic policy in each of five MiniGrid environments
and record the ground-truth state at every timestep. Each (step, action)
sequence is rendered as a text prompt and fed through the LM exactly like the
Cart-State trajectories.

State variables tracked
-----------------------
    agent_x, agent_y           : ints   -> Ridge regression
    agent_dir                  : 0..3   -> 4-way LogReg
    carrying                   : string -> multi-class LogReg
    door_locked                : bool   -> LogReg
    door_open                  : bool   -> LogReg

If MiniGrid is not installed we fall back to a synthetic substitute that
exercises the same code path (so downstream figures still render).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from tqdm import tqdm


_DIR_NAMES = ["right", "down", "left", "up"]
_ACTION_NAMES = ["turn_left", "turn_right", "forward",
                 "pickup", "drop", "toggle", "done"]


# ----------------------------------------------------------------- env wrap
def _try_import_minigrid():
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
        return gym
    except Exception:
        return None


ENVS = [
    ("MiniGrid-Empty-6x6-v0", "GoToGoal", 15,
     ("agent_x", "agent_y", "agent_dir")),
    ("MiniGrid-DoorKey-5x5-v0", "PickUpKey", 25,
     ("agent_x", "agent_y", "agent_dir", "carrying", "door_locked")),
    ("MiniGrid-Unlock-v0", "UnlockDoor", 25,
     ("agent_x", "agent_y", "agent_dir", "carrying", "door_locked", "door_open")),
    ("MiniGrid-Empty-Random-5x5-v0", "GoToObject", 15,
     ("agent_x", "agent_y", "agent_dir")),
    ("MiniGrid-DoorKey-6x6-v0", "PutNext", 30,
     ("agent_x", "agent_y", "agent_dir", "carrying", "door_locked", "door_open")),
]


# ----------------------------------------------------------------- trajectory
@dataclass
class MiniGridTrajectory:
    traj_id: str
    env_id: str
    task: str                                   # human-readable name
    mission: str
    actions: list[int]                          # taken
    states: list[dict]                          # one per step (post-action)
    tracked_vars: tuple[str, ...]
    checkpoints: list[int]

    # ---------------------------------------------------------------- prompts
    def render_query(self, upto: int) -> str:
        lines = []
        lines.append(f"You control an agent in a MiniGrid '{self.task}' task.")
        lines.append(f"Mission: {self.mission}")
        lines.append(f"Initial state: {json.dumps(self.states[0], sort_keys=True)}")
        # We replay the initial-state line at index 0 already, so start from 1.
        for i in range(1, upto + 1):
            a = self.actions[i - 1]
            lines.append(f"Step {i}: action={_ACTION_NAMES[a]}")
        lines.append("Current state as JSON:")
        return "\n".join(lines)

    def ground_truth(self, upto: int) -> dict:
        return self.states[upto]

    def to_dict(self) -> dict:
        return {
            "traj_id": self.traj_id, "env_id": self.env_id, "task": self.task,
            "mission": self.mission, "actions": self.actions,
            "states": self.states, "tracked_vars": list(self.tracked_vars),
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MiniGridTrajectory":
        return cls(
            traj_id=d["traj_id"], env_id=d["env_id"], task=d["task"],
            mission=d["mission"], actions=list(d["actions"]),
            states=list(d["states"]),
            tracked_vars=tuple(d["tracked_vars"]),
            checkpoints=list(d["checkpoints"]),
        )


# ----------------------------------------------------------------- snapshot
def _snapshot(env, vars_: tuple[str, ...]) -> dict:
    """Build a tracked-variables dict from the live env."""
    out: dict = {}
    out["agent_x"] = int(env.unwrapped.agent_pos[0])
    out["agent_y"] = int(env.unwrapped.agent_pos[1])
    out["agent_dir"] = int(env.unwrapped.agent_dir)
    if "carrying" in vars_:
        c = env.unwrapped.carrying
        out["carrying"] = "none" if c is None else f"{c.color}_{c.type}"
    if "door_locked" in vars_ or "door_open" in vars_:
        from minigrid.core.world_object import Door
        is_locked = False
        is_open = False
        grid = env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if isinstance(obj, Door):
                    is_locked = is_locked or bool(obj.is_locked)
                    is_open = is_open or bool(obj.is_open)
        if "door_locked" in vars_:
            out["door_locked"] = bool(is_locked)
        if "door_open" in vars_:
            out["door_open"] = bool(is_open)
    return out


# ----------------------------------------------------------------- policy
def _scripted_policy(env_id: str, rng: random.Random) -> list[int]:
    """A deterministic action sequence that exercises each env."""
    if "DoorKey" in env_id or "Unlock" in env_id:
        base = [1, 2, 2, 0, 2, 3, 1, 2, 5, 2, 0, 2, 2,
                1, 2, 5, 2, 0, 2, 4, 0, 2, 2, 2, 1, 2,
                2, 0, 2, 5]
    else:
        base = [2, 2, 1, 2, 2, 0, 2, 1, 2, 2, 0, 2,
                2, 1, 2, 0, 2, 2, 1, 2, 2, 2, 2, 0, 2,
                1, 2, 2, 0, 2]
    # Add a touch of seed-driven noise so trajectories differ across seeds.
    out = list(base)
    for i in range(len(out)):
        if rng.random() < 0.07:
            out[i] = rng.randint(0, 5)
    return out


# ----------------------------------------------------------------- generator
def minigrid_backend() -> str:
    """Which backend generate_minigrid_dataset will actually use:
    'minigrid' (real gymnasium + minigrid envs) or 'synthetic_fallback'
    (hand-rolled toy state machine, used only when those packages are
    not importable). Callers should persist this alongside any
    MiniGrid results so a reader can verify provenance from the
    artifact alone rather than trusting an unstated assumption."""
    return "minigrid" if _try_import_minigrid() is not None else "synthetic_fallback"


def generate_minigrid_dataset(
    n_per_env: int = 20, base_seed: int = 0,
) -> list[MiniGridTrajectory]:
    gym = _try_import_minigrid()
    if gym is None:
        return _synthetic_dataset(n_per_env, base_seed)
    trajs: list[MiniGridTrajectory] = []
    for env_id, task, length, vars_ in ENVS:
        for i in range(n_per_env):
            seed = base_seed + i + hash(env_id) % 1000
            rng = random.Random(seed)
            env = gym.make(env_id)
            env.reset(seed=seed)
            mission = getattr(env.unwrapped, "mission", task)
            states = [_snapshot(env, vars_)]
            actions: list[int] = []
            policy = _scripted_policy(env_id, rng)[:length]
            for a in policy:
                _, _, term, trunc, _ = env.step(a)
                actions.append(int(a))
                states.append(_snapshot(env, vars_))
                if term or trunc:
                    # Pad with a 'done' no-op so length is fixed.
                    while len(actions) < length:
                        actions.append(6)
                        states.append(states[-1])
                    break
            # Ensure exactly length actions.
            while len(actions) < length:
                actions.append(6)
                states.append(states[-1])

            T = len(actions)
            checkpoints = sorted({T // 4, T // 2, 3 * T // 4, T - 1})
            trajs.append(MiniGridTrajectory(
                traj_id=f"{task}_{i:03d}", env_id=env_id, task=task,
                mission=str(mission), actions=actions, states=states,
                tracked_vars=vars_, checkpoints=checkpoints,
            ))
    return trajs


def _synthetic_dataset(n: int, seed: int) -> list[MiniGridTrajectory]:
    """Fallback when MiniGrid not installed: hand-rolled toy state machine."""
    trajs = []
    rng = random.Random(seed)
    for env_id, task, length, vars_ in ENVS:
        for i in range(n):
            x, y, d = rng.randint(1, 4), rng.randint(1, 4), rng.randint(0, 3)
            carrying = "none"
            door_locked = True
            door_open = False
            actions, states = [], [{"agent_x": x, "agent_y": y, "agent_dir": d}]
            for _ in range(length):
                a = rng.randint(0, 5)
                if a == 0:
                    d = (d - 1) % 4
                elif a == 1:
                    d = (d + 1) % 4
                elif a == 2:
                    dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][d]
                    x, y = max(0, min(5, x + dx)), max(0, min(5, y + dy))
                elif a == 3 and "carrying" in vars_:
                    carrying = "yellow_key"
                elif a == 5 and "door_locked" in vars_:
                    if carrying != "none":
                        door_locked = False
                    if not door_locked:
                        door_open = not door_open
                snap = {"agent_x": x, "agent_y": y, "agent_dir": d}
                if "carrying" in vars_:
                    snap["carrying"] = carrying
                if "door_locked" in vars_:
                    snap["door_locked"] = door_locked
                if "door_open" in vars_:
                    snap["door_open"] = door_open
                actions.append(a)
                states.append(snap)
            T = len(actions)
            checkpoints = sorted({T // 4, T // 2, 3 * T // 4, T - 1})
            trajs.append(MiniGridTrajectory(
                traj_id=f"{task}_{i:03d}", env_id=env_id, task=task,
                mission=task, actions=actions, states=states,
                tracked_vars=vars_, checkpoints=checkpoints,
            ))
    return trajs


# ----------------------------------------------------------------- I/O
def save_minigrid(trajs: list[MiniGridTrajectory], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([t.to_dict() for t in trajs], f)


def load_minigrid(path: str | Path) -> list[MiniGridTrajectory]:
    with open(path) as f:
        return [MiniGridTrajectory.from_dict(d) for d in json.load(f)]


# ----------------------------------------------------------------- probes/behavior
def train_minigrid_probes(H: torch.Tensor, examples: list,
                          n_splits: int = 5) -> dict:
    """Train per-variable probes over MiniGrid state variables.
    Reuses the cart probe machinery via a thin adapter."""
    from probing.train_probes import _train_layers, _kfold_traj
    folds = _kfold_traj(examples, k=n_splits, seed=0)
    var_keys = set()
    for e in examples:
        for k in e.target.keys():
            var_keys.add(k)
    results = {}
    for v in sorted(var_keys):
        # Decide kind: bool/string -> classification; int/float -> regression.
        # Heuristic from a sample value.
        sample_val = None
        for e in examples:
            if v in e.target:
                sample_val = e.target[v]
                break
        if isinstance(sample_val, bool) or isinstance(sample_val, str):
            kind = "classification"
        else:
            kind = "regression"
        # Build a 'var_kind' dispatch via a closure-friendly trick: store
        # in target dict directly under top-level keys (already so).
        results[v] = _train_layers_mg(H, examples, v, kind, folds)
    return results


def _train_layers_mg(H: torch.Tensor, examples: list, key: str,
                     kind: str, folds: list) -> "object":
    """A minimal copy of probing._train_layers but reading flat ``target`` dicts."""
    from probing.train_probes import ProbeResult
    import numpy as np
    sel = [i for i, e in enumerate(examples) if key in e.target]
    if len(sel) < 10:
        return ProbeResult(variable=key, kind=kind, n_examples=len(sel),
                           per_layer=[float("nan")] * H.shape[1],
                           best_layer=-1, best_score=float("nan"),
                           per_example=[float("nan")] * len(examples))
    sel = np.asarray(sel)
    pos_of = {g: l for l, g in enumerate(sel.tolist())}
    sub_folds = []
    sel_set = set(sel.tolist())
    for tr_g, te_g in folds:
        tr_l = np.array([pos_of[g] for g in tr_g.tolist() if g in sel_set])
        te_l = np.array([pos_of[g] for g in te_g.tolist() if g in sel_set])
        sub_folds.append((tr_l, te_l))
    if kind == "classification":
        y_raw = [examples[i].target[key] for i in sel]
        # Encode to ints
        vocab = {v: i for i, v in enumerate(sorted({str(x) for x in y_raw}))}
        y = np.asarray([vocab[str(v)] for v in y_raw])
    else:
        y = np.asarray([float(examples[i].target[key]) for i in sel])

    from probing.train_probes import _train_one_layer
    per_layer = []
    per_example = np.full(len(examples), np.nan)
    best_score, best_layer, best_correct = -1e18, -1, None
    for L in range(H.shape[1]):
        Hl = H[sel, L].numpy()
        score, correct = _train_one_layer(Hl, y, sub_folds, kind)
        per_layer.append(score)
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_layer = L
            best_correct = correct
    if best_correct is not None:
        for li, gi in enumerate(sel):
            per_example[gi] = best_correct[li]
    return ProbeResult(
        variable=key, kind=kind, n_examples=int(len(sel)),
        per_layer=per_layer, best_layer=int(best_layer),
        best_score=float(best_score),
        per_example=per_example.tolist(),
    )


def evaluate_minigrid_behavior(trajs: list[MiniGridTrajectory],
                               examples: list,
                               extractor,
                               show_progress: bool = True) -> dict:
    from evaluation.behavior import _extract_first_json
    by_id = {t.traj_id: t for t in trajs}
    pvar_lists: dict[str, list[float]] = {}
    overall = []
    raw_gens = []
    it = tqdm(examples, desc="mg-behavior", disable=not show_progress)
    for k, e in enumerate(it):
        t = by_id[e.traj_id]
        prompt = t.render_query(e.checkpoint)
        gen = extractor.generate(prompt, max_new_tokens=120)
        raw_gens.append(gen)
        obj = _extract_first_json(gen) or {}
        scores = {}
        for v, gt in e.target.items():
            pred = obj.get(v, None)
            if isinstance(gt, bool):
                scores[v] = 1.0 if pred == gt else 0.0
            elif isinstance(gt, str):
                scores[v] = 1.0 if str(pred) == gt else 0.0
            else:
                try:
                    scores[v] = 1.0 if float(pred) == float(gt) else 0.0
                except Exception:
                    scores[v] = 0.0
        overall.append(float(np.mean(list(scores.values()))))
        for v, s in scores.items():
            pvar_lists.setdefault(v, [float("nan")] * len(examples))
            pvar_lists[v][k] = s
    return {"per_example_overall": overall,
            "per_variable": pvar_lists,
            "raw_generations": raw_gens}
