"""Cart-State trajectory generator.

A trajectory = (initial state, list of instructions, list of post-step states).
The generator is fully deterministic from a seed and supports a complexity
ladder of lengths L ∈ {10, 20, 30, 40}.

Branch-and-discard checkpoints are stored as a list of indices
    t_c ∈ {⌊L/4⌋, ⌊L/2⌋, ⌊3L/4⌋, L-1}
into the trajectory. Probing/behavior are evaluated at these timesteps only;
the main trajectory itself is never polluted with the JSON-state query.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from cart_state.state import CartState, _round_money


# ----------------------------------------------------------------- vocabulary
ITEM_POOL = [
    "apple", "bread", "candle", "diaper", "egg",
    "fig", "grape", "honey", "ink", "jam",
]
FLAG_POOL = [
    "gift_wrap", "express", "fragile", "insured",
    "signature", "weekend_delivery",
]


# ----------------------------------------------------------------- complexity
@dataclass(frozen=True)
class Rung:
    length: int
    n_items: int
    n_flags: int
    n_trajectories: int
    allowed: tuple[str, ...]
    name: str


LADDER: tuple[Rung, ...] = (
    Rung(
        length=10, n_items=3, n_flags=2, n_trajectories=50,
        allowed=("ADD", "REMOVE", "SET_PRICE"),
        name="L10",
    ),
    Rung(
        length=20, n_items=5, n_flags=3, n_trajectories=50,
        allowed=("ADD", "REMOVE", "SET_PRICE", "ENABLE_FLAG", "DISABLE_FLAG"),
        name="L20",
    ),
    Rung(
        length=30, n_items=6, n_flags=4, n_trajectories=50,
        allowed=("ADD", "REMOVE", "SET_PRICE", "ENABLE_FLAG", "DISABLE_FLAG",
                 "APPLY_DISCOUNT", "CLEAR_CART"),
        name="L30",
    ),
    Rung(
        length=40, n_items=7, n_flags=5, n_trajectories=50,
        allowed=("ADD", "REMOVE", "SET_PRICE", "ENABLE_FLAG", "DISABLE_FLAG",
                 "APPLY_DISCOUNT", "CLEAR_CART", "DOUBLE_QUANTITY"),
        name="L40",
    ),
)


# ----------------------------------------------------------------- instruction
@dataclass
class Instruction:
    op: str
    args: tuple

    def render(self) -> str:
        if self.op == "ADD":
            return f"ADD {self.args[0]} {self.args[1]}"
        if self.op == "REMOVE":
            return f"REMOVE {self.args[0]} {self.args[1]}"
        if self.op == "SET_PRICE":
            return f"SET_PRICE {self.args[0]} {self.args[1]:.2f}"
        if self.op == "ENABLE_FLAG":
            return f"ENABLE_FLAG {self.args[0]}"
        if self.op == "DISABLE_FLAG":
            return f"DISABLE_FLAG {self.args[0]}"
        if self.op == "CLEAR_CART":
            return "CLEAR_CART"
        if self.op == "DOUBLE_QUANTITY":
            return f"DOUBLE_QUANTITY {self.args[0]}"
        if self.op == "APPLY_DISCOUNT":
            return f"APPLY_DISCOUNT {self.args[0]}"
        raise ValueError(self.op)

    def apply(self, s: CartState) -> CartState:
        """Return a *new* state with this instruction applied."""
        s = s.copy()
        op = self.op
        if op == "ADD":
            item, qty = self.args
            s.counts[item] = s.counts.get(item, 0) + int(qty)
        elif op == "REMOVE":
            item, qty = self.args
            s.counts[item] = max(0, s.counts.get(item, 0) - int(qty))
        elif op == "SET_PRICE":
            item, price = self.args
            s.prices[item] = _round_money(float(price))
        elif op == "ENABLE_FLAG":
            s.flags[self.args[0]] = True
        elif op == "DISABLE_FLAG":
            s.flags[self.args[0]] = False
        elif op == "CLEAR_CART":
            for k in s.counts:
                s.counts[k] = 0
        elif op == "DOUBLE_QUANTITY":
            item = self.args[0]
            s.counts[item] = s.counts.get(item, 0) * 2
        elif op == "APPLY_DISCOUNT":
            pct = float(self.args[0])
            s.discount = max(0.0, min(1.0, s.discount * (1.0 - pct / 100.0)))
        else:
            raise ValueError(op)
        return s


# ----------------------------------------------------------------- trajectory
@dataclass
class Trajectory:
    traj_id: str
    rung: str
    items: list[str]
    flags: list[str]
    initial: CartState
    instructions: list[Instruction]
    states: list[CartState]  # length L; states[i] is post-instructions[0..i]
    checkpoints: list[int]

    # ---------------------------------------------------------------- prompts
    def render_prefix(self, upto: int) -> str:
        """Render the instruction history up to *and including* step `upto`."""
        lines = []
        lines.append("You manage a shopping cart. Track its state exactly.")
        lines.append(f"Items: [{', '.join(self.items)}]")
        lines.append(f"Flags: [{', '.join(self.flags)}]")
        lines.append(f"Initial state: {self.initial.to_json()}")
        for i in range(upto + 1):
            lines.append(f"Step {i + 1}: {self.instructions[i].render()}")
        return "\n".join(lines)

    def render_query(self, upto: int) -> str:
        return self.render_prefix(upto) + "\nCurrent cart state as JSON:"

    def ground_truth(self, upto: int) -> CartState:
        return self.states[upto]

    # ---------------------------------------------------------------- ser/de
    def to_dict(self) -> dict:
        return {
            "traj_id": self.traj_id,
            "rung": self.rung,
            "items": self.items,
            "flags": self.flags,
            "initial": self.initial.to_dict(),
            "instructions": [{"op": i.op, "args": list(i.args)}
                             for i in self.instructions],
            "states": [s.to_dict() for s in self.states],
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            traj_id=d["traj_id"],
            rung=d["rung"],
            items=list(d["items"]),
            flags=list(d["flags"]),
            initial=CartState.from_dict(d["initial"]),
            instructions=[Instruction(i["op"], tuple(i["args"]))
                          for i in d["instructions"]],
            states=[CartState.from_dict(s) for s in d["states"]],
            checkpoints=list(d["checkpoints"]),
        )


# ----------------------------------------------------------------- sampling
def _sample_instruction(
    rng: random.Random, items: list[str], flags: list[str],
    allowed: tuple[str, ...], current: CartState,
) -> Instruction:
    op = rng.choice(allowed)
    if op == "ADD":
        item = rng.choice(items)
        qty = rng.randint(1, 5)
        return Instruction("ADD", (item, qty))
    if op == "REMOVE":
        # Bias toward items with positive count so REMOVE has an effect.
        pos = [it for it in items if current.counts.get(it, 0) > 0]
        item = rng.choice(pos) if pos else rng.choice(items)
        qty = rng.randint(1, 3)
        return Instruction("REMOVE", (item, qty))
    if op == "SET_PRICE":
        item = rng.choice(items)
        price = round(rng.uniform(0.50, 9.99), 2)
        return Instruction("SET_PRICE", (item, price))
    if op == "ENABLE_FLAG":
        return Instruction("ENABLE_FLAG", (rng.choice(flags),))
    if op == "DISABLE_FLAG":
        return Instruction("DISABLE_FLAG", (rng.choice(flags),))
    if op == "CLEAR_CART":
        return Instruction("CLEAR_CART", ())
    if op == "DOUBLE_QUANTITY":
        return Instruction("DOUBLE_QUANTITY", (rng.choice(items),))
    if op == "APPLY_DISCOUNT":
        return Instruction("APPLY_DISCOUNT", (rng.choice([5, 10, 15, 20, 25]),))
    raise ValueError(op)


def _checkpoint_indices(length: int) -> list[int]:
    # ⌊L/4⌋, ⌊L/2⌋, ⌊3L/4⌋, L-1 — deduped and clipped.
    raw = [length // 4, length // 2, 3 * length // 4, length - 1]
    out: list[int] = []
    for x in raw:
        x = max(0, min(length - 1, x))
        if x not in out:
            out.append(x)
    return out


def generate_trajectory(seed: int, rung: Rung, traj_id: str) -> Trajectory:
    rng = random.Random(seed)
    items = sorted(rng.sample(ITEM_POOL, rung.n_items))
    flags = sorted(rng.sample(FLAG_POOL, rung.n_flags))
    prices = {it: round(rng.uniform(0.50, 9.99), 2) for it in items}
    initial = CartState.initial(items, flags, prices)

    instructions: list[Instruction] = []
    states: list[CartState] = []
    cur = initial
    for _ in range(rung.length):
        ins = _sample_instruction(rng, items, flags, rung.allowed, cur)
        cur = ins.apply(cur)
        instructions.append(ins)
        states.append(cur)

    return Trajectory(
        traj_id=traj_id,
        rung=rung.name,
        items=items,
        flags=flags,
        initial=initial,
        instructions=instructions,
        states=states,
        checkpoints=_checkpoint_indices(rung.length),
    )


def generate_dataset(
    base_seed: int = 0,
    ladder: Iterable[Rung] = LADDER,
) -> list[Trajectory]:
    out: list[Trajectory] = []
    counter = 0
    for rung in ladder:
        for i in range(rung.n_trajectories):
            seed = base_seed * 100_000 + counter
            tid = f"{rung.name}_{i:03d}"
            out.append(generate_trajectory(seed, rung, tid))
            counter += 1
    return out


# ----------------------------------------------------------------- I/O
def save_dataset(trajs: list[Trajectory], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([t.to_dict() for t in trajs], f)


def load_dataset(path: str | Path) -> list[Trajectory]:
    with open(path) as f:
        data = json.load(f)
    return [Trajectory.from_dict(d) for d in data]


# ----------------------------------------------------------------- CLI
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate the Cart-State dataset.")
    ap.add_argument("--out", default="data/cart_state.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="Generate 5 trajectories per rung (smoke).")
    args = ap.parse_args()

    ladder = LADDER
    if args.quick:
        ladder = tuple(
            Rung(r.length, r.n_items, r.n_flags, 5, r.allowed, r.name)
            for r in LADDER
        )

    trajs = generate_dataset(args.seed, ladder)
    save_dataset(trajs, args.out)
    print(f"wrote {len(trajs)} trajectories to {args.out}")
    # Tiny sanity print:
    sample = trajs[0]
    print(f"  example: {sample.traj_id}  L={len(sample.instructions)}")
    print(f"  checkpoints: {sample.checkpoints}")
    print(f"  final state: {sample.states[-1].to_json()}")


if __name__ == "__main__":
    main()
