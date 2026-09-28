"""Unit tests for the Cart-State environment."""
from __future__ import annotations

import json

import pytest

from cart_state import generate_dataset, LADDER
from cart_state.generator import (
    Instruction, generate_trajectory, save_dataset, load_dataset, Rung,
)
from cart_state.state import CartState, _round_money


def test_initial_state_zero_total():
    s = CartState.initial(["apple", "egg"], ["gift_wrap"], {"apple": 1.0, "egg": 0.5})
    assert s.total == 0.0
    assert s.flags == {"gift_wrap": False}


def test_total_recomputed_from_first_principles():
    s = CartState.initial(["apple"], [], {"apple": 1.30})
    s.counts["apple"] = 3
    assert s.total == round(1.30 * 3, 2)
    # Apply a discount and verify.
    s.discount = 0.80
    assert s.total == round(1.30 * 3 * 0.80, 2)


def test_money_rounding_no_banker():
    # 0.005 should go to 0.01 (commercial), not 0.00 (banker).
    assert _round_money(0.005) == 0.01


def test_apply_instruction_does_not_mutate_original():
    s = CartState.initial(["apple"], [], {"apple": 1.0})
    s2 = Instruction("ADD", ("apple", 3)).apply(s)
    assert s.counts["apple"] == 0
    assert s2.counts["apple"] == 3


def test_clear_cart_only_zeros_counts():
    s = CartState.initial(["a", "b"], [], {"a": 1.0, "b": 2.0})
    s.counts["a"] = 5
    s.counts["b"] = 3
    s.flags = {"x": True}
    s2 = Instruction("CLEAR_CART", ()).apply(s)
    assert all(v == 0 for v in s2.counts.values())
    # Flags untouched.
    assert s2.flags == {"x": True}
    # Prices preserved.
    assert s2.prices == s.prices


def test_double_quantity():
    s = CartState.initial(["x"], [], {"x": 1.0})
    s.counts["x"] = 4
    s2 = Instruction("DOUBLE_QUANTITY", ("x",)).apply(s)
    assert s2.counts["x"] == 8


def test_apply_discount_compounds_and_clamps():
    s = CartState.initial(["x"], [], {"x": 1.0})
    s.counts["x"] = 1
    s2 = Instruction("APPLY_DISCOUNT", (50,)).apply(s)
    s3 = Instruction("APPLY_DISCOUNT", (50,)).apply(s2)
    assert abs(s2.discount - 0.5) < 1e-9
    assert abs(s3.discount - 0.25) < 1e-9


def test_remove_never_negative():
    s = CartState.initial(["x"], [], {"x": 1.0})
    s2 = Instruction("REMOVE", ("x", 5)).apply(s)
    assert s2.counts["x"] == 0


def test_generator_deterministic():
    a = generate_trajectory(42, LADDER[1], "t")
    b = generate_trajectory(42, LADDER[1], "t")
    assert a.to_dict() == b.to_dict()


def test_generator_different_seeds_differ():
    a = generate_trajectory(0, LADDER[1], "t")
    b = generate_trajectory(1, LADDER[1], "t")
    assert a.to_dict() != b.to_dict()


def test_trajectory_states_match_apply_chain():
    t = generate_trajectory(7, LADDER[2], "t")
    s = t.initial
    for ins, expected in zip(t.instructions, t.states):
        s = ins.apply(s)
        assert s.equals(expected)


def test_checkpoints_in_range():
    for rung in LADDER:
        t = generate_trajectory(0, rung, "t")
        for c in t.checkpoints:
            assert 0 <= c < rung.length


def test_render_query_ends_with_marker():
    t = generate_trajectory(0, LADDER[0], "t")
    q = t.render_query(t.checkpoints[-1])
    assert q.endswith("Current cart state as JSON:")


def test_dataset_full_size():
    rungs = tuple(Rung(r.length, r.n_items, r.n_flags, 3, r.allowed, r.name)
                  for r in LADDER)
    trajs = generate_dataset(0, rungs)
    assert len(trajs) == 12  # 4 rungs * 3 trajectories
    # Unique ids.
    assert len({t.traj_id for t in trajs}) == 12


def test_round_trip_io(tmp_path):
    rungs = tuple(Rung(r.length, r.n_items, r.n_flags, 2, r.allowed, r.name)
                  for r in LADDER)
    trajs = generate_dataset(0, rungs)
    p = tmp_path / "ds.json"
    save_dataset(trajs, p)
    loaded = load_dataset(p)
    assert len(loaded) == len(trajs)
    for a, b in zip(trajs, loaded):
        assert a.to_dict() == b.to_dict()


def test_ground_truth_is_valid_json():
    t = generate_trajectory(0, LADDER[3], "t")
    for s in t.states:
        json.loads(s.to_json())  # parses cleanly
