"""The cooperative scheduler only moves statements: schedule order must compute what straight-line
order computes, and no level may read a shared-memory slot that the same level writes.

    python -m pytest tests/test_tiered.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import cricket
from cricket import tiered

ROOT = Path(__file__).resolve().parents[1] / "resources"


def _python(expr: str) -> str:
    expr = re.sub(r"(\d(?:[\d.]*)(?:e[-+]?\d+)?)f\b", r"\1", expr)
    for fn in ("sin", "cos", "sqrt", "tan", "exp", "log", "atan", "asin", "acos"):
        expr = expr.replace(f"{fn}f(", f"math.{fn}(")
    return expr.replace("fabsf(", "abs(")


def _trace(robot: str, key: str) -> tuple[str, int, int]:
    cfg = json.loads((ROOT / f"{robot}.json").read_text())
    gen = cricket.generate_robot_source(cricket.GenOptions(
        urdf=ROOT / cfg["urdf"], srdf=ROOT / cfg["srdf"], end_effector=cfg["end_effector"],
        language="cuda", data={"name": cfg["name"]}))
    return gen.data[f"{key}_code"], gen.dimension, gen.data[f"{key}_code_output"]


@pytest.mark.parametrize("robot,key", [("panda", "eejac"), ("panda", "spherefk"), ("baxter", "eejac")])
@pytest.mark.parametrize("units", [3, 8, 32])
def test_schedule_order_matches_straight_line(robot, key, units):
    code, n_in, n_out = _trace(robot, key)
    x = list(np.random.default_rng(0).uniform(-1.0, 1.0, n_in))

    env = {"x": x, "v": [0.0] * 100000, "y": [0.0] * n_out, "math": math}
    for line in code.splitlines():
        m = re.match(r"\s*([vy]\[\d+\])\s*=\s*(.*);", line)
        if m:
            exec(f"{m.group(1)} = {_python(m.group(2))}", env)

    sched = tiered.schedule(code, units)
    slots, y = [math.nan] * sched.n_slots, [0.0] * n_out
    for level in range(1, sched.levels + 1):
        now = [s for s in sched.statements if s.level == level]
        written = {sched.slots[s.out[1]] for s in now if s.out[0] == "t"}
        read = {sched.slots[d] for s in now for d in s.deps}
        assert not written & read, f"level {level} reads a slot it writes"
        # Worst case for a race: evaluate the level in reverse unit order.
        for s in sorted(now, key=lambda s: -s.unit):
            rhs = re.sub(r"\{t(\d+)\}", lambda m: f"S[{sched.slots[int(m.group(1))]}]", s.rhs)
            rhs = re.sub(r"\{y(\d+)\}", r"Y[\1]", rhs)
            value = eval(_python(rhs), {"x": x, "S": slots, "Y": y, "math": math})
            if s.out[0] == "t":
                slots[sched.slots[s.out[1]]] = value
            else:
                y[s.out[1]] = value
    assert y == env["y"]
