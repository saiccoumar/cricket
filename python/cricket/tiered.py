"""Cooperative (warp/block) CUDA emission for cricket's straight-line traces.

A trace is a list of assignments ``v[i] = ...;`` / ``y[i] = ...;`` over inputs ``x[...]``.
:func:`schedule` levels the dataflow graph (a statement's level is one more than its deepest
input) and assigns each level's statements to ``units`` workers by cost, longest first.
:func:`emit` writes the schedule as straight-line CUDA, one ``switch (rank)`` per level
followed by a barrier. Values that cross levels live in a shared-memory slot, reused once
their last reader's level has passed, so the emitted code keeps the thread tier's property
of no runtime indexing into register arrays.

Two mappings, ``mode``:

``"warp"`` (use this one)
    32 configurations per block, one per lane, and the block's warps split the statements.
    ``rank`` is the warp (``threadIdx.x >> 5``); every lane of a warp runs the same statement
    for its own configuration, so warps are the parallel unit and lanes stay convergent.
    Measured on an RTX A5000: 1.5-2.4x over one config per thread for G1 traces at 16-1024
    configurations; slower from about 4k configurations, where the thread tier fills the GPU.
``"lane"``
    One configuration per group; the group's lanes split the statements (GLASS's warp/block
    scopes). Lanes of one warp then run *different* statements, which SIMT serializes: 3-5x
    slower than the thread tier on every trace measured. Kept for reference.

Results equal the thread tier's up to FMA contraction (bit-identical under ``-fmad=false``):
each statement is emitted unchanged, only moved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ASSIGN = re.compile(r"^\s*([vy])\[(\d+)\]\s*=\s*(.*);\s*$")
_REF = re.compile(r"\b([vy])\[(\d+)\]")
_COST = re.compile(r"[-+*/]|\b[a-z]+f\(")


@dataclass
class Statement:
    out: tuple[str, int]  # ("t", ssa id) for a temporary, ("y", index) for an output
    rhs: str  # with v/y reads rewritten to "{t<id>}" / "{y<index>}" placeholders
    deps: list[int]  # ssa ids of temporaries read
    ydeps: list[int]  # output indices read (CppADCodeGen occasionally reuses outputs)
    level: int = 0
    cost: int = 1
    unit: int = 0


@dataclass
class Schedule:
    statements: list[Statement]
    levels: int
    units: int
    slots: dict[int, int]  # ssa temporary -> shared-memory slot
    n_slots: int


def parse(code: str) -> list[Statement]:
    """Split a trace into statements in SSA form (CppADCodeGen reuses ``v`` slots)."""
    current: dict[int, int] = {}
    statements: list[Statement] = []
    next_id = 0
    for line in code.splitlines():
        m = _ASSIGN.match(line)
        if not m:
            if line.strip() and not line.strip().startswith("//"):
                raise ValueError(f"unexpected trace line: {line!r}")
            continue
        kind, idx, rhs = m.group(1), int(m.group(2)), m.group(3)
        deps, ydeps = [], []

        def sub(r):
            k, i = r.group(1), int(r.group(2))
            if k == "v":
                deps.append(current[i])
                return "{t%d}" % current[i]
            ydeps.append(i)
            return "{y%d}" % i

        rhs = _REF.sub(sub, rhs)
        if kind == "v":
            current[idx] = next_id
            out = ("t", next_id)
            next_id += 1
        else:
            out = ("y", idx)
        statements.append(Statement(out, rhs, deps, ydeps, cost=max(1, len(_COST.findall(rhs)))))
    return statements


def schedule(code: str, units: int) -> Schedule:
    stmts = parse(code)
    level_of_t: dict[int, int] = {}
    level_of_y: dict[int, int] = {}
    for s in stmts:
        s.level = 1 + max([level_of_t[d] for d in s.deps] + [level_of_y[d] for d in s.ydeps] + [0])
        (level_of_t if s.out[0] == "t" else level_of_y)[s.out[1]] = s.level
    levels = max((s.level for s in stmts), default=0)

    # Longest-processing-time assignment per level.
    for lvl in range(1, levels + 1):
        load = [0] * units
        for s in sorted((s for s in stmts if s.level == lvl), key=lambda s: -s.cost):
            u = min(range(units), key=load.__getitem__)
            s.unit = u
            load[u] += s.cost

    # Every temporary is read at a later level (same-level statements are independent), so it
    # lives in shared memory from its level until its last reader's level; slots are reused
    # by interval colouring over levels.
    last_use: dict[int, int] = {}
    for s in stmts:
        for d in s.deps:
            last_use[d] = max(last_use.get(d, 0), s.level)
    temps = sorted((s for s in stmts if s.out[0] == "t"), key=lambda s: s.level)
    free: list[int] = []
    busy: list[tuple[int, int]] = []  # (last level, slot)
    slots: dict[int, int] = {}
    n_slots = 0
    for s in temps:
        # A slot frees once its last reader's level is behind a barrier (end < this level);
        # reusing it within the reader's own level would race with that read.
        free += [slot for end, slot in busy if end < s.level]
        busy = [b for b in busy if b[0] >= s.level]
        if free:
            slot = free.pop()
        else:
            slot, n_slots = n_slots, n_slots + 1
        slots[s.out[1]] = slot
        busy.append((last_use.get(s.out[1], s.level), slot))
    return Schedule(stmts, levels, units, slots, n_slots)


def emit(sched: Schedule, name: str, mode: str, qualifiers: str = "__device__ __forceinline__",
         y_stride: str | None = None) -> str:
    """CUDA for ``name(int rank, const float* x, float* y, float* s)``.

    ``s`` holds the cross-level values: ``smem_floats(sched, mode)`` floats per group, in shared
    memory or, when that is too small, global memory. In ``"warp"`` mode ``x`` and ``y`` are the
    calling lane's own configuration. ``y_stride`` (a C expression) spaces output ``i`` at
    ``y[i * y_stride]``, e.g. the batch size for coalesced structure-of-arrays outputs.
    """
    if mode not in ("lane", "warp"):
        raise ValueError(mode)
    barrier = "__syncthreads()" if mode == "warp" else "group_sync()"

    def slot(t: int) -> str:
        k = sched.slots[t]
        return f"s[{k} * 32 + lane]" if mode == "warp" else f"s[{k}]"

    def out(i: int) -> str:
        return f"y[{i} * {y_stride}]" if y_stride else f"y[{i}]"

    def render(s: Statement) -> str:
        rhs = re.sub(r"\{t(\d+)\}", lambda m: slot(int(m.group(1))), s.rhs)
        rhs = re.sub(r"\{y(\d+)\}", lambda m: out(int(m.group(1))), rhs)
        lhs = slot(s.out[1]) if s.out[0] == "t" else out(s.out[1])
        return f"{lhs} = {rhs};"

    body = []
    for lvl in range(1, sched.levels + 1):
        by_unit: dict[int, list[str]] = {}
        for s in sched.statements:
            if s.level == lvl:
                by_unit.setdefault(s.unit, []).append(render(s))
        body.append("    switch (rank) {")
        for u in sorted(by_unit):
            body.append(f"    case {u}: " + " ".join(by_unit[u]) + " break;")
        body.append("    default: break;\n    }")
        body.append(f"    {barrier};")
    lane = "    const int lane = threadIdx.x & 31;\n" if mode == "warp" else ""
    sync = "" if mode == "warp" else "template <class Sync>\n"
    params = "int rank, const float* x, float* y, float* s" + (", Sync group_sync" if mode == "lane" else "")
    return (f"{sync}{qualifiers} void {name}({params})\n{{\n{lane}    (void)x;\n"
            + "\n".join(body) + "\n}\n")


def smem_floats(sched: Schedule, mode: str) -> int:
    return max(sched.n_slots, 1) * (32 if mode == "warp" else 1)
