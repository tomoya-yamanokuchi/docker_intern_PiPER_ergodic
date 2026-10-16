"""One closed-loop episode on the peg_slide task (Scene B), shared by M1, M3 and M4.

A condition is a dict: statistic ("full", "reset", "exponential" or "frequency"), expanding (bool),
marks (a marks.VARIANTS entry) and slider_phase (bool: phi_tilde capped by the block's progress).
A scenario is a dict: block_offset (world units) and prefix (None, or "incomplete_push": a scripted
peg goes into the slot, pushes the block push_fraction of the way and lifts out; the controller
starts there with an empty statistic), and optionally warmup_T (s): a scripted peg wanders the first
stage (the master's way down, above the block) for that long, and its states enter the statistic as
visits before the controller starts at the master's start.
"""

from dataclasses import dataclass

import numpy as np

from experiments.sequential_ergodic import scene_b
from experiments.sequential_ergodic.basis import ergodic_metric, make_basis, values
from experiments.sequential_ergodic.controller import Plant, trajectory
from experiments.sequential_ergodic.demos import make_demos, master_path
from experiments.sequential_ergodic.detector import NegativeProgressDetector
from experiments.sequential_ergodic.marks import Mark, MarkingTarget, MarkParams
from experiments.sequential_ergodic.metrics import revisit_overlap, visits
from experiments.sequential_ergodic.phase import PhaseEstimator
from experiments.sequential_ergodic.statistic import (
    Statistic,
    exponential,
    frequency_forgetting,
    full_memory,
    hard_reset,
)
from experiments.sequential_ergodic.target import ExpandingTarget, PhaseConditionedTarget
from simulation.peg_slide import C0, CLEARANCE, PEG_HALF, SIZE, SLOT_FLOOR, TRAVEL, Slide

# The real slot's column, the way into it from above, in world units about the block's centre:
# marks there sit on the correct route.
HOLE_HALF, HOLE_TOP = CLEARANCE + 2 * PEG_HALF, 7.0

# The one parameter set of Scene B, shared by every milestone; a script adds only its sweeps.
CONFIG = {
    "K": [16, 16],
    "dt": 0.01,
    "u_max": 0.2,
    "T_max": 120.0,
    "seeds": 30,
    "start_jitter": 0.03,
    "demos": {"n": 8, "noise": 0.008},
    "master": {"samples": 101, "sigma_min": 0.03},
    "phase": {"window": 0.1, "alpha": 0.1, "beta": 0.005, "kappa_d": 0.05, "d_tube": 2.0},
    "target": {
        "n_components": 20,
        "beta0": 0.15,
        "delta": 0.08,
        "sigma_f": 0.04,
        "eps": 0.02,
        "eps_u": 0.05,
    },
    "lam": 0.1,
    # C3f: lambda_lo = lambda / ratio, lambda_hi = lambda * ratio, so the geometric mean is lambda.
    "frequency": {"ratio": 4.0, "kappa": 4.0},
    # B1 restarts when the target has moved as far as hard_reset_progress of phase from phi = 0.5
    # moves it. One delta (0.08) reset every 0.7 s on the wobble of phi_tilde alone in Scene A.
    "hard_reset_at_phi": 0.5,
    "hard_reset_progress": 0.25,
    "overlap_cell": 0.02,
    # Tuned in M2 on replays (results/M2): no false positive, most failures caught, shortest delay.
    "detector": {"K_c": 0.0, "h": 0.02, "delta_drop": 0.03, "T_ref": 5.0},
    "marks": {},  # MarkParams' defaults
    "slider_lead": 0.05,  # phase allowed beyond the block's progress
}


@dataclass(frozen=True)
class Job:
    name: str
    condition: dict
    scenario: dict
    lam: float | None
    seed: int
    cfg: dict
    keep_log: bool = False
    # s; > 0 also returns (t, p_k, c_hat_k) that often, for drawing the target and the statistic
    snapshot_every: float = 0.0


def statistic_for(
    kind: str, lam: float | None, basis, target: ExpandingTarget, cfg: dict
) -> Statistic:
    if kind == "full":
        return full_memory(basis)
    if kind == "exponential":
        return exponential(basis, lam)
    if kind == "frequency":
        r = cfg["frequency"]["ratio"]
        return frequency_forgetting(basis, lam / r, lam * r, cfg["frequency"]["kappa"])
    phi = cfg["hard_reset_at_phi"]
    moved = target.coefficients(phi + cfg["hard_reset_progress"]) - target.coefficients(phi)
    return hard_reset(basis, float(np.linalg.norm(moved)))


def setup(seed: int, cfg: dict) -> tuple[tuple, np.ndarray, list[np.ndarray]]:
    """The master (s, mu, sigma), the start (cube) and the demos of one seed."""
    rng = np.random.default_rng(seed)
    waypoints = scene_b.master_waypoints()
    d = cfg["demos"]
    demos = make_demos(waypoints, d["n"], d["noise"], rng)
    master = master_path(demos, cfg["master"]["samples"], cfg["master"]["sigma_min"])
    jitter = cfg["start_jitter"]
    return master, waypoints[0] + rng.uniform(-jitter, jitter, size=2), demos


def incomplete_push(world: Slide, start: np.ndarray, fraction: float) -> np.ndarray:  # (T, 2) cube
    """Down into the real slot, push the block fraction of TRAVEL, lift out above the block."""
    c, floor = world.c, SLOT_FLOOR + 0.05
    end = c + fraction * TRAVEL
    waypoints = np.array([start * SIZE, (c, 4.0), (c, floor), (end, floor), (end + 0.1, 5.0)])
    return scene_b.follow(world, waypoints, 0.02)[0]


def _phase_target(job: Job, basis, master: tuple, x0: np.ndarray, world: Slide):
    cfg, (s, mu, sigma) = job.cfg, master
    model = ExpandingTarget(basis, s, mu, sigma, **cfg["target"])
    estimator = PhaseEstimator(s, mu, sigma, **cfg["phase"])
    estimator.reset(x0)
    cap = None
    if job.condition.get("slider_phase"):

        def cap() -> float:
            return scene_b.slider_phase_cap(s, mu, world.c - world.home, cfg["slider_lead"])

    target = PhaseConditionedTarget(
        estimator, model, cfg["dt"], job.condition["expanding"], cap=cap
    )
    return model, target


def warm_up(job: Job, statistic: Statistic, basis, model: ExpandingTarget, x0: np.ndarray) -> None:
    """warmup_T seconds of a scripted peg in the first stage, added to the statistic as visits.

    It wanders at u_max between random points of the band x0_x +- 0.3, y 4.0 to x0_y (world), the
    master's way down well above the block's top at 3.0; nothing there touches the block.
    """
    cfg, T = job.cfg, job.scenario.get("warmup_T", 0.0)
    if not T:
        return
    rng = np.random.default_rng(5000 + job.seed)
    x0w = x0 * SIZE
    n_points = (
        round(T * cfg["u_max"] * SIZE / 0.6) + 2
    )  # legs average about 0.6 world units or more
    points = np.column_stack(
        [rng.uniform(x0w[0] - 0.3, x0w[0] + 0.3, n_points), rng.uniform(4.0, x0w[1], n_points)]
    )
    path, _ = scene_b.follow(Slide(), np.vstack([x0w, points]), cfg["u_max"] * cfg["dt"] * SIZE)
    steps = round(T / cfg["dt"])
    assert len(path) >= steps, "the scripted warm-up ran out of waypoints"
    p0 = model.coefficients(0.0)
    for z in path[:steps]:
        statistic.update(values(basis, z), cfg["dt"], p0)


def run_episode(job: Job) -> dict:
    """One episode, until the task is done or T_max."""
    cfg, cond = job.cfg, job.condition
    basis = make_basis(tuple(cfg["K"]))
    master, x0, _ = setup(job.seed, cfg)
    world = Slide(home=C0 + job.scenario["block_offset"])
    if job.scenario.get("prefix") == "incomplete_push":
        x0 = incomplete_push(world, x0, job.scenario["push_fraction"])[-1]
    model, phase_target = _phase_target(job, basis, master, x0, world)
    statistic = statistic_for(cond["statistic"], job.lam, basis, model, cfg)
    warm_up(job, statistic, basis, model, x0)
    target = MarkingTarget(
        phase_target,
        statistic,
        basis,
        NegativeProgressDetector(**cfg["detector"]),
        cond.get("marks"),
        MarkParams(**(cfg["marks"] | cond.get("mark_params", {}))),
    )
    plant = Plant(lambda z, d: scene_b.move(world, z, d), cfg["dt"], cfg["u_max"])
    log = []  # x, y, block centre, barrier, in the dead end, ergodic metric, phi_tilde
    every, snapshots = round(job.snapshot_every / cfg["dt"]), []
    for x, c, p in trajectory(basis, statistic, target, x0, plant):
        dead_end = scene_b.in_dead_end(world, x)
        log.append(
            (
                *x,
                world.c,
                world.barrier,
                dead_end,
                ergodic_metric(basis, c, p),
                phase_target.trace[-1][1],
            )
        )
        if every and (len(log) - 1) % every == 0:
            snapshots.append(((len(log) - 1) * cfg["dt"], p.tolist(), c.tolist()))
        if scene_b.done(world, x) or len(log) * cfg["dt"] >= cfg["T_max"]:
            break
    return _result(job, np.array(log, dtype=float), world, target, snapshots if every else None)


def _result(
    job: Job, log: np.ndarray, world: Slide, target: MarkingTarget, snapshots: list | None
) -> dict:
    """The episode's metrics, marks and firings; its log, phase trace and snapshots when asked."""
    cfg = job.cfg
    out = {"name": job.name, "lam": job.lam, "seed": job.seed}
    out |= episode_metrics(log, scene_b.done(world, log[-1, :2]), cfg)
    out |= mark_metrics(target.marks, log, cfg)
    out["firings"] = [(f.t_peak, f.t_det, f.branch) for f in target.firings]
    # Every 10th point of each mark: enough for a heatmap over seeds, light enough to keep for all.
    out["marks"] = [
        (m.points[::10].tolist(), (10 * m.weights[::10]).tolist(), m.mass, m.phi, m.t_fire)
        for m in target.marks
    ]
    if job.keep_log:
        out["log"] = log.tolist()
        out["trace"] = np.array(target.inner.trace).tolist()
    if snapshots is not None:
        out["snapshots"] = snapshots
    return out


def episode_metrics(log: np.ndarray, success: bool, cfg: dict) -> dict:
    dt, path = cfg["dt"], log[:, :2]
    runs = visits(log[:, 4] > 0)
    t_success = len(log) * dt if success else None
    first_exit = runs[0].stop * dt if runs and runs[0].stop < len(log) else None
    return {
        "success": bool(success),
        "time_to_success": t_success,
        "entries": len(runs),
        "time_in_dead_end": sum(r.stop - r.start for r in runs) * dt,
        "dwells": [(r.stop - r.start) * dt for r in runs],
        "entry_times": [r.start * dt for r in runs],
        "first_exit": first_exit,
        "recovery": t_success - first_exit if success and first_exit is not None else None,
        "barrier_removed": bool(log[-1, 3] == 0),
        "final_block_travel": float(log[-1, 2] - log[0, 2]),
        "final_metric": float(log[-1, 5]),
        "overlaps": revisit_overlap(path, runs, cfg["overlap_cell"]),
    }


def in_hole_column(points: np.ndarray, c: float) -> np.ndarray:  # (P, 2) cube -> (P,)
    x, y = points[:, 0] * SIZE, points[:, 1] * SIZE
    return (np.abs(x - c) <= HOLE_HALF) & (y >= SLOT_FLOOR) & (y <= HOLE_TOP)


def mark_metrics(marks: list[Mark], log: np.ndarray, cfg: dict) -> dict:
    """Marks; the share of their mass on the way into the real slot (false) and in the dead end;
    the dead end after the first mark. A mark's points are the samples up to its firing."""
    dt = cfg["dt"]
    if not marks:
        return {
            "n_marks": 0,
            "mark_mass": 0.0,
            "false_mark_rate": None,
            "marks_at_hole": 0,
            "dead_end_mark_share": None,
            "dead_end_after_first_mark": None,
            "entries_after_first_mark": None,
        }
    on_hole, in_dead, total, at_hole = 0.0, 0.0, 0.0, 0
    for m in marks:
        c = log[min(round(m.t_fire / dt), len(log) - 1), 2]  # the block where it was at firing
        share = float(m.weights @ in_hole_column(m.points, c))
        i_det = min(round(m.t_fire / dt), len(log) - 1)
        in_dead += m.mass * float(m.weights @ log[i_det - len(m.points) + 1 : i_det + 1, 4])
        on_hole, total, at_hole = on_hole + m.mass * share, total + m.mass, at_hole + (share > 0.5)
    first = round(marks[0].t_fire / dt)
    return {
        "n_marks": len(marks),
        "mark_mass": total,
        "false_mark_rate": on_hole / total if total > 0 else None,
        "marks_at_hole": at_hole,
        "dead_end_mark_share": in_dead / total if total > 0 else None,
        "dead_end_after_first_mark": float(log[first:, 4].sum() * dt),
        "entries_after_first_mark": len(visits(log[first:, 4] > 0)),
    }
