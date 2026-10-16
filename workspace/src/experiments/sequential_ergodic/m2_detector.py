"""M2: the phase estimate and the negative-progress detector, offline on replays, then closed-loop.

Replays per seed, all at u_max through the peg_slide environment (the demos positionally):
- demo: each of the seed's 8 demonstrations; any firing is a false positive;
- right_miss: block 0.7 short; onto the block's top, down beside it, along to the barrier, wander
  there, back out, into the real slot and push to the goal;
- stall: the same to the barrier, then small, mostly vertical motion in place for 15 s;
- left_miss: block 0.7 long; down on the block's left, against its side, wander, back out over it,
  into the slot, push to the goal.
A failure is detected by a firing between its dead-end entry and its arrival in the real slot
(the end of the replay for stall); any other firing is a false positive. The delay is from the
dead-end entry. The detector's (K_c, h, delta_drop, T_ref) are chosen on a grid: no false positive
on any replay, then the most failures detected, then the shortest mean delay.

Writes results/M2/: config.json, tuning.json, closed_loop.json, SUMMARY numbers and figures.
"""

import itertools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from experiments.sequential_ergodic import scene_b
from experiments.sequential_ergodic.demos import resample
from experiments.sequential_ergodic.detector import Firing, NegativeProgressDetector
from experiments.sequential_ergodic.episode import CONFIG, Job, setup
from experiments.sequential_ergodic.phase import PhaseEstimator
from experiments.sequential_ergodic.report import PALETTE, draw_world, row, run_all
from simulation.peg_slide import C0, SIZE, Slide, in_slot, missed

OUT = Path(__file__).parent / "results" / "M2"
GRID = {
    "K_c": [0.0, 0.002, 0.005, 0.01],
    "h": [0.02, 0.05, 0.1, 0.2],
    "delta_drop": [0.03, 0.05, 0.1],
    "T_ref": [2.0, 5.0],
}
FAILURES = ("right_miss", "stall", "left_miss")


def _failure_waypoints(kind: str, x0: np.ndarray, rng: np.random.Generator) -> list:
    """World waypoints of one scripted failure."""
    start = tuple(x0 * SIZE)
    if kind == "left_miss":
        to_dead_end = [start, (2.3, 3.15), (2.0, 3.15), (2.0, 1.62), (2.3, 1.62)]
        wander = [(2.0, 2.4), (2.0, 1.5), (2.05, 1.62)]
        back = [(2.0, 3.3), (3.0, 3.3), (3.0, 1.55), (8.6, 1.55)]
        return to_dead_end + wander + back
    to_dead_end = [start, (2.3, 3.15), (2.65, 3.15), (2.65, 1.62), (3.85, 1.62)]
    if kind == "stall":
        wiggle = np.column_stack([rng.uniform(3.75, 3.85, 100), rng.uniform(1.45, 2.2, 100)])
        return to_dead_end + [tuple(p) for p in wiggle]
    wander = [(3.85, 2.3), (3.85, 1.4), (3.85, 2.0)]
    back = [(2.65, 1.7), (2.65, 3.3), (1.6, 3.3), (1.6, 1.55), (8.5, 1.55)]
    return to_dead_end + wander + back


def failure_replay(kind: str, seed: int, cfg: dict) -> dict:
    _, x0, _ = setup(seed, cfg)
    rng = np.random.default_rng(1000 + seed)
    world = Slide(home=C0 + (0.7 if kind == "left_miss" else -0.7))
    waypoints = _failure_waypoints(kind, x0, rng)
    step = cfg["u_max"] * cfg["dt"] * SIZE
    path, cs = scene_b.follow(world, np.array(waypoints), step)
    w = path * SIZE
    world_then = [
        Slide(world.home, c, True) for c in cs
    ]  # the barrier only matters before the slot
    dead = np.array([missed(wt, s) for wt, s in zip(world_then, w, strict=True)])
    entry = int(np.argmax(dead))
    slot = np.array([in_slot(c, *s) for c, s in zip(cs, w, strict=True)])
    arrived = entry + int(np.argmax(slot[entry:])) if slot[entry:].any() else len(path)
    return {
        "kind": kind,
        "seed": seed,
        "path": path,
        "c": cs,
        "home": world.home,
        "entry": entry,
        "end": arrived,
    }


def demo_replays(seed: int, cfg: dict) -> list[dict]:
    _, _, demos = setup(seed, cfg)
    step = cfg["u_max"] * cfg["dt"]
    out = []
    for d in demos:
        length = np.linalg.norm(np.diff(d, axis=0), axis=1).sum()
        path = resample(d, int(length / step) + 1)
        out.append(
            {"kind": "demo", "seed": seed, "path": path, "c": None, "entry": None, "end": None}
        )
    return out


def phase_trace(replay: dict, cfg: dict) -> np.ndarray:  # (T, 4) phi_hat, phi_tilde, d_perp, capped
    (s, mu, sigma), _, _ = setup(replay["seed"], cfg)
    est = PhaseEstimator(s, mu, sigma, **cfg["phase"])
    est.reset(replay["path"][0])
    rows = []
    for i, x in enumerate(replay["path"]):
        phi_hat, phi_tilde, d = est.update(x, cfg["dt"])
        capped = phi_tilde
        if replay["c"] is not None:
            pushed = replay["c"][i] - replay["home"]
            capped = min(phi_tilde, scene_b.slider_phase_cap(s, mu, pushed, cfg["slider_lead"]))
        rows.append((phi_hat, phi_tilde, d, capped))
    return np.array(rows)


def detect(phi: np.ndarray, path: np.ndarray, params: dict, cfg: dict) -> list[Firing]:
    det = NegativeProgressDetector(**params)
    per = round(det.period / cfg["dt"])
    out = []
    for n in range(per, len(phi), per):
        speed = np.linalg.norm(np.diff(path[n - per : n + 1], axis=0), axis=1).sum() / (
            per * cfg["dt"]
        )
        f = det.step(n * cfg["dt"], phi[n], speed)
        if f is not None:
            out.append(f)
    return out


def score(replay: dict, firings: list[Firing], dt: float) -> tuple[int, bool, float | None]:
    """(false positives, detected, delay from the dead-end entry)."""
    if replay["kind"] == "demo":
        return len(firings), False, None
    lo, hi = replay["entry"] * dt, replay["end"] * dt
    inside = [f for f in firings if lo <= f.t_det <= hi]
    return len(firings) - len(inside), bool(inside), (inside[0].t_det - lo) if inside else None


def _evaluate(args: tuple[dict, list[tuple[dict, np.ndarray]], dict]) -> dict:
    params, prepared, cfg = args
    fp, detected, delays = 0, {k: 0 for k in FAILURES}, []
    for replay, trace in prepared:
        f, d, delay = score(replay, detect(trace[:, 1], replay["path"], params, cfg), cfg["dt"])
        fp += f
        if d:
            detected[replay["kind"]] += 1
            delays.append(delay)
    return {
        "params": params,
        "false_positives": fp,
        "detected": detected,
        "mean_delay": float(np.mean(delays)) if delays else None,
    }


def tune(prepared: list, cfg: dict) -> list[dict]:
    combos = [dict(zip(GRID, v, strict=True)) for v in itertools.product(*GRID.values())]
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as pool:
        results = list(pool.map(_evaluate, [(c, prepared, cfg) for c in combos]))
    return sorted(
        results,
        key=lambda r: (r["false_positives"], -sum(r["detected"].values()), r["mean_delay"] or 1e9),
    )


def draw_replays(prepared: list, params: dict, cfg: dict) -> Figure:
    fig = Figure(figsize=(15, 9), layout="constrained")
    fig.suptitle(
        r"Seed 0 replays: path, then $\hat\varphi$, $\tilde\varphi$ and the slider-capped $\tilde\varphi$, then $d_\perp$"
    )
    axes = fig.subplots(3, 3, height_ratios=[1.2, 1, 0.6])
    (_, mu, _), _, _ = setup(0, cfg)
    for j, kind in enumerate(FAILURES):
        replay, trace = next((r, t) for r, t in prepared if r["kind"] == kind and r["seed"] == 0)
        firings = detect(trace[:, 1], replay["path"], params, cfg)
        t = cfg["dt"] * np.arange(len(trace))
        world, phase, dist = axes[:, j]
        draw_world(world, replay["c"][-1], False, mu)
        world.plot(*(replay["path"] * SIZE).T, "-k", lw=0.7)
        for f in firings:
            seg = (t >= f.t_peak) & (t <= f.t_det)
            world.plot(*(replay["path"][seg] * SIZE).T, "-", color="#d03b3b", lw=3, alpha=0.7)
        world.set_title(f"{kind}: {len(firings)} firings", fontsize=10)
        for ax in (phase, dist):
            ax.axvspan(replay["entry"] * cfg["dt"], replay["end"] * cfg["dt"], color="0.9", lw=0)
            for f in firings:
                ax.axvline(f.t_det, color="#d03b3b", lw=1)
        for col, label, colour in (
            (0, r"$\hat\varphi$", PALETTE[0]),
            (1, r"$\tilde\varphi$", PALETTE[1]),
            (3, "slider-capped", PALETTE[2]),
        ):
            phase.plot(t, trace[:, col], color=colour, lw=1.2, label=label)
        phase.set(ylabel="phase", ylim=(-0.02, 1.02))
        dist.plot(t, trace[:, 2], color="0.3", lw=1)
        dist.axhline(cfg["phase"]["d_tube"], color="0.6", ls=":")
        dist.set(xlabel="time (s)", ylabel=r"$d_\perp$ (tube widths)")
    axes[1, 0].legend(frameon=False, fontsize=8)
    return fig


def rise_in_dead_end(prepared: list) -> dict:
    """Mean rise of each phase signal over the dead-end stay, from entry to its highest there."""
    out = {}
    for kind in FAILURES:
        rises = []
        for replay, trace in prepared:
            if replay["kind"] != kind:
                continue
            stay = trace[replay["entry"] : replay["end"]]
            rises.append(stay.max(axis=0) - trace[replay["entry"]])
        r = np.array(rises)
        out[kind] = {
            "phi_hat": float(r[:, 0].mean()),
            "phi_tilde": float(r[:, 1].mean()),
            "capped": float(r[:, 3].mean()),
        }
    return out


def closed_loop(params: dict, cfg: dict) -> dict:
    """C3 at the main lambda with the tuned detector: what it marks (nothing is applied)."""
    tuned = cfg | {"detector": params}
    cond = {"statistic": "exponential", "expanding": True}
    eps = run_all(
        [Job("C3", cond, {"block_offset": -0.7}, cfg["lam"], s, tuned) for s in range(cfg["seeds"])]
    )
    firings = [f for e in eps for f in e["firings"]]
    return {
        "row": row(eps, np.random.default_rng(0)),
        "firings_per_episode": len(firings) / len(eps),
        "branches": {b: sum(f[2] == b for f in firings) for b in ("cusum", "stall")},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    replays = [failure_replay(k, s, cfg) for s in range(cfg["seeds"]) for k in FAILURES]
    replays += [d for s in range(cfg["seeds"]) for d in demo_replays(s, cfg)]
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as pool:
        traces = list(pool.map(phase_trace, replays, [cfg] * len(replays)))
    prepared = list(zip(replays, traces, strict=True))
    ranked = tune(prepared, cfg)
    best = ranked[0]["params"]
    default = _evaluate(({}, prepared, cfg))  # the detector's own defaults
    (OUT / "config.json").write_text(json.dumps({"config": cfg, "grid": GRID}, indent=2))
    (OUT / "tuning.json").write_text(
        json.dumps({"default": default, "ranked": ranked[:10]}, indent=2)
    )
    draw_replays(prepared, best, cfg).savefig(OUT / "replays_seed0.png", dpi=150)
    rises = rise_in_dead_end(prepared)
    loop = closed_loop(best, cfg)
    (OUT / "closed_loop.json").write_text(
        json.dumps({"tuned": best, "rises": rises, "C3": loop}, indent=2)
    )
    print(json.dumps({"default": default, "best": ranked[0], "rises": rises}, indent=2))
    print(json.dumps({k: v for k, v in loop.items() if k != "row"}, indent=2))
    print({k: np.round(v, 3).tolist() for k, v in loop["row"].items() if k != "n"})


if __name__ == "__main__":
    main()
