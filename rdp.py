import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def mod2pi(angle: float) -> float:
    return angle % (2.0 * math.pi)


@dataclass
class RelaxedCandidate:
    name: str
    modes: List[str]
    params: Optional[Tuple[float, ...]]
    total_length: Optional[float]
    feasible: bool


def turning_circle_center(state: Tuple[float, float, float], mode: str, rho: float):
    x, y, yaw = state
    if mode == "L":
        return (x - rho * math.sin(yaw), y + rho * math.cos(yaw))
    if mode == "R":
        return (x + rho * math.sin(yaw), y - rho * math.cos(yaw))
    raise ValueError("Only L/R have turning circles.")


def advance_state(state: Tuple[float, float, float], mode: str, amount: float, rho: float):
    x, y, yaw = state

    if mode == "S":
        return x + amount * math.cos(yaw), y + amount * math.sin(yaw), yaw

    if mode == "L":
        cx, cy = turning_circle_center(state, "L", rho)
        new_yaw = yaw + amount / rho
        return cx + rho * math.sin(new_yaw), cy - rho * math.cos(new_yaw), new_yaw

    if mode == "R":
        cx, cy = turning_circle_center(state, "R", rho)
        new_yaw = yaw - amount / rho
        return cx - rho * math.sin(new_yaw), cy + rho * math.cos(new_yaw), new_yaw

    raise ValueError(f"Unknown mode {mode}")


def sample_segment(
    state: Tuple[float, float, float],
    mode: str,
    amount: float,
    rho: float,
    ds: float,
):
    xs, ys = [], []
    travelled = 0.0
    cur = state

    while travelled < amount - 1e-12:
        step = min(ds, amount - travelled)
        cur = advance_state(cur, mode, step, rho)
        xs.append(cur[0])
        ys.append(cur[1])
        travelled += step

    return cur, xs, ys


def trace_relaxed_candidate(
    start: Tuple[float, float, float],
    candidate: RelaxedCandidate,
    rho: float,
    ds: float,
):
    if not candidate.feasible or candidate.params is None:
        return None

    x_points = [start[0]]
    y_points = [start[1]]
    turn_circles = []
    state = start

    for mode, param in zip(candidate.modes, candidate.params):
        amount = param if mode == "S" else param * rho

        if mode in ("L", "R"):
            turn_circles.append(turning_circle_center(state, mode, rho))

        state, xs, ys = sample_segment(state, mode, amount, rho, ds)
        x_points.extend(xs)
        y_points.extend(ys)

    return {
        "x": x_points,
        "y": y_points,
        "final_state": state,
        "turn_circles": turn_circles,
    }



def relaxed_LS(start: Tuple[float, float, float], goal_xy: Tuple[float, float], rho: float):
    x0, y0, yaw0 = start
    gx, gy = goal_xy

    cx, cy = turning_circle_center(start, "L", rho)
    vx = gx - cx
    vy = gy - cy
    d = math.hypot(vx, vy)

    if d < rho - 1e-9:
        return None

    phi = math.atan2(vy, vx)
    delta = math.acos(rho / d)

    theta = phi - delta
    tx = cx + rho * math.cos(theta)
    ty = cy + rho * math.sin(theta)

    start_radial = math.atan2(y0 - cy, x0 - cx)
    arc = mod2pi(theta - start_radial)
    straight = math.hypot(gx - tx, gy - ty)

    return arc, straight


def relaxed_RS(start: Tuple[float, float, float], goal_xy: Tuple[float, float], rho: float):
    x0, y0, yaw0 = start
    gx, gy = goal_xy

    cx, cy = turning_circle_center(start, "R", rho)
    vx = gx - cx
    vy = gy - cy
    d = math.hypot(vx, vy)

    if d < rho - 1e-9:
        return None

    phi = math.atan2(vy, vx)
    delta = math.acos(rho / d)

    theta = phi + delta
    tx = cx + rho * math.cos(theta)
    ty = cy + rho * math.sin(theta)

    start_radial = math.atan2(y0 - cy, x0 - cx)
    arc = mod2pi(start_radial - theta)
    straight = math.hypot(gx - tx, gy - ty)

    return arc, straight


def relaxed_LR(start: Tuple[float, float, float], goal_xy: Tuple[float, float], rho: float):
    x0, y0, yaw0 = start
    gx, gy = goal_xy

    c1x, c1y = turning_circle_center(start, "L", rho)

    dx = gx - c1x
    dy = gy - c1y
    D = math.hypot(dx, dy)

    if D > 3.0 * rho + 1e-9 or D < rho - 1e-9:
        return None

    phi = math.atan2(dy, dx)
    cos_delta = D / (4.0 * rho)
    if abs(cos_delta) > 1.0:
        return None
    delta = math.acos(max(-1.0, min(1.0, cos_delta)))


    psi = phi - delta

    c2x = c1x + 2.0 * rho * math.cos(psi)
    c2y = c1y + 2.0 * rho * math.sin(psi)


    tx = c1x + rho * math.cos(psi)
    ty = c1y + rho * math.sin(psi)

    a0 = math.atan2(y0 - c1y, x0 - c1x)
    at = math.atan2(ty - c1y, tx - c1x)
    arc1 = mod2pi(at - a0)

    bt = math.atan2(ty - c2y, tx - c2x)
    bg = math.atan2(gy - c2y, gx - c2x)
    arc2 = mod2pi(bt - bg)

    return arc1, arc2


def relaxed_RL(start: Tuple[float, float, float], goal_xy: Tuple[float, float], rho: float):
    x0, y0, yaw0 = start
    gx, gy = goal_xy

    c1x, c1y = turning_circle_center(start, "R", rho)

    dx = gx - c1x
    dy = gy - c1y
    D = math.hypot(dx, dy)

    if D > 3.0 * rho + 1e-9 or D < rho - 1e-9:
        return None

    phi = math.atan2(dy, dx)
    cos_delta = D / (4.0 * rho)
    if abs(cos_delta) > 1.0:
        return None
    delta = math.acos(max(-1.0, min(1.0, cos_delta)))

    psi = phi + delta

    c2x = c1x + 2.0 * rho * math.cos(psi)
    c2y = c1y + 2.0 * rho * math.sin(psi)

    tx = c1x + rho * math.cos(psi)
    ty = c1y + rho * math.sin(psi)

    a0 = math.atan2(y0 - c1y, x0 - c1x)
    at = math.atan2(ty - c1y, tx - c1x)
    arc1 = mod2pi(a0 - at)

    bt = math.atan2(ty - c2y, tx - c2x)
    bg = math.atan2(gy - c2y, gx - c2x)
    arc2 = mod2pi(bg - bt)

    return arc1, arc2


RELAXED_BUILDERS = {
    "LS": (["L", "S"], relaxed_LS),
    "RS": (["R", "S"], relaxed_RS),
    "LR": (["L", "R"], relaxed_LR),
    "RL": (["R", "L"], relaxed_RL),
}


def compute_all_relaxed_paths(
    start: Tuple[float, float, float],
    goal_xy: Tuple[float, float],
    rho: float,
) -> List[RelaxedCandidate]:
    candidates = []

    for name, (modes, builder) in RELAXED_BUILDERS.items():
        params = builder(start, goal_xy, rho)

        if params is None:
            candidates.append(
                RelaxedCandidate(
                    name=name,
                    modes=modes,
                    params=None,
                    total_length=None,
                    feasible=False,
                )
            )
        else:
            total_length = 0.0
            for mode, p in zip(modes, params):
                total_length += p if mode == "S" else p * rho

            candidates.append(
                RelaxedCandidate(
                    name=name,
                    modes=modes,
                    params=params,
                    total_length=total_length,
                    feasible=True,
                )
            )

    return candidates


def draw_pose(ax, pose: Tuple[float, float, float], label: str, scale: float):
    x, y, yaw = pose
    ax.plot(x, y, "o", markersize=5)
    ax.arrow(
        x,
        y,
        0.9 * scale * math.cos(yaw),
        0.9 * scale * math.sin(yaw),
        head_width=0.25 * scale,
        head_length=0.35 * scale,
        length_includes_head=True,
    )
    ax.text(x, y, f"  {label}", fontsize=9, va="bottom")


def plot_all_relaxed_candidates(
    start: Tuple[float, float, float],
    goal_xy: Tuple[float, float],
    rho: float,
    ds: float = 0.05,
    save_path: Optional[str] = None,
):
    candidates = compute_all_relaxed_paths(start, goal_xy, rho)
    feasible = [c for c in candidates if c.feasible]
    best_name = min(feasible, key=lambda c: c.total_length).name if feasible else None

    traces = {}
    gx, gy = goal_xy

    all_x = [start[0], gx]
    all_y = [start[1], gy]

    for cand in candidates:
        trace = trace_relaxed_candidate(start, cand, rho, ds)
        traces[cand.name] = trace

        if trace is not None:
            all_x.extend(trace["x"])
            all_y.extend(trace["y"])
            for cx, cy in trace["turn_circles"]:
                all_x.extend([cx - rho, cx + rho])
                all_y.extend([cy - rho, cy + rho])

    margin = 1.5 * rho
    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()

    ordered_names = ["LS", "RS", "LR", "RL"]

    for ax, name in zip(axes, ordered_names):
        cand = next(c for c in candidates if c.name == name)
        trace = traces[name]

        draw_pose(ax, start, "start", scale=0.6 * rho)
        ax.plot(gx, gy, "o", markersize=6)
        ax.text(gx, gy, "  goal", fontsize=9, va="bottom")

        if trace is None:
            ax.set_title(f"{name}\nnot feasible")
            ax.text(
                0.5,
                0.5,
                "This path does not exist\nfor the chosen configuration",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
        else:
            color = "green" if name == best_name else "blue"
            ax.plot(trace["x"], trace["y"], linewidth=2.5, color=color)

            for center in trace["turn_circles"]:
                ax.add_patch(
                    Circle(center, rho, fill=False, linestyle="--", linewidth=1.5, alpha=0.8, edgecolor="0.25")
                )

            fx, fy, fyaw = trace["final_state"]
            pos_err = math.hypot(fx - gx, fy - gy)

            ax.arrow(
                fx,
                fy,
                0.7 * rho * math.cos(fyaw),
                0.7 * rho * math.sin(fyaw),
                head_width=0.18 * rho,
                head_length=0.25 * rho,
                length_includes_head=True,
                color="orange",
            )

            label = f"{name}\nlength = {cand.total_length:.3f}"
            if name == best_name:
                label += " (best)"

            ax.set_title(label, fontsize=12, pad=10)

            ax.text(
                0.02,
                0.02,
                f"end err = {pos_err:.2e}",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("All relaxed Dubins path candidates: LS, RS, LR, RL", fontsize=16)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot all relaxed Dubins path candidates.")
    parser.add_argument("--x0", type=float, default=3.0, help="Start x")
    parser.add_argument("--y0", type=float, default=5.0, help="Start y")
    parser.add_argument("--yaw0_deg", type=float, default=140.0, help="Start heading in degrees")
    parser.add_argument("--x1", type=float, default=3.0, help="Goal x")
    parser.add_argument("--y1", type=float, default=1.0, help="Goal y")
    parser.add_argument("--rho", type=float, default=1.5, help="Minimum turning radius")
    parser.add_argument("--ds", type=float, default=0.01, help="Sampling step for drawing")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
    args = parser.parse_args()

    start = (args.x0, args.y0, math.radians(args.yaw0_deg))
    goal_xy = (args.x1, args.y1)

    plot_all_relaxed_candidates(start, goal_xy, rho=args.rho, ds=args.ds, save_path=args.save)


if __name__ == "__main__":
    main()