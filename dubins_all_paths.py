import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle



def mod2pi(angle: float) -> float:
    return angle % (2.0 * math.pi)


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


@dataclass
class CandidatePath:
    name: str
    modes: List[str]
    params: Optional[Tuple[float, float, float]]  # normalized by rho
    total_length: Optional[float]
    feasible: bool



def dubins_LSL(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp = d + sa - sb
    p2 = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sa - sb)
    if p2 < -1e-9:
        return None
    p = math.sqrt(max(0.0, p2))
    tmp2 = math.atan2(cb - ca, tmp)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(beta - tmp2)
    return t, p, q


def dubins_RSR(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp = d - sa + sb
    p2 = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (-sa + sb)
    if p2 < -1e-9:
        return None
    p = math.sqrt(max(0.0, p2))
    tmp2 = math.atan2(ca - cb, tmp)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(-beta + tmp2)
    return t, p, q


def dubins_LSR(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p2 = -2.0 + d * d + 2.0 * c_ab + 2.0 * d * (sa + sb)
    if p2 < -1e-9:
        return None
    p = math.sqrt(max(0.0, p2))
    tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp)
    q = mod2pi(-beta + tmp)
    return t, p, q


def dubins_RSL(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p2 = -2.0 + d * d + 2.0 * c_ab - 2.0 * d * (sa + sb)
    if p2 < -1e-9:
        return None
    p = math.sqrt(max(0.0, p2))
    tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp)
    q = mod2pi(beta - tmp)
    return t, p, q


def dubins_RLR(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    tmp = max(-1.0, min(1.0, tmp))
    p = mod2pi(2.0 * math.pi - math.acos(tmp))
    t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + 0.5 * p)
    q = mod2pi(alpha - beta - t + p)
    return t, p, q


def dubins_LRL(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (-sa + sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    tmp = max(-1.0, min(1.0, tmp))
    p = mod2pi(2.0 * math.pi - math.acos(tmp))
    t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + 0.5 * p)
    q = mod2pi(mod2pi(beta) - alpha - t + p)
    return t, p, q

PATH_BUILDERS = {
    "LSL": (list("LSL"), dubins_LSL),
    "LSR": (list("LSR"), dubins_LSR),
    "RSL": (list("RSL"), dubins_RSL),
    "RSR": (list("RSR"), dubins_RSR),
    "LRL": (list("LRL"), dubins_LRL),
    "RLR": (list("RLR"), dubins_RLR),
}


def compute_all_dubins_paths(
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    rho: float,
) -> List[CandidatePath]:
    x0, y0, yaw0 = start
    x1, y1, yaw1 = goal

    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy)
    d = D / rho
    theta = math.atan2(dy, dx)
    alpha = mod2pi(yaw0 - theta)
    beta = mod2pi(yaw1 - theta)

    candidates: List[CandidatePath] = []
    for name, (modes, builder) in PATH_BUILDERS.items():
        params = builder(alpha, beta, d)
        if params is None:
            candidates.append(CandidatePath(name=name, modes=modes, params=None, total_length=None, feasible=False))
        else:
            total_length = rho * sum(params)
            candidates.append(CandidatePath(name=name, modes=modes, params=params, total_length=total_length, feasible=True))
    return candidates



def turning_circle_center(state: Tuple[float, float, float], mode: str, rho: float):
    x, y, yaw = state
    if mode == "L":
        return (x - rho * math.sin(yaw), y + rho * math.cos(yaw))
    if mode == "R":
        return (x + rho * math.sin(yaw), y - rho * math.cos(yaw))
    raise ValueError("Straight segment does not have a turning circle.")


def advance_state(state: Tuple[float, float, float], mode: str, amount: float, rho: float):
    x, y, yaw = state
    if mode == "S":
        return x + amount * math.cos(yaw), y + amount * math.sin(yaw), yaw

    if mode == "L":
        cx, cy = turning_circle_center(state, mode, rho)
        new_yaw = yaw + amount / rho
        return cx + rho * math.sin(new_yaw), cy - rho * math.cos(new_yaw), new_yaw

    if mode == "R":
        cx, cy = turning_circle_center(state, mode, rho)
        new_yaw = yaw - amount / rho
        return cx - rho * math.sin(new_yaw), cy + rho * math.cos(new_yaw), new_yaw

    raise ValueError(f"Unknown mode: {mode}")



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



def trace_candidate(
    start: Tuple[float, float, float],
    candidate: CandidatePath,
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
        physical_amount = param * rho
        if mode in ("L", "R"):
            center = turning_circle_center(state, mode, rho)
            turn_circles.append(center)
        state, xs, ys = sample_segment(state, mode, physical_amount, rho, ds)
        x_points.extend(xs)
        y_points.extend(ys)

    return {
        "x": x_points,
        "y": y_points,
        "final_state": state,
        "turn_circles": turn_circles,
    }


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



def plot_all_candidates(
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    rho: float,
    ds: float = 0.05,
    save_path: Optional[str] = None,
):
    candidates = compute_all_dubins_paths(start, goal, rho)
    feasible = [c for c in candidates if c.feasible]
    best_name = min(feasible, key=lambda c: c.total_length).name if feasible else None

    traces = {}
    all_x = [start[0], goal[0]]
    all_y = [start[1], goal[1]]

    for cand in candidates:
        trace = trace_candidate(start, cand, rho, ds)
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()

    ordered_names = ["LSL", "LSR", "RSL", "RSR", "LRL", "RLR"]
    for ax, name in zip(axes, ordered_names):
        cand = next(c for c in candidates if c.name == name)
        trace = traces[name]

        draw_pose(ax, start, "start", scale=0.6 * rho)
        draw_pose(ax, goal, "goal", scale=0.6 * rho)

        if trace is None:
            title = f"{name}\nnot feasible"
            ax.text(0.5, 0.5, "This path does not exist\nfor the chosen configuration", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        else:
            ax.plot(trace["x"], trace["y"], linewidth=2)
            for center in trace["turn_circles"]:
                ax.add_patch(Circle(center, rho, fill=False, linestyle="--", linewidth=1.5, alpha=0.8, edgecolor="0.25"))

            label = f"{name}\nlength = {cand.total_length:.3f}"
            if name == best_name:
                label += " (best)"
            title = label

            fx, fy, fyaw = trace["final_state"]
            pos_err = math.hypot(fx - goal[0], fy - goal[1])
            yaw_err = angle_diff(fyaw, goal[2])
            ax.text(
                0.02,
                0.02,
                f"end err = {pos_err:.2e}\nyaw err = {yaw_err:.2e}",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
            )

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("All 6 classical Dubins path candidates", fontsize=16)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Plot all 6 classical Dubins path candidates.")
    parser.add_argument("--x0", type=float, default=3.0, help="Start x")
    parser.add_argument("--y0", type=float, default=5.0, help="Start y")
    parser.add_argument("--yaw0_deg", type=float, default=70.0, help="Start heading in degrees")
    parser.add_argument("--x1", type=float, default=7.0, help="Goal x")
    parser.add_argument("--y1", type=float, default=1.0, help="Goal y")
    parser.add_argument("--yaw1_deg", type=float, default=45.0, help="Goal heading in degrees")
    parser.add_argument("--rho", type=float, default=1.5, help="Minimum turning radius")
    parser.add_argument("--ds", type=float, default=0.01, help="Sampling step for drawing")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
    args = parser.parse_args()

    start = (args.x0, args.y0, math.radians(args.yaw0_deg))
    goal = (args.x1, args.y1, math.radians(args.yaw1_deg))

    plot_all_candidates(start, goal, rho=args.rho, ds=args.ds, save_path=args.save)


if __name__ == "__main__":
    main()
