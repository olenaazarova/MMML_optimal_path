import math
import numpy as np
import matplotlib.pyplot as plt

def mod2pi(theta):
    return theta % (2.0 * math.pi)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# 6 types of Dubins path
# Returns (t, p, q) or None
# t, p, q are the lengths of the segments in normalized space

def LSL(alpha, beta, d):
    tmp = d + math.sin(alpha) - math.sin(beta)
    p_squared = 2 + d**2 - 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))
    if p_squared < -1e-9:
        return None
    p = math.sqrt(max(0.0, p_squared))
    tmp2 = math.atan2(math.cos(beta) - math.cos(alpha), tmp)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(beta - tmp2)
    return t, p, q


def RSR(alpha, beta, d):
    p_squared = (
        2 + d**2
        - 2 * math.cos(alpha - beta)
        + 2 * d * (-math.sin(alpha) + math.sin(beta))
    )
    if p_squared < -1e-9:
        return None

    p = math.sqrt(max(0.0, p_squared))
    tmp = math.atan2(
        math.cos(alpha) - math.cos(beta),
        d - math.sin(alpha) + math.sin(beta)
    )
    t = mod2pi(alpha - tmp)
    q = mod2pi(-beta + tmp)
    return t, p, q


def LSR(alpha, beta, d):
    p_squared = (
        -2 + d**2 + 2 * math.cos(alpha - beta)
        + 2 * d * (math.sin(alpha) + math.sin(beta))
    )
    if p_squared < -1e-9:
        return None

    p = math.sqrt(max(0.0, p_squared))
    tmp = math.atan2(
        -math.cos(alpha) - math.cos(beta),
        d + math.sin(alpha) + math.sin(beta)
    ) - math.atan2(-2.0, p)

    t = mod2pi(-alpha + tmp)
    q = mod2pi(-beta + tmp)
    return t, p, q


def RSL(alpha, beta, d):
    p_squared = (
        -2 + d**2 + 2 * math.cos(alpha - beta)
        - 2 * d * (math.sin(alpha) + math.sin(beta))
    )
    if p_squared < -1e-9:
        return None

    p = math.sqrt(max(0.0, p_squared))
    tmp = math.atan2(
        math.cos(alpha) + math.cos(beta),
        d - math.sin(alpha) - math.sin(beta)
    ) - math.atan2(2.0, p)

    t = mod2pi(alpha - tmp)
    q = mod2pi(beta - tmp)
    return t, p, q


def RLR(alpha, beta, d):
    tmp = (
        6.0 - d**2 + 2 * math.cos(alpha - beta)
        + 2 * d * (math.sin(alpha) - math.sin(beta))
    ) / 8.0
    if abs(tmp) > 1.0:
        return None

    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(
        alpha
        - math.atan2(
            math.cos(alpha) - math.cos(beta),
            d - math.sin(alpha) + math.sin(beta)
        )
        + p / 2.0
    )
    q = mod2pi(alpha - beta - t + p)
    return t, p, q


def LRL(alpha, beta, d):
    tmp = (
        6.0 - d**2 + 2 * math.cos(alpha - beta)
        + 2 * d * (-math.sin(alpha) + math.sin(beta))
    ) / 8.0
    if abs(tmp) > 1.0:
        return None

    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(
        -alpha
        - math.atan2(
            math.cos(alpha) - math.cos(beta),
            d + math.sin(alpha) - math.sin(beta)
        )
        + p / 2.0
    )
    q = mod2pi(beta - alpha - t + p)
    return t, p, q


PATH_TYPES = {
    "LSL": LSL,
    "RSR": RSR,
    "LSR": LSR,
    "RSL": RSL,
    "RLR": RLR,
    "LRL": LRL,
}


def dubins_shortest_path(start, goal, rho):
    x0, y0, th0 = start
    x1, y1, th1 = goal

    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy)
    d = D / rho

    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(th0 - theta)
    beta = mod2pi(th1 - theta)

    best = None

    for path_name, path_func in PATH_TYPES.items():
        result = path_func(alpha, beta, d)
        if result is None:
            continue

        t, p, q = result
        cost = (t + p + q) * rho

        candidate = {
            "type": path_name,
            "segments": (t, p, q),
            "cost": cost,
            "rho": rho,
            "start": start,
            "goal": goal,
        }

        if best is None or candidate["cost"] < best["cost"]:
            best = candidate

    return best


def get_all_dubins_paths(start, goal, rho):
    x0, y0, th0 = start
    x1, y1, th1 = goal

    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy)
    d = D / rho

    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(th0 - theta)
    beta = mod2pi(th1 - theta)

    paths = []

    for path_name, path_func in PATH_TYPES.items():
        result = path_func(alpha, beta, d)
        if result is None:
            continue

        t, p, q = result
        cost = (t + p + q) * rho

        paths.append({
            "type": path_name,
            "segments": (t, p, q),
            "cost": cost,
            "rho": rho,
            "start": start,
            "goal": goal,
        })

    return paths


def sample_segment(x, y, yaw, segment_type, length, rho, step_size):
    dist = length * rho
    eps = 1e-10

    xs, ys, yaws = [x], [y], [yaw]

    while dist > eps:
        ds = min(step_size, dist)

        if segment_type == 'S':
            x += ds * math.cos(yaw)
            y += ds * math.sin(yaw)

        elif segment_type == 'L':
            dtheta = ds / rho
            x += rho * (math.sin(yaw + dtheta) - math.sin(yaw))
            y += -rho * (math.cos(yaw + dtheta) - math.cos(yaw))
            yaw += dtheta

        elif segment_type == 'R':
            dtheta = ds / rho
            x += rho * (math.sin(yaw) - math.sin(yaw - dtheta))
            y += rho * (math.cos(yaw - dtheta) - math.cos(yaw))
            yaw -= dtheta

        yaw = mod2pi(yaw)
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)

        dist -= ds

    return x, y, yaw, xs, ys, yaws


def generate_dubins_segments(path, step_size=0.05):
    x, y, yaw = path["start"]
    rho = path["rho"]
    path_type = path["type"]
    t, p, q = path["segments"]

    segments = []
    segment_types = list(path_type)
    segment_lengths = [t, p, q]

    for seg_type, seg_len in zip(segment_types, segment_lengths):
        x, y, yaw, xs, ys, yaws = sample_segment(x, y, yaw, seg_type, seg_len, rho, step_size)
        segments.append({
            "type": seg_type,
            "x": xs,
            "y": ys,
            "yaw": yaws
        })

    return segments


def generate_dubins_path(path, step_size=0.1):
    x, y, yaw = path["start"]
    rho = path["rho"]
    path_type = path["type"]
    t, p, q = path["segments"]

    segment_types = list(path_type)

    full_x = [x]
    full_y = [y]
    full_yaw = [yaw]

    for seg_type, seg_length in zip(segment_types, [t, p, q]):
        x, y, yaw, xs, ys, yaws = sample_segment(x, y, yaw, seg_type, seg_length, rho, step_size)
        full_x.extend(xs[1:])
        full_y.extend(ys[1:])
        full_yaw.extend(yaws[1:])

    return np.array(full_x), np.array(full_y), np.array(full_yaw)


def plot_arrow(x, y, yaw, length=0.7, color='red'):
    plt.arrow(
        x, y,
        length * math.cos(yaw),
        length * math.sin(yaw),
        head_width=0.22,
        head_length=0.28,
        fc=color,
        ec=color,
        length_includes_head=True
    )


def plot_all_dubins_paths(start, goal, rho, step_size=0.05):
    paths = get_all_dubins_paths(start, goal, rho)

    if not paths:
        print("Жодного допустимого Dubins path не знайдено.")
        return

    best_path = min(paths, key=lambda p: p["cost"])

    plt.figure(figsize=(10, 8))

    for path in paths:
        segments = generate_dubins_segments(path, step_size=step_size)

        for seg in segments:
            linestyle = '--' if seg["type"] in ['L', 'R'] else '-'
            plt.plot(
                seg["x"],
                seg["y"],
                color='blue',
                linestyle=linestyle,
                linewidth=1.8,
                alpha=0.75
            )

    best_segments = generate_dubins_segments(best_path, step_size=step_size)

    for seg in best_segments:
        linestyle = '--' if seg["type"] in ['L', 'R'] else '-'
        plt.plot(
            seg["x"],
            seg["y"],
            color='green',
            linestyle=linestyle,
            linewidth=3.0
        )

    plt.scatter([start[0]], [start[1]], color='orange', s=90, label='Start', zorder=5)
    plt.scatter([goal[0]], [goal[1]], color='purple', s=90, label='Goal', zorder=5)

    plot_arrow(*start, color='orange')
    plot_arrow(*goal, color='purple')

    plt.title(f"All Dubins Paths (best: {best_path['type']}, length={best_path['cost']:.3f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Усі допустимі шляхи:")
    for p in sorted(paths, key=lambda x: x["cost"]):
        print(f"{p['type']}: length = {p['cost']:.4f}")

    print(f"\nНайкоротший шлях: {best_path['type']}, length = {best_path['cost']:.4f}")


if __name__ == "__main__":
    start = (1.0, 2.0, math.radians(0))
    goal  = (2.0, 5.0, math.radians(90))
    rho = 1.5

    plot_all_dubins_paths(start, goal, rho, step_size=0.05)