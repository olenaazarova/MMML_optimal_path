import math
import numpy as np
import matplotlib.pyplot as plt


def mod2pi(theta):
    return theta % (2.0 * math.pi)


def wrap_to_pi(theta):
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _dubins_LSL(alpha, beta, d):
    tmp = d + math.sin(alpha) - math.sin(beta)
    p_sq = 2 + d**2 - 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))
    if p_sq < 0:
        return None
    tmp2 = math.atan2(math.cos(beta) - math.cos(alpha), tmp)
    t = mod2pi(-alpha + tmp2)
    p = math.sqrt(p_sq)
    q = mod2pi(beta - tmp2)
    return ("LSL", [t, p, q])


def _dubins_RSR(alpha, beta, d):
    tmp = d - math.sin(alpha) + math.sin(beta)
    p_sq = 2 + d**2 - 2 * math.cos(alpha - beta) + 2 * d * (-math.sin(alpha) + math.sin(beta))
    if p_sq < 0:
        return None
    tmp2 = math.atan2(math.cos(alpha) - math.cos(beta), tmp)
    t = mod2pi(alpha - tmp2)
    p = math.sqrt(p_sq)
    q = mod2pi(-beta + tmp2)
    return ("RSR", [t, p, q])


def _dubins_LSR(alpha, beta, d):
    p_sq = -2 + d**2 + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp)
    q = mod2pi(-beta + tmp)
    return ("LSR", [t, p, q])


def _dubins_RSL(alpha, beta, d):
    p_sq = d**2 - 2 + 2 * math.cos(alpha - beta) - 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) - math.sin(beta)) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp)
    q = mod2pi(beta - tmp)
    return ("RSL", [t, p, q])


def _dubins_RLR(alpha, beta, d):
    tmp = (6.0 - d**2 + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + p / 2.0)
    q = mod2pi(alpha - beta - t + p)
    return ("RLR", [t, p, q])


def _dubins_LRL(alpha, beta, d):
    tmp = (6.0 - d**2 + 2 * math.cos(alpha - beta) + 2 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(-alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p / 2.0)
    q = mod2pi(mod2pi(beta - alpha) - t + p)
    return ("LRL", [t, p, q])


_DUBINS_BUILDERS = [
    _dubins_LSL,
    _dubins_RSR,
    _dubins_LSR,
    _dubins_RSL,
    _dubins_RLR,
    _dubins_LRL,
]

def _propagate_segment(x, y, yaw, seg_type, seg_norm, R, step_size=0.05):
    xs = [x]
    ys = [y]
    yaws = [yaw]

    if seg_type == "S":
        seg_len = seg_norm * R
        traveled = 0.0
        while traveled < seg_len:
            ds = min(step_size, seg_len - traveled)
            x += ds * math.cos(yaw)
            y += ds * math.sin(yaw)
            traveled += ds
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)

    elif seg_type == "L":
        turned = 0.0
        dtheta_step = step_size / R
        while turned < seg_norm:
            dtheta = min(dtheta_step, seg_norm - turned)
            x += R * (math.sin(yaw + dtheta) - math.sin(yaw))
            y += R * (-math.cos(yaw + dtheta) + math.cos(yaw))
            yaw = mod2pi(yaw + dtheta)
            turned += dtheta
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)

    elif seg_type == "R":
        turned = 0.0
        dtheta_step = step_size / R
        while turned < seg_norm:
            dtheta = min(dtheta_step, seg_norm - turned)
            x += R * (math.sin(yaw) - math.sin(yaw - dtheta))
            y += R * (math.cos(yaw - dtheta) - math.cos(yaw))
            yaw = mod2pi(yaw - dtheta)
            turned += dtheta
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)

    else:
        raise ValueError(f"Unknown segment type: {seg_type}")

    return x, y, yaw, xs, ys, yaws


def sample_dubins_path(start, mode, segs_norm, R, step_size=0.05):
    x, y, yaw = start
    path_x = [x]
    path_y = [y]
    path_yaw = [yaw]

    for seg_type, seg_norm in zip(mode, segs_norm):
        x, y, yaw, xs, ys, yaws = _propagate_segment(x, y, yaw, seg_type, seg_norm, R, step_size)
        path_x.extend(xs[1:])
        path_y.extend(ys[1:])
        path_yaw.extend(yaws[1:])

    return np.array(path_x), np.array(path_y), np.array(path_yaw), (x, y, yaw)


def dubins_shortest_path(start, goal, R, step_size=0.05):
    sx, sy, syaw = start
    gx, gy, gyaw = goal

    dx = gx - sx
    dy = gy - sy
    D = math.hypot(dx, dy)

    if R <= 0:
        raise ValueError("R must be > 0")

    d = D / R
    theta = math.atan2(dy, dx)
    alpha = mod2pi(syaw - theta)
    beta = mod2pi(gyaw - theta)

    best = None

    for builder in _DUBINS_BUILDERS:
        candidate = builder(alpha, beta, d)
        if candidate is None:
            continue

        mode, segs_norm = candidate
        total_length = R * sum(segs_norm)

        if best is None or total_length < best["length"]:
            px, py, pyaw, end_pose = sample_dubins_path(start, mode, segs_norm, R, step_size)
            best = {
                "mode": mode,
                "segments_norm": segs_norm,
                "length": total_length,
                "x": px,
                "y": py,
                "yaw": pyaw,
                "end_pose": end_pose,
                "goal": goal,
            }

    return best


def relaxed_dubins_path(start, goal_xy, R, n_heading_samples=720, step_size=0.05):
    gx, gy = goal_xy
    best = None

    for gyaw in np.linspace(0.0, 2.0 * math.pi, n_heading_samples, endpoint=False):
        candidate = dubins_shortest_path(start, (gx, gy, gyaw), R, step_size=step_size)
        if candidate is None:
            continue


        end_x, end_y, end_yaw = candidate["end_pose"]
        pos_err = math.hypot(end_x - gx, end_y - gy)

        score = candidate["length"] + 1000.0 * pos_err

        if best is None or score < best["score"]:
            best = {
                **candidate,
                "score": score,
                "goal_xy": goal_xy,
                "goal_yaw": gyaw,
                "pos_err": pos_err,
            }

    return best


if __name__ == "__main__":
    start = (0.0, 0.0, np.deg2rad(20.0))
    goal_xy = (6.0, 4.0)

    R = 1.0

    best = relaxed_dubins_path(
        start=start,
        goal_xy=goal_xy,
        R=R,
        n_heading_samples=720,
        step_size=0.03
    )

    if best is None:
        print("No relaxed Dubins path found.")
    else:
        print("Best relaxed Dubins path found:")
        print("  mode           =", best["mode"])
        print("  length         =", best["length"])
        print("  goal heading   =", np.rad2deg(best["goal_yaw"]))
        print("  endpoint error =", best["pos_err"])

        plt.figure(figsize=(7, 7))
        plt.plot(best["x"], best["y"], label=f"{best['mode']}, L={best['length']:.3f}")
        plt.scatter([start[0]], [start[1]], c="green", label="start")
        plt.scatter([goal_xy[0]], [goal_xy[1]], c="red", label="goal point")

        plt.arrow(
            start[0], start[1],
            0.6 * math.cos(start[2]), 0.6 * math.sin(start[2]),
            head_width=0.12, length_includes_head=True
        )
        plt.arrow(
            goal_xy[0], goal_xy[1],
            0.6 * math.cos(best["goal_yaw"]), 0.6 * math.sin(best["goal_yaw"]),
            head_width=0.12, length_includes_head=True
        )

        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.title("Relaxed Dubins path (free terminal heading)")
        plt.show()
