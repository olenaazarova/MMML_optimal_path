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
    if p_squared < 0:
        return None
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((math.cos(beta) - math.cos(alpha)), tmp)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(beta - tmp2)
    return t, p, q


def RSR(alpha, beta, d):
    tmp = d - math.sin(alpha) + math.sin(beta)
    p_squared = 2 + d**2 - 2 * math.cos(alpha - beta) + 2 * d * (-math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        return None
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((math.cos(alpha) - math.cos(beta)), tmp)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(-beta + tmp2)
    return t, p, q


def LSR(alpha, beta, d):
    p_squared = -2 + d**2 + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        return None
    p = math.sqrt(p_squared)
    tmp = math.atan2((-math.cos(alpha) - math.cos(beta)), (d + math.sin(alpha) + math.sin(beta))) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp)
    q = mod2pi(-mod2pi(beta) + tmp)
    return t, p, q


def RSL(alpha, beta, d):
    p_squared = -2 + d**2 + 2 * math.cos(alpha - beta) - 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        return None
    p = math.sqrt(p_squared)
    tmp = math.atan2((math.cos(alpha) + math.cos(beta)), (d - math.sin(alpha) - math.sin(beta))) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp)
    q = mod2pi(beta - tmp)
    return t, p, q


def RLR(alpha, beta, d):
    tmp = (6.0 - d**2 + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp) > 1:
        return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + p / 2.0)
    q = mod2pi(alpha - beta - t + p)
    return t, p, q


def LRL(alpha, beta, d):
    tmp = (6.0 - d**2 + 2 * math.cos(alpha - beta) + 2 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp) > 1:
        return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(-alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + p)
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
    """
    start = (x, y, yaw)
    goal  = (x, y, yaw)
    rho   = min turn radius

    return dict with the shortest Dubins path
    """
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



def sample_segment(x, y, yaw, segment_type, length, rho, step_size):
    dist = length * rho
    n_steps = max(1, int(dist / step_size))

    xs, ys, yaws = [], [], []

    for _ in range(n_steps):
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
            x += rho * (-math.sin(yaw - dtheta) + math.sin(yaw))
            y += rho * ( math.cos(yaw - dtheta) - math.cos(yaw))
            yaw -= dtheta

        yaw = mod2pi(yaw)
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)

        dist -= ds
        if dist <= 1e-10:
            break

    return x, y, yaw, xs, ys, yaws


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
        full_x.extend(xs)
        full_y.extend(ys)
        full_yaw.extend(yaws)

    return np.array(full_x), np.array(full_y), np.array(full_yaw)



def plot_arrow(x, y, yaw, length=0.6):
    plt.arrow(
        x, y,
        length * math.cos(yaw),
        length * math.sin(yaw),
        head_width=0.25,
        head_length=0.3,
        fc='red',
        ec='red'
    )


if __name__ == "__main__":
    start = (1.0, 2.0, math.radians(0))
    goal  = (10.0, 7.0, math.radians(90))     # x, y, yaw
    rho = 1.5                                 # min turn radius

    best_path = dubins_shortest_path(start, goal, rho)

    if best_path is None:
        print("Dubins path не знайдено.")
    else:
        print("Найкращий тип шляху:", best_path["type"])
        print("Сегменти (t, p, q):", best_path["segments"])
        print("Загальна довжина:", best_path["cost"])

        xs, ys, yaws = generate_dubins_path(best_path, step_size=0.05)

        plt.figure(figsize=(8, 6))
        plt.plot(xs, ys, label=f'Dubins path: {best_path["type"]}')
        plt.scatter([start[0], goal[0]], [start[1], goal[1]], c=['green', 'blue'], s=80)

        plot_arrow(*start)
        plot_arrow(*goal)

        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Shortest Dubins Path")
        plt.show()