import math
import numpy as np
import matplotlib.pyplot as plt

def mod2pi(theta):
    return theta % (2.0 * math.pi)

# 6 типів функцій для обчислення сегментів шляху
def LSL(alpha, beta, d):
    tmp = d + math.sin(alpha) - math.sin(beta)
    p_squared = 2 + d**2 - 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))
    if p_squared < -1e-9: return None
    p = math.sqrt(max(0.0, p_squared))
    tmp2 = math.atan2(math.cos(beta) - math.cos(alpha), tmp)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(beta - tmp2)
    return t, p, q

def RSR(alpha, beta, d):
    p_squared = 2 + d**2 - 2 * math.cos(alpha - beta) + 2 * d * (-math.sin(alpha) + math.sin(beta))
    if p_squared < -1e-9: return None
    p = math.sqrt(max(0.0, p_squared))
    tmp = math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
    t = mod2pi(alpha - tmp)
    q = mod2pi(-beta + tmp)
    return t, p, q

def LSR(alpha, beta, d):
    p_squared = -2 + d**2 + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < -1e-9: return None
    p = math.sqrt(max(0.0, p_squared))
    tmp = math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp)
    q = mod2pi(-beta + tmp)
    return t, p, q

def RSL(alpha, beta, d):
    p_squared = -2 + d**2 + 2 * math.cos(alpha - beta) - 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < -1e-9: return None
    p = math.sqrt(max(0.0, p_squared))
    tmp = math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) - math.sin(beta)) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp)
    q = mod2pi(beta - tmp)
    return t, p, q

def RLR(alpha, beta, d):
    tmp = (6.0 - d**2 + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp) > 1.0: return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + p / 2.0)
    q = mod2pi(alpha - beta - t + p)
    return t, p, q

def LRL(alpha, beta, d):
    tmp = (6.0 - d**2 + 2 * math.cos(alpha - beta) + 2 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp) > 1.0: return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(-alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p / 2.0)
    q = mod2pi(beta - alpha - t + p)
    return t, p, q

PATH_TYPES = {"LSL": LSL, "RSR": RSR, "LSR": LSR, "RSL": RSL, "RLR": RLR, "LRL": LRL}

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
    for seg_type, seg_len in zip(segment_types, [t, p, q]):
        x, y, yaw, xs, ys, yaws = sample_segment(x, y, yaw, seg_type, seg_len, rho, step_size)
        segments.append({"type": seg_type, "x": xs, "y": ys})
    return segments

def plot_arrow(ax, x, y, yaw, length=0.7, color='red'):
    ax.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), 
             head_width=0.2, head_length=0.3, fc=color, ec=color, length_includes_head=True)

def plot_dubins_grid(start, goal, rho, step_size=0.05):
    x0, y0, th0 = start
    x1, y1, th1 = goal
    dx, dy = x1 - x0, y1 - y0
    D = math.hypot(dx, dy)
    d = D / rho
    theta = mod2pi(math.atan2(dy, dx))
    alpha, beta = mod2pi(th0 - theta), mod2pi(th1 - theta)

    # Створюємо сітку 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    path_names = ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]
    
    for i, name in enumerate(path_names):
        ax = axes[i]
        path_func = PATH_TYPES[name]
        result = path_func(alpha, beta, d)
        
        # Початкова та кінцева точки
        ax.scatter([start[0]], [start[1]], color='orange', s=60, zorder=5)
        ax.scatter([goal[0]], [goal[1]], color='purple', s=60, zorder=5)
        plot_arrow(ax, *start, color='orange')
        plot_arrow(ax, *goal, color='purple')
        
        if result:
            t, p, q = result
            cost = (t + p + q) * rho
            path = {"type": name, "segments": (t, p, q), "rho": rho, "start": start}
            segments = generate_dubins_segments(path, step_size=step_size)
            
            for seg in segments:
                style = '--' if seg["type"] in ['L', 'R'] else '-'
                ax.plot(seg["x"], seg["y"], linestyle=style, linewidth=2, label=f"{name}")
            ax.set_title(f"Type: {name}\nLength: {cost:.3f}")
        else:
            ax.set_title(f"Type: {name}\n(Path not possible)")
            
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_pos = (3.0, 5.0, math.radians(70))
    goal_pos  = (7.0, 1.0, math.radians(45))
    min_radius = 1.5
    
    plot_dubins_grid(start_pos, goal_pos, min_radius)