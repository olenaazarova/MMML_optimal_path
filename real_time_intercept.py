from rdp import *
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


class TargetKalmanFilter:
    def __init__(self, first_obs, second_obs, process_noise=0.01, measurement_noise=0.1):
        t0, x0, y0 = first_obs
        t1, x1, y1 = second_obs

        dt = t1 - t0
        if dt <= 0:
            raise ValueError("Observation times must be increasing.")

        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt

        self.x = np.array([x1, y1, vx, vy], dtype=float)
        self.P = np.eye(4)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self.R = np.eye(2) * measurement_noise
        self.process_noise = process_noise
        self.last_time = t1

    def update(self, obs):
        t, mx, my = obs
        dt = t - self.last_time

        if dt <= 0:
            raise ValueError("Observation times must be increasing.")

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        Q = np.eye(4) * self.process_noise

        # prediction
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # correction
        z = np.array([mx, my], dtype=float)
        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.last_time = t

    def predict_position(self, t):
        dt = t - self.last_time
        x, y, vx, vy = self.x
        return x + vx * dt, y + vy * dt

    def latest_state(self):
        x, y, vx, vy = self.x
        return (x, y), (vx, vy), self.last_time
    
def best_relaxed_path_to_point(start, goal_xy, rho):
    candidates = compute_all_relaxed_paths(start, goal_xy, rho)
    feasible = [c for c in candidates if c.feasible]

    if not feasible:
        return None

    return min(feasible, key=lambda c: c.total_length)


def find_interception_from_kalman(
    pursuer_state,
    pursuer_speed,
    kf,
    rho,
    current_time,
    horizon=50.0,
    dt=0.05,
):
    prev_t = current_time

    target_xy = kf.predict_position(prev_t)
    prev_path = best_relaxed_path_to_point(pursuer_state, target_xy, rho)

    if prev_path is None:
        return None, None, None

    prev_error = prev_path.total_length / pursuer_speed

    t = current_time + dt

    while t <= current_time + horizon:
        target_xy = kf.predict_position(t)
        path = best_relaxed_path_to_point(pursuer_state, target_xy, rho)

        if path is not None:
            error = path.total_length / pursuer_speed - (t - current_time)

            if prev_error * error <= 0:
                return t, target_xy, path

            prev_error = error

        t += dt

    return None, None, None

def move_along_candidate(state, candidate, distance, rho):
    cur = state
    remaining = distance

    for mode, param in zip(candidate.modes, candidate.params):
        segment_length = param if mode == "S" else param * rho

        step = min(remaining, segment_length)
        cur = advance_state(cur, mode, step, rho)

        remaining -= step

        if remaining <= 1e-9:
            break

    return cur

def simulate_realtime_interception(
    start,
    target_initial,
    target_velocity,
    pursuer_speed,
    rho,
    total_time=30.0,
    dt=0.1,
    observation_noise=0.05,
):
    current_time = 0.0
    pursuer_state = start

    true_target_positions = []
    observed_positions = []
    pursuer_positions = []

    obs1_true = target_initial

    obs1 = (
        0.0,
        obs1_true[0] + np.random.normal(0, observation_noise),
        obs1_true[1] + np.random.normal(0, observation_noise),
    )

    true2 = (
        target_initial[0] + target_velocity[0] * dt,
        target_initial[1] + target_velocity[1] * dt,
    )

    obs2 = (
        dt,
        true2[0] + np.random.normal(0, observation_noise),
        true2[1] + np.random.normal(0, observation_noise),
    )

    kf = TargetKalmanFilter(obs1, obs2)

    current_time = 2 * dt

    while current_time <= total_time:
        true_target = (
            target_initial[0] + target_velocity[0] * current_time,
            target_initial[1] + target_velocity[1] * current_time,
        )

        noisy_obs = (
            current_time,
            true_target[0] + np.random.normal(0, observation_noise),
            true_target[1] + np.random.normal(0, observation_noise),
        )

        kf.update(noisy_obs)

        interception_time, interception_xy, path = find_interception_from_kalman(
            pursuer_state=pursuer_state,
            pursuer_speed=pursuer_speed,
            kf=kf,
            rho=rho,
            current_time=current_time,
        )

        if path is not None:
            pursuer_state = move_along_candidate(
                pursuer_state,
                path,
                pursuer_speed * dt,
                rho,
            )

        true_target_positions.append(true_target)
        observed_positions.append((noisy_obs[1], noisy_obs[2]))
        pursuer_positions.append((pursuer_state[0], pursuer_state[1]))

        distance_to_target = math.hypot(
            pursuer_state[0] - true_target[0],
            pursuer_state[1] - true_target[1],
        )

        if distance_to_target < 0.05:
            print(f"Intercepted at time {current_time:.2f}")
            break

        current_time += dt

    return true_target_positions, observed_positions, pursuer_positions

def plot_realtime_result(true_target_positions, observed_positions, pursuer_positions):
    fig, ax = plt.subplots(figsize=(9, 7))

    tx = [p[0] for p in true_target_positions]
    ty = [p[1] for p in true_target_positions]

    ox = [p[0] for p in observed_positions]
    oy = [p[1] for p in observed_positions]

    px = [p[0] for p in pursuer_positions]
    py = [p[1] for p in pursuer_positions]

    ax.plot(tx, ty, linewidth=2, label="true target trajectory")
    ax.scatter(ox, oy, s=12, alpha=0.5, label="noisy observations")
    ax.plot(px, py, linewidth=2.5, label="pursuer trajectory")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Real-time relaxed Dubins interception with Kalman filtering")

    plt.show()


def animate_realtime_result(
    true_target_positions,
    observed_positions,
    pursuer_positions,
    save_path="realtime_interception.gif",
    interval=120,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    tx = [p[0] for p in true_target_positions]
    ty = [p[1] for p in true_target_positions]

    ox = [p[0] for p in observed_positions]
    oy = [p[1] for p in observed_positions]

    px = [p[0] for p in pursuer_positions]
    py = [p[1] for p in pursuer_positions]

    all_x = tx + ox + px
    all_y = ty + oy + py

    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Real-time relaxed Dubins interception")

    true_line, = ax.plot([], [], linewidth=2, label="true target trajectory")
    obs_scatter = ax.scatter([], [], s=12, alpha=0.5, label="noisy observations")
    pursuer_line, = ax.plot([], [], linewidth=2.5, label="pursuer trajectory")

    target_dot, = ax.plot([], [], "o", markersize=8, label="target")
    pursuer_dot, = ax.plot([], [], "o", markersize=8, label="pursuer")

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.legend()

    def init():
        true_line.set_data([], [])
        pursuer_line.set_data([], [])
        target_dot.set_data([], [])
        pursuer_dot.set_data([], [])
        obs_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return true_line, pursuer_line, target_dot, pursuer_dot, obs_scatter, time_text

    def update(frame):
        true_line.set_data(tx[:frame + 1], ty[:frame + 1])
        pursuer_line.set_data(px[:frame + 1], py[:frame + 1])

        target_dot.set_data([tx[frame]], [ty[frame]])
        pursuer_dot.set_data([px[frame]], [py[frame]])

        obs_points = np.column_stack([ox[:frame + 1], oy[:frame + 1]])
        obs_scatter.set_offsets(obs_points)

        time_text.set_text(f"step = {frame}")

        return true_line, pursuer_line, target_dot, pursuer_dot, obs_scatter, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(true_target_positions),
        init_func=init,
        interval=interval,
        blit=True,
    )

    anim.save(save_path, writer=PillowWriter(fps=1000 // interval))
    plt.close(fig)

    print(f"GIF saved to: {save_path}")
if __name__ == "__main__":
    start = (0.0, 0.0, math.radians(0.0))

    target_initial = (1.0, 3.0)
    target_velocity = (0.15, 0.2)

    rho = 1.0
    pursuer_speed = 1.0

    true_target_positions, observed_positions, pursuer_positions = simulate_realtime_interception(
        start=start,
        target_initial=target_initial,
        target_velocity=target_velocity,
        pursuer_speed=pursuer_speed,
        rho=rho,
        total_time=40.0,
        dt=0.1,
        observation_noise=0.05,
    )

    # plot_realtime_result(
    #     true_target_positions,
    #     observed_positions,
    #     pursuer_positions,
    # )
    animate_realtime_result(
    true_target_positions,
    observed_positions,
    pursuer_positions,
    save_path="realtime_interception.gif",
    interval=120,
)