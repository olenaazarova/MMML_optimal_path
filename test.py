import argparse
import glob
import importlib.util
import math
import os
import sys
import unittest
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))


def _find_target_file() -> str:
    path = "dubins_all_paths.py"
    candidates = sorted(
        p for p in glob.glob(os.path.join(HERE, "dubins_all_paths*.py"))
        if os.path.basename(p) != "test.py"
    )
    if not candidates:
        raise FileNotFoundError("Could not find dubins_all_paths*.py next to test.py")
    return candidates[0]


TARGET_FILE = _find_target_file()


def _load_module():
    spec = importlib.util.spec_from_file_location("dubins_under_test", TARGET_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_module()

EPS_POS = 2e-6
EPS_YAW = 2e-6
DRAW_DIR = os.path.join(HERE, "test_outputs")


SCENARIOS: Dict[str, Dict] = {
    "default_case": {
        "start": (3.0, 5.0, math.radians(70.0)),
        "goal": (7.0, 1.0, math.radians(45.0)),
        "rho": 1.5,
        "expected_infeasible": ["RSL"],
    },
    "lsr_infeasible": {
        "start": (0.0, 0.0, 0.0),
        "goal": (-1.0, 1.0, 0.0),
        "rho": 1.0,
        "expected_infeasible": ["LSR"],
    },
    "rsl_infeasible": {
        "start": (3.0, 5.0, math.radians(70.0)),
        "goal": (7.0, 1.0, math.radians(45.0)),
        "rho": 1.5,
        "expected_infeasible": ["RSL"],
    },
    "all_six_feasible": {
        "start": (0.5533932670961299, 3.9384973280685855, -0.7546336666999074),
        "goal": (-1.2069956326863385, 0.6576150143724178, -2.2318670927476383),
        "rho": 1.463423610190707,
        "expected_infeasible": [],
    },
    "start_equals_goal": {
        "start": (0.0, 0.0, 0.0),
        "goal": (0.0, 0.0, 0.0),
        "rho": 1.0,
        "expected_infeasible": [],
    },
    "same_position_diff_heading": {
        "start": (0.0, 0.0, 0.0),
        "goal": (0.0, 0.0, math.pi / 2),
        "rho": 1.0,
        "expected_infeasible": ["LSR", "RSL"],
    },
    "angle_wraparound": {
        "start": (0.0, 0.0, math.pi - 1e-6),
        "goal": (3.0, -2.0, -math.pi + 2e-6),
        "rho": 1.0,
        "expected_infeasible": None,
    },
    "very_small_rho": {
        "start": (0.0, 0.0, math.radians(10.0)),
        "goal": (2.0, 1.0, math.radians(-30.0)),
        "rho": 0.05,
        "expected_infeasible": ["LRL", "RLR"],
    },
    "very_large_rho": {
        "start": (0.0, 0.0, 0.0),
        "goal": (8.0, 5.0, math.radians(20.0)),
        "rho": 10.0,
        "expected_infeasible": ["LSR"],
    },
}


def get_candidates(scenario_name: str):
    s = SCENARIOS[scenario_name]
    return mod.compute_all_dubins_paths(s["start"], s["goal"], s["rho"])


def get_candidate(candidates, name: str):
    for cand in candidates:
        if cand.name == name:
            return cand
    raise AssertionError(f"Candidate {name} not found")



def best_candidate(candidates):
    feasible = [c for c in candidates if c.feasible]
    return min(feasible, key=lambda c: c.total_length) if feasible else None



def trace_errors(start, goal, rho, candidate, ds=1e-3):
    trace = mod.trace_candidate(start, candidate, rho, ds)
    if trace is None:
        return None
    fx, fy, fyaw = trace["final_state"]
    pos_err = math.hypot(fx - goal[0], fy - goal[1])
    yaw_err = mod.angle_diff(fyaw, goal[2])
    return pos_err, yaw_err



def render_scenario(name: str, out_dir: str = DRAW_DIR, ds: float = 1e-3) -> str:
    os.makedirs(out_dir, exist_ok=True)
    s = SCENARIOS[name]
    out_path = os.path.join(out_dir, f"{name}.png")
    mod.plot_all_candidates(s["start"], s["goal"], rho=s["rho"], ds=ds, save_path=out_path)
    plt.close("all")
    if not os.path.exists(out_path):
        raise AssertionError(f"Plot was not saved for scenario {name}")
    if os.path.getsize(out_path) == 0:
        raise AssertionError(f"Saved plot is empty for scenario {name}")
    return out_path



def render_all_scenarios(out_dir: str = DRAW_DIR, ds: float = 1e-3):
    paths = []
    for name in SCENARIOS:
        paths.append(render_scenario(name, out_dir=out_dir, ds=ds))
    return paths


class DubinsBaseTestCase(unittest.TestCase):
    def assert_goal_reached(self, start, goal, rho, candidate, ds=1e-3):
        self.assertTrue(candidate.feasible, f"Candidate {candidate.name} is not feasible")
        errors = trace_errors(start, goal, rho, candidate, ds)
        self.assertIsNotNone(errors, f"Trace for {candidate.name} should not be None")
        pos_err, yaw_err = errors
        self.assertLessEqual(pos_err, EPS_POS, f"{candidate.name} position error too large: {pos_err}")
        self.assertLessEqual(yaw_err, EPS_YAW, f"{candidate.name} yaw error too large: {yaw_err}")


class TestHelpers(DubinsBaseTestCase):
    def test_mod2pi_range(self):
        for angle in [0.0, 1.0, -1.0, 7.0, -8.5, 100 * math.pi]:
            wrapped = mod.mod2pi(angle)
            self.assertGreaterEqual(wrapped, 0.0)
            self.assertLess(wrapped, 2.0 * math.pi)

    def test_angle_diff_basic_properties(self):
        self.assertAlmostEqual(mod.angle_diff(0.0, 0.0), 0.0)
        self.assertAlmostEqual(mod.angle_diff(0.0, 2.0 * math.pi), 0.0)
        self.assertAlmostEqual(mod.angle_diff(0.0, math.pi), math.pi)
        self.assertAlmostEqual(mod.angle_diff(0.1, 2.0 * math.pi + 0.1), 0.0)
        self.assertAlmostEqual(mod.angle_diff(0.3, 1.7), mod.angle_diff(1.7, 0.3))

    def test_turning_circle_centers_for_heading_zero(self):
        state = (2.0, 3.0, 0.0)
        rho = 1.5
        left = mod.turning_circle_center(state, "L", rho)
        right = mod.turning_circle_center(state, "R", rho)
        self.assertAlmostEqual(left[0], 2.0)
        self.assertAlmostEqual(left[1], 4.5)
        self.assertAlmostEqual(right[0], 2.0)
        self.assertAlmostEqual(right[1], 1.5)

    def test_advance_state_straight(self):
        state = (1.0, 2.0, math.pi / 2)
        new_state = mod.advance_state(state, "S", 3.0, rho=2.0)
        self.assertAlmostEqual(new_state[0], 1.0, places=9)
        self.assertAlmostEqual(new_state[1], 5.0, places=9)
        self.assertAlmostEqual(new_state[2], math.pi / 2, places=9)


class TestCoreDubinsComputation(DubinsBaseTestCase):
    def test_compute_returns_exactly_six_named_candidates(self):
        candidates = mod.compute_all_dubins_paths((0.0, 0.0, 0.0), (4.0, 2.0, math.pi / 4), rho=1.0)
        self.assertEqual(len(candidates), 6)
        self.assertEqual([c.name for c in candidates], ["LSL", "LSR", "RSL", "RSR", "LRL", "RLR"])

    def test_expected_infeasible_candidates_match_known_cases(self):
        for name, scenario in SCENARIOS.items():
            expected = scenario["expected_infeasible"]
            if expected is None:
                continue
            candidates = get_candidates(name)
            infeasible = sorted(c.name for c in candidates if not c.feasible)
            self.assertEqual(infeasible, sorted(expected), msg=f"Scenario {name} has unexpected infeasible set")

    def test_feasible_candidates_hit_goal_in_every_scenario(self):
        for name, scenario in SCENARIOS.items():
            candidates = get_candidates(name)
            ds = 5e-4 if name == "very_small_rho" else 1e-3
            for cand in candidates:
                if cand.feasible:
                    self.assert_goal_reached(scenario["start"], scenario["goal"], scenario["rho"], cand, ds=ds)
                else:
                    self.assertIsNone(mod.trace_candidate(scenario["start"], cand, scenario["rho"], ds))

    def test_all_six_feasible_case_really_has_six(self):
        candidates = get_candidates("all_six_feasible")
        self.assertTrue(all(c.feasible for c in candidates))

    def test_lengths_of_feasible_paths_are_nonnegative(self):
        for name in SCENARIOS:
            candidates = get_candidates(name)
            for cand in candidates:
                if cand.feasible:
                    self.assertIsNotNone(cand.total_length)
                    self.assertGreaterEqual(cand.total_length, 0.0)
                    self.assertEqual(len(cand.params), 3)
                    self.assertTrue(all(p >= 0.0 for p in cand.params))


class TestBestPathSelection(DubinsBaseTestCase):
    def test_best_path_is_true_minimum(self):
        for name in SCENARIOS:
            candidates = get_candidates(name)
            feasible = [c for c in candidates if c.feasible]
            best = best_candidate(candidates)
            self.assertIsNotNone(best)
            self.assertEqual(best.total_length, min(c.total_length for c in feasible))

    def test_mirror_symmetry_swaps_left_right_families(self):
        start = (3.0, 5.0, math.radians(70.0))
        goal = (7.0, 1.0, math.radians(45.0))
        rho = 1.5

        original = {c.name: c for c in mod.compute_all_dubins_paths(start, goal, rho)}
        mirrored_start = (start[0], -start[1], -start[2])
        mirrored_goal = (goal[0], -goal[1], -goal[2])
        mirrored = {c.name: c for c in mod.compute_all_dubins_paths(mirrored_start, mirrored_goal, rho)}

        swaps = [("LSL", "RSR"), ("LSR", "RSL"), ("LRL", "RLR")]
        for left, right in swaps:
            self.assertEqual(original[left].feasible, mirrored[right].feasible)
            self.assertEqual(original[right].feasible, mirrored[left].feasible)
            if original[left].feasible and mirrored[right].feasible:
                self.assertAlmostEqual(original[left].total_length, mirrored[right].total_length, places=9)
            if original[right].feasible and mirrored[left].feasible:
                self.assertAlmostEqual(original[right].total_length, mirrored[left].total_length, places=9)


class TestPlotting(DubinsBaseTestCase):
    def test_plot_all_scenarios(self):
        paths = render_all_scenarios()
        self.assertEqual(len(paths), len(SCENARIOS))
        for path in paths:
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)


def _build_suite() -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for case in [TestHelpers, TestCoreDubinsComputation, TestBestPathSelection, TestPlotting]:
        suite.addTests(loader.loadTestsFromTestCase(case))
    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests for dubins_all_paths file, with mandatory rendering.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose unittest output")
    parser.add_argument("--draw-only", action="store_true", help="Only render all predefined scenarios without running tests")
    parser.add_argument("--out-dir", type=str, default=DRAW_DIR, help="Directory where rendered PNG files are saved")
    args = parser.parse_args()

    if args.draw_only:
        paths = render_all_scenarios(out_dir=args.out_dir)
        print(f"Rendered {len(paths)} scenario figures to: {args.out_dir}")
        for path in paths:
            print(path)
        raise SystemExit(0)

    DRAW_DIR = args.out_dir
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(_build_suite())
    if result.wasSuccessful():
        print(f"\nAll tests passed. Rendered figures are in: {DRAW_DIR}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
