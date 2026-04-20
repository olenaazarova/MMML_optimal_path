import argparse
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUTPUT_DIR = Path('test_rdp_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rdp_module():
    here = Path(__file__).resolve().parent
    rdp_path = here / "rdp.py"

    spec = importlib.util.spec_from_file_location("rdp_module", rdp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rdp = load_rdp_module()


SCENARIOS = [
    {
        'name': 'default_project_case',
        'start': (3.0, 5.0, math.radians(140.0)),
        'goal_xy': (3.0, 1.0),
        'rho': 1.5,
        'ds': 0.01,
    },
    {
        'name': 'all_four_feasible',
        'start': (0.0, 0.0, 0.0),
        'goal_xy': (1.0, 0.0),
        'rho': 1.0,
        'ds': 0.01,
    },
    {
        'name': 'LS_infeasible',
        'start': (0.0, 0.0, 0.0),
        'goal_xy': (0.0, 1.0),
        'rho': 1.0,
        'ds': 0.01,
    },
    {
        'name': 'RS_infeasible',
        'start': (0.0, 0.0, 0.0),
        'goal_xy': (0.0, -1.0),
        'rho': 1.0,
        'ds': 0.01,
    },
    {
        'name': 'upper_goal_case',
        'start': (0.0, 0.0, 0.0),
        'goal_xy': (0.0, 3.0),
        'rho': 1.0,
        'ds': 0.01,
    },
    {
        'name': 'lower_goal_case',
        'start': (0.0, 0.0, 0.0),
        'goal_xy': (0.0, -3.0),
        'rho': 1.0,
        'ds': 0.01,
    },
    {
        'name': 'start_equals_goal',
        'start': (0.0, 0.0, 0.0),
        'goal_xy': (0.0, 0.0),
        'rho': 1.0,
        'ds': 0.01,
    },
    {
        'name': 'near_pi_heading',
        'start': (0.0, 0.0, math.pi - 1e-6),
        'goal_xy': (-2.0, 1.0),
        'rho': 1.1,
        'ds': 0.01,
    },
    {
        'name': 'tiny_rho',
        'start': (0.0, 0.0, math.radians(20.0)),
        'goal_xy': (1.5, -0.2),
        'rho': 0.1,
        'ds': 0.002,
    },
    {
        'name': 'large_rho',
        'start': (0.0, 0.0, math.radians(20.0)),
        'goal_xy': (4.0, 0.2),
        'rho': 5.0,
        'ds': 0.02,
    },
]


def scenario_output_path(name: str) -> Path:
    return OUTPUT_DIR / f'{name}.png'


def draw_all_scenarios(verbose: bool = True):
    created = []
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        for case in SCENARIOS:
            save_path = scenario_output_path(case['name'])
            rdp.plot_all_relaxed_candidates(
                case['start'],
                case['goal_xy'],
                rho=case['rho'],
                ds=case['ds'],
                save_path=str(save_path),
            )
            plt.close('all')
            created.append(save_path)
            if verbose:
                print(f'[drawn] {case["name"]} -> {save_path}')
    finally:
        plt.show = original_show
    return created


class TestRDPHelpers(unittest.TestCase):
    def test_mod2pi_range(self):
        vals = [-7.0, -0.1, 0.0, 0.1, 7.0, 100.0]
        for v in vals:
            wrapped = rdp.mod2pi(v)
            self.assertGreaterEqual(wrapped, 0.0)
            self.assertLess(wrapped, 2.0 * math.pi)

    def test_turning_circle_centers(self):
        state = (0.0, 0.0, 0.0)
        self.assertAlmostEqual(rdp.turning_circle_center(state, 'L', 1.0)[0], 0.0)
        self.assertAlmostEqual(rdp.turning_circle_center(state, 'L', 1.0)[1], 1.0)
        self.assertAlmostEqual(rdp.turning_circle_center(state, 'R', 1.0)[0], 0.0)
        self.assertAlmostEqual(rdp.turning_circle_center(state, 'R', 1.0)[1], -1.0)

    def test_advance_state_straight(self):
        state = (1.0, 2.0, 0.0)
        x, y, yaw = rdp.advance_state(state, 'S', 3.0, 1.0)
        self.assertAlmostEqual(x, 4.0, places=9)
        self.assertAlmostEqual(y, 2.0, places=9)
        self.assertAlmostEqual(yaw, 0.0, places=9)

    def test_advance_state_left_arc_preserves_radius(self):
        state = (0.0, 0.0, 0.0)
        rho = 2.0
        center = rdp.turning_circle_center(state, 'L', rho)
        end = rdp.advance_state(state, 'L', math.pi, rho)
        dist = math.hypot(end[0] - center[0], end[1] - center[1])
        self.assertAlmostEqual(dist, rho, places=8)


class TestRDPComputation(unittest.TestCase):
    def test_returns_exactly_four_named_candidates(self):
        case = SCENARIOS[0]
        cands = rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho'])
        self.assertEqual([c.name for c in cands], ['LS', 'RS', 'LR', 'RL'])

    def test_all_four_feasible_case(self):
        case = next(c for c in SCENARIOS if c['name'] == 'all_four_feasible')
        cands = rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho'])
        self.assertTrue(all(c.feasible for c in cands))

    def test_ls_infeasible_case(self):
        case = next(c for c in SCENARIOS if c['name'] == 'LS_infeasible')
        by_name = {c.name: c for c in rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho'])}
        self.assertFalse(by_name['LS'].feasible)
        self.assertTrue(by_name['RS'].feasible)

    def test_rs_infeasible_case(self):
        case = next(c for c in SCENARIOS if c['name'] == 'RS_infeasible')
        by_name = {c.name: c for c in rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho'])}
        self.assertFalse(by_name['RS'].feasible)
        self.assertTrue(by_name['LS'].feasible)

    def test_upper_goal_best_path(self):
        case = next(c for c in SCENARIOS if c['name'] == 'upper_goal_case')
        feasible = [c for c in rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho']) if c.feasible]
        best = min(feasible, key=lambda c: c.total_length)
        self.assertEqual(best.name, 'LS')

    def test_lower_goal_best_path(self):
        case = next(c for c in SCENARIOS if c['name'] == 'lower_goal_case')
        feasible = [c for c in rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho']) if c.feasible]
        best = min(feasible, key=lambda c: c.total_length)
        self.assertEqual(best.name, 'RS')

    def test_start_equals_goal_contains_zero_length_solution(self):
        case = next(c for c in SCENARIOS if c['name'] == 'start_equals_goal')
        by_name = {c.name: c for c in rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho'])}

        self.assertFalse(by_name['LS'].feasible)
        self.assertFalse(by_name['RS'].feasible)

        self.assertTrue(by_name['LR'].feasible)
        self.assertTrue(by_name['RL'].feasible)
        self.assertAlmostEqual(by_name['LR'].total_length, 0.0, places=8)
        self.assertAlmostEqual(by_name['RL'].total_length, 0.0, places=8)

    def test_best_path_is_minimum_length(self):
        for case in SCENARIOS:
            cands = rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho'])
            feasible = [c for c in cands if c.feasible]
            self.assertGreater(len(feasible), 0, msg=case['name'])
            best = min(feasible, key=lambda c: c.total_length)
            self.assertTrue(
                all(best.total_length <= c.total_length + 1e-12 for c in feasible),
                msg=case['name'],
            )

    def test_feasible_lengths_are_nonnegative(self):
        for case in SCENARIOS:
            for cand in rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho']):
                if cand.feasible:
                    self.assertIsNotNone(cand.total_length)
                    self.assertGreaterEqual(cand.total_length, -1e-12, msg=f"{case['name']}:{cand.name}")


class TestRDPTracing(unittest.TestCase):
    def test_all_feasible_traces_reach_goal_point(self):
        for case in SCENARIOS:
            cands = rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho'])
            for cand in cands:
                trace = rdp.trace_relaxed_candidate(case['start'], cand, case['rho'], case['ds'])

                if not cand.feasible:
                    self.assertIsNone(trace, msg=f"{case['name']}:{cand.name}")
                    continue

                self.assertIsNotNone(trace, msg=f"{case['name']}:{cand.name}")
                fx, fy, _ = trace['final_state']
                gx, gy = case['goal_xy']
                pos_err = math.hypot(fx - gx, fy - gy)
                self.assertLess(
                    pos_err,
                    5e-6 + 5 * case['ds'],
                    msg=f"{case['name']}:{cand.name} pos_err={pos_err}",
                )

    def test_infeasible_trace_returns_none(self):
        case = next(c for c in SCENARIOS if c['name'] == 'LS_infeasible')
        cand = next(c for c in rdp.compute_all_relaxed_paths(case['start'], case['goal_xy'], case['rho']) if c.name == 'LS')
        self.assertFalse(cand.feasible)
        self.assertIsNone(rdp.trace_relaxed_candidate(case['start'], cand, case['rho'], case['ds']))


class TestRDPPlotting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.created = draw_all_scenarios(verbose=False)

    def test_all_scenario_images_created(self):
        self.assertEqual(len(self.created), len(SCENARIOS))
        for p in self.created:
            self.assertTrue(p.exists(), msg=str(p))
            self.assertGreater(p.stat().st_size, 0, msg=str(p))


def main():
    parser = argparse.ArgumentParser(description='Tests and drawing for rdp.py')
    parser.add_argument('--draw-only', action='store_true', help='Only draw all prepared scenarios and save PNG files.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose unittest output.')
    args, remaining = parser.parse_known_args()

    if args.draw_only:
        created = draw_all_scenarios(verbose=True)
        print(f'Created {len(created)} image(s) in {OUTPUT_DIR}')
        return

    unittest_argv = [sys.argv[0]]
    if args.verbose:
        unittest_argv.append('-v')
    unittest_argv.extend(remaining)
    unittest.main(argv=unittest_argv)


if __name__ == '__main__':
    main()
