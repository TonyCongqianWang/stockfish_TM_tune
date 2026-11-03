"""Unit tests for async experiment generation and optimization without chess games."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from async_tune import directions, schedule, create_experiment, gradient, apply


class AsyncTuneTests(unittest.TestCase):
    def test_fresh_async_run_does_not_create_sync_state(self):
        from tune import run
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = {'strata': [{'tc': '1.5+0'}],
                   'async_spsa': {'enabled': True}}
            specs = {'TMTest': {'min': 0, 'max': 100, 'default': 50}}
            with patch('tune.options', return_value=specs):
                with patch('async_tune.run_async', return_value='async-called') as runner:
                    result = run(out, cfg, {'candidate': 'candidate', 'baseline': 'baseline'})
            self.assertEqual(result, 'async-called')
            self.assertFalse((out / 'state.json').exists())
            runner.assert_called_once()

    def test_hypercube_near_orthogonal(self):
        names = [f'p{i}' for i in range(37)]
        vectors = directions(names, 123)
        self.assertEqual(vectors, directions(names, 123))
        for v in vectors:
            self.assertEqual(set(v), set(names))
            self.assertEqual(set(v.values()), {-1, 1})
        for i in range(4):
            for j in range(i):
                dot = sum(vectors[i][k] * vectors[j][k] for k in names)
                self.assertEqual(abs(dot), 1)

    def test_all_combinations_unique_per_cycle(self):
        cfg = {'seed': 31, 'async_spsa': {'loads': [8, 12, 16]},
               'strata': [{'id': f't{i}'} for i in range(20)]}
        sample = [schedule(cfg, i) for i in range(120)]
        self.assertEqual(len(set(sample)), 120)
        self.assertEqual(sample, [schedule(cfg, i) for i in range(120)])
        self.assertEqual({x[1] for x in sample}, {'self', 'master'})

    def test_experiment_budget_and_determinism(self):
        cfg = {'seed': 31, 'async_spsa': {'loads': [8, 12, 16]},
               'strata': [{'id': f't{i}'} for i in range(20)],
               'spsa': {'perturbation': .05, 'gamma': .101}}
        specs = {f'p{i}': {'min': 0, 'max': 1000} for i in range(37)}
        theta = {k: .5 for k in specs}
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(4):
                meta = create_experiment(Path(tmp), cfg, specs, theta, 0, i, i%2)
                self.assertEqual(len(meta['samples']), 4 if meta['objective'] == 'self' else 2)
                self.assertEqual(meta, create_experiment(Path(tmp), cfg, specs, theta, 0, i, i%2))

    def test_gradient_scale_and_movement(self):
        theta = {'x': .5, 'y': .5}
        meta = {'base_theta': theta, 'c': .1,
                'samples': [{'direction': {'x': 1, 'y': -1}}]}
        grad = gradient(meta, [.2])
        self.assertAlmostEqual(grad['x'], 1)
        self.assertAlmostEqual(grad['y'], -1)
        cfg = {'spsa': {'learning_rate': .01, 'stability': 10,
                         'alpha': .602, 'max_step': .02}}
        specs = {k: {'min': 0, 'max': 1000} for k in theta}
        new, movement = apply(theta, specs, grad, cfg, 1, theta, 0)
        self.assertGreater(new['x'], theta['x'])
        self.assertLess(new['y'], theta['y'])
        self.assertAlmostEqual(movement['step_l2'], math.sqrt(2)*.01)
        self.assertAlmostEqual(movement['net_l2'], movement['step_l2'])


if __name__ == '__main__':
    unittest.main()
