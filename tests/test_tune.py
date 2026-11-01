import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from tune import allocation, read_stats, score_summary, perturb, update, validate_config
from summarize import summarize


class TuneTests(unittest.TestCase):
    def test_exact_budget(self):
        strata = [{'weight': 1}] * 20
        self.assertEqual(sum(allocation(15000, strata)), 15000)
        self.assertEqual(allocation(7, [{'weight': 2}, {'weight': 1}]), [5, 2])

    def test_paired_variance(self):
        # Identical marginal WDL can have different opening-pair variance.
        base = dict(wins=2, losses=2, draws=0, penta_LL=0, penta_LD=0,
                    penta_WL=2, penta_DD=0, penta_WD=0, penta_WW=0)
        self.assertEqual(score_summary(base)['score_se'], 0)
        volatile = dict(base, penta_WL=0, penta_LL=1, penta_WW=1)
        self.assertGreater(score_summary(volatile)['score_se'], 0)

    def test_reversed_engine_stats_and_incomplete(self):
        stats = dict(wins=0, losses=2, draws=0, penta_LL=1, penta_LD=0,
                     penta_WL=0, penta_DD=0, penta_WD=0, penta_WW=0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fastchess.json'
            p.write_text(json.dumps({'stats': {'B vs A': stats}}))
            self.assertEqual(read_stats(p, 1)['score'], 1)
            with self.assertRaises(ValueError):
                read_stats(p, 2)

    def test_spsa_direction_and_boundary(self):
        cfg = {'seed': 5, 'spsa': {'perturbation': .1, 'gamma': .101, 'learning_rate': .01,
                                 'alpha': .602, 'stability': 10, 'max_step': .02}}
        specs = {'x': {'min': 0, 'max': 10000}}
        for theta in ({'x': 0}, {'x': .5}, {'x': 1}):
            plus, minus = perturb(theta, specs, 1, cfg)
            response = (plus['x'] - minus['x']) / 10000
            new = update(theta, specs, plus, minus, response, 1, cfg)
            self.assertGreaterEqual(new['x'], theta['x'])
            self.assertLessEqual(new['x'], 1)
            self.assertEqual((plus, minus), perturb(theta, specs, 1, cfg))

    def test_kaggle_worker_ceiling(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/kaggle.json').read_text())
        cfg['oversubscription'] = 2
        validate_config(cfg)
        self.assertEqual(cfg['effective_workers'], 2)
        cfg['workers'] = 3
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_summary_real_fastchess_format_and_reentry(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            match = out / 'iteration-000001' / 'fast' / 'self'
            match.mkdir(parents=True)
            (match / 'result.json').write_text(json.dumps({'tc': '1.5+0', 'games': 1}))
            (match / 'games.pgn').write_text('''[Event "test"]
[White "A"]
[Black "B"]
[Result "1-0"]
[FEN "8/8/8/8/8/8/8/8 b - - 0 8"]
[Termination "time forfeit"]

8... e5 {0.12/10 0.015s, tl=1.400s, n=1200, nps=80000, latency=0.001s}
9. e4 {0.10/11 0.020s, tl=1.300s, n=1400, nps=70000} 1-0
''')
            first = summarize(out)
            self.assertEqual(len(first['move_averages']), 2)
            a = next(r for r in first['move_averages'] if r['engine'] == 'A')
            self.assertAlmostEqual(a['mean_spent_seconds'], .020)
            self.assertAlmostEqual(a['mean_tl'], 1.3)
            self.assertEqual(first['terminations'][0]['reason'], 'time forfeit')
            self.assertFalse((match / 'games.pgn').exists())
            self.assertEqual(first, summarize(out))


if __name__ == '__main__':
    unittest.main()
