import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from resources import parse_cpus, select_cores, training_workers


class ResourcesTests(unittest.TestCase):
    def test_physical_core_selection_respects_allocation_and_excludes_siblings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Two packages, two cores each, two siblings per core.
            for cpu in range(8):
                topology = root / f'cpu{cpu}' / 'topology'
                topology.mkdir(parents=True)
                (topology / 'physical_package_id').write_text(str(cpu // 4))
                (topology / 'core_id').write_text(str((cpu % 4) // 2))
            self.assertEqual(select_cores({1, 2, 3, 4, 5, 7}, 3, root), [1, 2, 4])
            self.assertEqual(select_cores({1, 3}, 16, root), [1, 3])
            with self.assertRaises(ValueError):
                select_cores({9}, 2, root)

    def test_cpu_list(self):
        self.assertEqual(parse_cpus('1-3,8,10-11'), {1, 2, 3, 8, 10, 11})
        with self.assertRaises(ValueError):
            parse_cpus('3-1')

    def test_load_visits_entire_range_and_is_reproducible(self):
        cfg = {'seed': 12, 'workers': 16, 'effective_workers': 32, 'strata': [1] * 20}
        loads = [training_workers(cfg, 1, i) for i in range(17)]
        self.assertEqual(sorted(loads), list(range(16, 33)))
        self.assertEqual(loads, [training_workers(cfg, 1, i) for i in range(17)])
        cfg['effective_workers'] = 16
        self.assertEqual({training_workers(cfg, 3, i) for i in range(20)}, {16})

    def test_affinity_is_inherited_by_child(self):
        # Confine only this subprocess; leave the test runner's affinity untouched.
        script = '''import json, os, subprocess, sys
from resources import confine
cfg = {'workers': 2, 'max_workers': 4, 'oversubscription': 2, 'physical_cores': 2}
cpus = confine(cfg)
child = json.loads(subprocess.check_output([sys.executable, '-c',
    'import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))']))
assert child == cpus and 1 <= len(cpus) <= 2
print(json.dumps({'cpus': cpus, 'max_workers': cfg['effective_workers']}))
'''
        result = subprocess.check_output([sys.executable, '-c', script],
                                         cwd=Path(__file__).resolve().parents[1] / 'tools', text=True)
        self.assertLessEqual(len(json.loads(result.splitlines()[-1])['cpus']), 2)


if __name__ == '__main__':
    unittest.main()
