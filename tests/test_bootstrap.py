import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from bootstrap import checkout, stable_cpu


class BootstrapTests(unittest.TestCase):
    def test_clone_at_requested_head_still_populates_worktree(self):
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d) / 'source', Path(d) / 'target'
            subprocess.run(['git', 'init', '-q', str(source)], check=True)
            (source / 'Makefile').write_text('all:\n\t@true\n')
            subprocess.run(['git', 'add', '.'], cwd=source, check=True)
            env = dict(os.environ, GIT_AUTHOR_NAME='Test', GIT_AUTHOR_EMAIL='test@example.com',
                       GIT_COMMITTER_NAME='Test', GIT_COMMITTER_EMAIL='test@example.com')
            subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=source, env=env, check=True)
            rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
            subprocess.run(['git', 'clone', '-q', '--no-checkout', str(source), str(target)], check=True)
            self.assertFalse((target / 'Makefile').exists())
            self.assertEqual(checkout(str(source), rev, target), rev)
            self.assertTrue((target / 'Makefile').exists())

    def test_cpu_fingerprint_omits_dynamic_frequency(self):
        self.assertNotIn('cpu MHz', stable_cpu())


if __name__ == '__main__':
    unittest.main()
