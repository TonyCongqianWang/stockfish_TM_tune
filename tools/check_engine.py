#!/usr/bin/env python3
"""Link and run TM invariants using the already-built candidate's compiler flags."""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', default='.cache/candidate/src')
    parser.add_argument('--arch', default='native')
    args = parser.parse_args()
    src = (ROOT / args.src).resolve()
    output = subprocess.check_output(['make', '-s', '--no-print-directory', f'ARCH={args.arch}',
                                      '--eval=tm-check-flags:;@echo TMFLAGS:$(CXXFLAGS)',
                                      'tm-check-flags'], cwd=src, text=True)
    flags = shlex.split(next(line.removeprefix('TMFLAGS:') for line in output.splitlines() if line.startswith('TMFLAGS:')))
    objects = [str(p) for p in sorted(src.glob('*.o')) if p.name != 'main.o']
    if not objects:
        raise SystemExit('Build the candidate first')
    binary = ROOT / '.cache/tm-invariants'
    binary.parent.mkdir(exist_ok=True)
    subprocess.run(['g++', *flags, f'-I{src}', str(ROOT / 'tests/timeman_invariants.cpp'),
                    *objects, '-lpthread', '-lrt', '-o', str(binary)], cwd=src, check=True)
    subprocess.run([binary], check=True)


if __name__ == '__main__':
    main()
