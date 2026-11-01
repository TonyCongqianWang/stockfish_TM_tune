"""Reproducible, dependency-free setup. No system Python packages are modified."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def command(args, cwd=None):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkout(url, revision, path):
    if not (path / '.git').exists():
        command(['git', 'init', path])
        command(['git', 'remote', 'add', 'origin', url], path)
    probe = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=path,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    current = probe.stdout.strip() if probe.returncode == 0 else ''
    if current != revision:
        command(['git', 'fetch', '--depth', '1', 'origin', revision], path)
    # A --no-checkout clone can already point at the requested SHA while its
    # worktree is empty. Always materialize it, even when HEAD already matches.
    command(['git', 'checkout', '--detach', revision], path)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path).decode().strip()


def stable_cpu():
    if not Path('/proc/cpuinfo').exists():
        return os.uname().machine
    keys = {'vendor_id', 'cpu family', 'model', 'model name', 'stepping', 'flags',
            'Features', 'CPU architecture', 'CPU implementer', 'CPU part'}
    # Exclude CPU MHz and other changing runtime fields so resume does not rebuild.
    first = Path('/proc/cpuinfo').read_text().split('\n\n')[0]
    return '\n'.join(line for line in first.splitlines() if line.split(':', 1)[0].strip() in keys)


def setup(jobs=2, arch='native'):
    source = json.loads((ROOT / 'sources.json').read_text())
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    # Each source tree is independent so one build cannot overwrite another's objects.
    revisions = {}
    for name, repo, ref in [('candidate', source['engine_repo'], source['engine_ref']),
                            ('baseline', source['engine_repo'], source['baseline_ref']),
                            ('fastchess', source['fastchess_repo'], source['fastchess_ref'])]:
        path = cache / name
        # An empty freshly initialized repository has no HEAD revision yet.
        if not (path / '.git').exists():
            command(['git', 'clone', '--no-checkout', '--filter=blob:none', repo, path])
        revisions[name] = checkout(repo, ref, path)
        executable = path / ('fastchess' if name == 'fastchess' else 'src/stockfish')
        stamp = path / '.tm-build.json'
        fingerprint = {'revision': revisions[name], 'arch': arch,
                       'machine': os.uname().machine,
                       'cpu': stable_cpu()}
        if not executable.exists() or not stamp.exists() or json.loads(stamp.read_text()) != fingerprint:
            if name == 'fastchess':
                command(['make', 'clean'], path)
                command(['make', f'-j{jobs}'], path)
            else:
                if name == 'baseline':
                    for net in (cache / 'candidate/src').glob('*.nnue'):
                        shutil.copy2(net, path / 'src' / net.name)
                command(['make', 'objclean'], path / 'src')
                command(['make', f'-j{jobs}', 'build', f'ARCH={arch}'], path / 'src')
            stamp.write_text(json.dumps(fingerprint))
    archive = cache / 'book.zip'
    if not archive.exists() or sha256(archive) != source['book_sha256']:
        temp = archive.with_suffix('.download')
        with urllib.request.urlopen(source['book_url'], timeout=300) as response, open(temp, 'wb') as out:
            shutil.copyfileobj(response, out)
        if sha256(temp) != source['book_sha256']:
            raise ValueError('Opening-book checksum mismatch')
        temp.replace(archive)
    book = cache / 'openings.epd'
    with zipfile.ZipFile(archive) as bundle:
        names = [n for n in bundle.namelist() if n.endswith('.epd')]
        if len(names) != 1:
            raise ValueError('Expected one EPD in opening archive')
        with bundle.open(names[0]) as inp, open(book, 'wb') as out:
            shutil.copyfileobj(inp, out)
    return {'candidate': str(cache / 'candidate/src/stockfish'),
            'baseline': str(cache / 'baseline/src/stockfish'),
            'fastchess': str(cache / 'fastchess/fastchess'), 'book': str(book),
            'revisions': revisions, 'book_sha256': sha256(book), 'arch': arch}
