#!/usr/bin/env python3
"""Reduce retained fastchess PGNs to averages after the run, then gzip raw files."""
from collections import defaultdict, Counter
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import sys


def games(stream):
    headers, moves = {}, []
    for line in stream:
        if line.startswith('['):
            if moves:
                yield headers, ''.join(moves)
                headers, moves = {}, []
            match = re.match(r'\[(\w+) "(.*)"\]', line)
            if match:
                headers[match[1]] = match[2]
        elif line.strip():
            moves.append(line)
    if moves:
        yield headers, ''.join(moves)


def summarize(out):
    out = Path(out)
    buckets = defaultdict(lambda: defaultdict(float))
    terminations = Counter()
    matches = []
    for result_file in sorted(out.glob('**/result.json')):
        result = json.loads(result_file.read_text())
        relative = result_file.parent.relative_to(out)
        stage = relative.parts[0]
        tc = result['tc']
        if stage.startswith('iteration-'):
            kind = relative.parts[-1]
        elif stage == 'sweep-eval-base':
            stage = str(Path(*relative.parts[:2]))
            kind = 'sweep'
        else:
            kind = 'verification'
        matches.append({'path': str(relative), **{k: v for k, v in result.items() if k != 'identity'}})
        path = result_file.parent / 'games.pgn'
        compressed = path.with_suffix('.pgn.gz')
        opener, actual = (open, path) if path.exists() else (gzip.open, compressed)
        with opener(actual, 'rt') as stream:
            for headers, moves in games(stream):
                terminations[(stage, tc, kind, headers.get('Termination', 'unknown'))] += 1
                fen = headers.get('FEN', '8/8/8/8/8/8/8/8 w - - 0 1').split()
                initial = (int(fen[5]) - 1) * 2 + (fen[1] == 'b')
                for i, comment in enumerate(re.findall(r'\{([^}]*)\}', moves)):
                    # fastchess comments: score/depth seconds, tl=seconds, ...
                    match = re.match(r'\s*[^\s/]*/\d+\s+([0-9.]+)', comment)
                    if not match:
                        continue
                    ply = initial + i
                    name = headers.get('White' if ply % 2 == 0 else 'Black', '?')
                    bucket = (stage, tc, kind, name, (ply // 20) * 20)
                    agg = buckets[bucket]
                    elapsed = float(match[1])
                    agg['moves'] += 1
                    agg['spent_seconds_sum'] += elapsed
                    agg['spent_log_shifted_sum'] += math.log(elapsed + .01)
                    for field in ('tl', 'nps', 'n', 'latency'):
                        value = re.search(rf'(?:^|[,\s]){field}=(-?[0-9.]+)', comment)
                        if value:
                            agg[field + '_sum'] += float(value[1])
                            agg[field + '_count'] += 1
        # Compression happens only after final aggregation, with atomic replacement.
        if path.exists():
            temp = compressed.with_suffix('.gz.tmp')
            with open(path, 'rb') as inp, gzip.open(temp, 'wb', compresslevel=6) as dest:
                shutil.copyfileobj(inp, dest)
            temp.replace(compressed)
            path.unlink()
    rows = []
    for (stage, tc, kind, engine, ply), agg in sorted(buckets.items()):
        row = {'stage': stage, 'tc': tc, 'kind': kind, 'engine': engine, 'ply_bucket_start': ply,
               'moves': int(agg['moves']), 'mean_spent_seconds': agg['spent_seconds_sum'] / agg['moves'],
               'shifted_geomean_spent_seconds': math.exp(agg['spent_log_shifted_sum'] / agg['moves']) - .01}
        for field in ('tl', 'nps', 'n', 'latency'):
            if agg[field + '_count']:
                row['mean_' + field] = agg[field + '_sum'] / agg[field + '_count']
        rows.append(row)
    summary = {'matches': matches, 'move_averages': rows,
               'terminations': [{'stage': stage, 'tc': tc, 'kind': kind, 'reason': reason, 'games': count}
                                for (stage, tc, kind, reason), count in sorted(terminations.items())],
               'note': 'Ply buckets use game ply, not moves played since the opening. PGNs retained as gzip.'}
    temp = out / 'summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2) + '\n')
    temp.replace(out / 'summary.json')
    # Logs are also preserved, but no live compression or deletion during tuning.
    for path in out.glob('**/fastchess.log'):
        target = path.with_suffix('.log.gz')
        temp = target.with_suffix('.gz.tmp')
        with open(path, 'rb') as inp, gzip.open(temp, 'wb') as dest:
            shutil.copyfileobj(inp, dest)
        temp.replace(target)
        path.unlink()
    return summary


if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
