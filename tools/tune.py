#!/usr/bin/env python3
"""Mixed-TC SPSA orchestrator. All games and clock decisions belong to fastchess."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import subprocess
import sys
import time

from bootstrap import ROOT, setup, sha256
from resources import confine, training_workers


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with open(temp, 'w') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def options(binary):
    result = subprocess.run([binary], input='uci\nquit\n', text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, check=True)
    if 'uciok' not in result.stdout:
        raise ValueError(f'UCI handshake failed: {binary}')
    found = {}
    for name, default, low, high in re.findall(
            r'^option name (TM\w+) type spin default (-?\d+) min (-?\d+) max (-?\d+)$',
            result.stdout, re.M):
        found[name] = {'default': int(default), 'min': int(low), 'max': int(high)}
    return found


def allocation(total, strata):
    """Largest-remainder allocation preserves exactly the declared pair budget."""
    weights = [s['weight'] for s in strata]
    raw = [total * w / sum(weights) for w in weights]
    result = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: (raw[i] - result[i], -i), reverse=True)[:total-sum(result)]:
        result[i] += 1
    return result


def score_summary(stats):
    n = stats['wins'] + stats['draws'] + stats['losses']
    penta = [stats['penta_LL'], stats['penta_LD'], stats['penta_WL'] + stats['penta_DD'],
             stats['penta_WD'], stats['penta_WW']]
    pairs = sum(penta)
    score = (stats['wins'] + stats['draws'] / 2) / n
    variance = sum(count * (i / 4 - score) ** 2 for i, count in enumerate(penta)) / (pairs - 1) if pairs > 1 else None
    se = math.sqrt(variance / pairs) if variance is not None else None
    elo = lambda p: 400 * math.log10(p / (1 - p)) if 0 < p < 1 else None
    return {'games': n, 'pairs': pairs, 'wdl': [stats['wins'], stats['draws'], stats['losses']],
            'penta': penta, 'score': score, 'score_se': se, 'elo': elo(score),
            'elo_95_normal': [elo(max(1e-9, score - 1.96 * se)), elo(min(1-1e-9, score + 1.96 * se))] if se is not None else None}


def read_stats(path, expected_pairs):
    config = json.loads(path.read_text())
    stats = config['stats']
    if len(stats) != 1:
        raise ValueError(f'Expected one matchup, got {list(stats)}')
    key, values = next(iter(stats.items()))
    if key == 'B vs A':
        values = dict(values)
        values['wins'], values['losses'] = values['losses'], values['wins']
        values['penta_WW'], values['penta_LL'] = values['penta_LL'], values['penta_WW']
        values['penta_WD'], values['penta_LD'] = values['penta_LD'], values['penta_WD']
    elif key != 'A vs B':
        raise ValueError(f'Unexpected matchup: {key}')
    result = score_summary(values)
    if result['games'] != 2 * expected_pairs or result['pairs'] != expected_pairs:
        raise ValueError(f'Incomplete fastchess match: {result}, expected {expected_pairs} pairs')
    return result


def run_match(folder, env, cfg, tc, pairs, a_options, b_options, seed, book, versus_master=False, workers=None, cpus=None):
    folder.mkdir(parents=True, exist_ok=True)
    engine_b = env['baseline'] if versus_master else env['candidate']
    cmd = [env['fastchess'], '-engine', f"cmd={env['candidate']}", 'name=A']
    cmd += [f'option.{k}={v}' for k, v in sorted(a_options.items())]
    cmd += ['-engine', f'cmd={engine_b}', 'name=B']
    cmd += [f'option.{k}={v}' for k, v in sorted(b_options.items())]
    cmd += ['-each', 'proto=uci', f'tc={tc}', 'timemargin=0', 'option.Threads=1',
            f"option.Hash={cfg['hash_mb']}", f"option.Move Overhead={cfg['move_overhead_ms']}",
            '-rounds', str(pairs), '-games', '2', '-repeat',
            '-concurrency', str(workers or cfg['effective_workers']), '-force-concurrency',
            '-openings', f'file={book}', 'format=epd', 'order=random', '-srand', str(seed),
            '-draw', 'movenumber=40', 'movecount=8', 'score=10',
            '-resign', 'movecount=3', 'score=600', 'twosided=true',
            '-maxmoves', '300', '-pgnout', 'file=games.pgn', 'append=false', 'nodes=true',
            'nps=true', 'timeleft=true', 'latency=true',
            '-config', 'outname=fastchess.json', '-autosaveinterval', '0',
            '-output', 'format=fastchess', '-report', 'penta=true']
    identity = {'cpu_affinity': sorted(cpus if cpus is not None else os.sched_getaffinity(0)), 'command': cmd, 'binaries': env['binary_sha256'], 'book': sha256(book)}
    if (folder / 'result.json').exists():
        result = json.loads((folder / 'result.json').read_text())
        if result['identity'] != identity:
            raise ValueError(f'Resume mismatch at {folder}')
        return result
    # Interrupted matches restart from their seed; completed matches are never replayed.
    # Preserve the interrupted artifacts for diagnosis rather than mixing partial scores.
    if (folder / 'command.json').exists():
        partial = folder / f'interrupted-{time.time_ns()}'
        partial.mkdir()
        for old in list(folder.iterdir()):
            if old.is_file():
                old.rename(partial / old.name)
    save(folder / 'command.json', identity)
    print(f"  {folder.name}: {tc}, {pairs} pairs, {'master' if versus_master else 'self'}, {workers or cfg['effective_workers']} workers", flush=True)
    start = time.monotonic()
    with open(folder / 'fastchess.log', 'w') as log:
        proc = subprocess.Popen(cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None)
        try:
            while True:
                try:
                    code = proc.wait(timeout=45)
                    break
                except subprocess.TimeoutExpired:
                    print(f'    running: {folder.name}, elapsed {time.monotonic()-start:.0f}s', flush=True)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if code:
        raise RuntimeError(f'fastchess exited {code}; inspect {folder / "fastchess.log"}')
    result = read_stats(folder / 'fastchess.json', pairs)
    result.update(identity=identity, elapsed_seconds=time.monotonic()-start, tc=tc,
                  workers=workers or cfg['effective_workers'],
                  oversubscription=(workers or cfg['effective_workers']) / cfg['workers'])
    # Cross-check games in the PGN, and reject infrastructure failures, never time losses.
    pgn = (folder / 'games.pgn').read_text()
    if len(re.findall(r'^\[Result "(?:1-0|0-1|1/2-1/2)"\]', pgn, re.M)) != 2 * pairs:
        raise ValueError(f'Missing PGN games: {folder}')
    if re.search(r'^\[Termination "(?:abandoned|illegal move|unterminated)"\]', pgn, re.M):
        raise RuntimeError(f'Engine/protocol failure; no SPSA update: {folder}')
    save(folder / 'result.json', result)
    return result


def integer_options(theta, specs):
    return {k: round(specs[k]['min'] + v * (specs[k]['max'] - specs[k]['min'])) for k, v in theta.items()}


def perturb(theta, specs, iteration, cfg):
    rng = random.Random(cfg['seed'] + iteration * 1000003)
    c = cfg['spsa']['perturbation'] / iteration ** cfg['spsa']['gamma']
    plus, minus = {}, {}
    for k, value in theta.items():
        sign = rng.choice((-1, 1))
        plus[k], minus[k] = min(1, max(0, value + sign * c)), min(1, max(0, value - sign * c))
    return integer_options(plus, specs), integer_options(minus, specs)


def update(theta, specs, plus, minus, response, iteration, cfg):
    schedule = cfg['spsa']
    gain = schedule['learning_rate'] * ((schedule['stability'] + 1) / (schedule['stability'] + iteration)) ** schedule['alpha']
    new = {}
    for k, value in theta.items():
        width = specs[k]['max'] - specs[k]['min']
        distance = (plus[k] - minus[k]) / width
        gradient = response / distance if distance else 0
        step = max(-schedule['max_step'], min(schedule['max_step'], gain * gradient))
        new[k] = min(1, max(0, value + step))
    return new


def validate_config(cfg):
    if cfg['iterations'] < 2 or cfg['iterations'] % 2:
        raise ValueError('Use an even number of iterations >= 2 for midpoint verification')
    if not 1 <= cfg['workers'] <= cfg['max_workers'] or not 1 <= cfg['oversubscription'] <= 2:
        raise ValueError('Invalid worker count or oversubscription (supported range 1..2)')
    if cfg.get('physical_cores', cfg['workers']) < 1:
        raise ValueError('Physical-core cap must be positive')
    cfg['effective_workers'] = min(cfg['max_workers'], math.floor(cfg['workers'] * cfg['oversubscription']))
    if cfg['verification_total_pairs'] < 2 * len(cfg['strata']):
        raise ValueError('Verification budget must include every TC at both checkpoints')
    if any(cfg['spsa'][k] <= 0 for k in ('perturbation', 'learning_rate', 'max_step')):
        raise ValueError('SPSA perturbation, learning rate and max step must be positive')
    if cfg['spsa']['stability'] < 0 or cfg['hash_mb'] < 1 or cfg['move_overhead_ms'] < 0:
        raise ValueError('Invalid SPSA stability, hash size or move overhead')
    if len({s['id'] for s in cfg['strata']}) != len(cfg['strata']):
        raise ValueError('Duplicate stratum ids')
    for s in cfg['strata']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', s['id']) or s['pairs'] < 1 or s['weight'] <= 0:
            raise ValueError(f'Invalid stratum: {s}')
    mix = cfg['mix']
    if min(mix.values()) < 0 or abs(sum(mix.values()) - 1) > 1e-9 or set(mix) != {'self', 'master'}:
        raise ValueError('self/master objective weights must be nonnegative and sum to 1')


def split_book(source, out, seed):
    lines = list(dict.fromkeys(Path(source).read_text().splitlines()))
    random.Random(seed).shuffle(lines)
    if len(lines) < 20:
        raise ValueError('Book must contain >=20 distinct positions for a held-out split')
    split = int(len(lines) * .8)
    for name, subset in [('training.epd', lines[:split]), ('validation.epd', lines[split:])]:
        (out / name).write_text('\n'.join(subset) + '\n')


def verify(out, iteration, theta, specs, env, cfg):
    folder = out / f'verify-{iteration:06d}'
    folder.mkdir(exist_ok=True)
    opts = integer_options(theta, specs)
    save(folder / 'parameters.json', opts)
    total = cfg['verification_total_pairs'] // 2
    if iteration == cfg['iterations']:
        total = cfg['verification_total_pairs'] - total
    results = []
    for i, (stratum, pairs) in enumerate(zip(cfg['strata'], allocation(total, cfg['strata']))):
        if pairs:
            # Same validation seeds at both checkpoints; no training positions.
            result = run_match(folder / stratum['id'], env, cfg, stratum['tc'], pairs,
                               opts, {}, cfg['seed'] + 90000000 + i, out / 'validation.epd', True,
                               workers=min(cfg['workers'], cfg['max_workers']))
            results.append({'stratum': stratum['id'], 'weight': stratum['weight'], **result})
    weight_sum = sum(r['weight'] for r in results)
    score = sum(r['weight'] * r['score'] for r in results) / weight_sum
    se = (math.sqrt(sum((r['weight'] / weight_sum * r['score_se']) ** 2 for r in results))
          if all(r['score_se'] is not None for r in results) else None)
    aggregate = {'weighted_score': score, 'weighted_score_se': se,
                 'games': sum(r['games'] for r in results),
                 'wdl': [sum(r['wdl'][i] for r in results) for i in range(3)]}
    save(folder / 'summary.json', {'iteration': iteration, 'pairs': total, 'strata': results,
                                 'aggregate': aggregate,
                                 'note': 'Per-stratum paired normal CIs are descriptive; no strength claim from tiny smoke samples.'})


def run(out, cfg, env):
    specs = options(env['candidate'])
    if not specs:
        raise ValueError('Candidate exposes no TM tuning options')
    for binary in (env['candidate'], env['baseline']):
        options(binary)
    active = {k: v for k, v in specs.items() if k != 'TMPonderBonus' and
              (not k.startswith('TMCyclic') or any('/' in s['tc'] for s in cfg['strata']))}
    active = {k: v for k, v in active.items() if k not in cfg.get('freeze', [])}
    save(out / 'parameters.json', {'all': specs, 'active': list(active),
                                 'inactive': sorted(set(specs) - set(active))})
    if not active:
        raise ValueError('No active TM parameters')
    if cfg.get('async_spsa', {}).get('enabled'):
        # The async runner owns its own state format. Never initialize a
        # synchronous state.json before dispatching to it.
        from async_tune import run_async
        return run_async(out, cfg, env, specs, active)
    state_path = out / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {'iteration': 0, 'theta': {k: (s['default'] - s['min']) / (s['max'] - s['min']) for k, s in active.items()}, 'history': []}
        save(state_path, state)
    for iteration in range(1, cfg['iterations'] + 1):
        # Verification is independently resumable, including a stop just after an update.
        if iteration > state['iteration']:
            theta = state['theta']
            plus, minus = perturb(theta, active, iteration, cfg)
            folder = out / f'iteration-{iteration:06d}'
            folder.mkdir(exist_ok=True)
            save(folder / 'perturbations.json', {'plus': plus, 'minus': minus})
            response = 0.0
            cells = []
            for i, stratum in enumerate(cfg['strata']):
                seed = cfg['seed'] + iteration * 10000 + i
                cell = folder / stratum['id']
                cell.mkdir(exist_ok=True)
                workers = training_workers(cfg, iteration, i)
                # At least two waves of games so requested concurrency can actually
                # be reached. Fastchess naturally has a less-loaded tail at the end.
                pairs = (max(stratum['pairs'], workers) if cfg['effective_workers'] > cfg['workers']
                         else stratum['pairs'])
                save(cell / 'load.json', {'workers': workers, 'pairs': pairs,
                                          'oversubscription': workers / cfg['workers']})
                signal_value = 0.0
                if cfg['mix']['self']:
                    result = run_match(cell / 'self', env, cfg, stratum['tc'], pairs,
                                       plus, minus, seed, out / 'training.epd', workers=workers)
                    signal_value += cfg['mix']['self'] * (result['score'] - .5)
                if cfg['mix']['master']:
                    master_scores = {}
                    # Alternate execution order to reduce systematic thermal/load drift.
                    order = [('plus-master', plus), ('minus-master', minus)]
                    if iteration % 2:
                        order.reverse()
                    for name, opts in order:
                        master_scores[name] = run_match(cell / name, env, cfg, stratum['tc'], pairs,
                                                        opts, {}, seed, out / 'training.epd', True, workers=workers)['score']
                    signal_value += cfg['mix']['master'] * (master_scores['plus-master'] - master_scores['minus-master'])
                response += stratum['weight'] * signal_value / sum(s['weight'] for s in cfg['strata'])
                cells.append({'stratum': stratum['id'], 'response': signal_value,
                              'workers': workers, 'pairs': pairs})
            state['theta'] = update(theta, active, plus, minus, response, iteration, cfg)
            state['iteration'] = iteration
            state['history'].append({'iteration': iteration, 'response': response, 'strata': cells,
                                     'parameters': integer_options(state['theta'], active)})
            save(state_path, state)
            save(out / 'current-options.json', integer_options(state['theta'], active))
            print(f'Iteration {iteration}/{cfg["iterations"]}: response={response:+.6f}', flush=True)
        if iteration in (cfg['iterations'] // 2, cfg['iterations']):
            # Use the historical checkpoint, not the most recent theta on later resumes.
            opts = state['history'][iteration - 1]['parameters']
            historical = {k: (v - active[k]['min']) / (active[k]['max'] - active[k]['min']) for k, v in opts.items()}
            verify(out, iteration, historical, active, env, cfg)
    save(out / 'final-options.json', integer_options(state['theta'], active))
    from summarize import summarize
    summarize(out)
    print(f'Complete: {out}', flush=True)


def main():
    def shutdown(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/kaggle.json')
    parser.add_argument('--output', default='runs/kaggle')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--oversubscription', type=float, help='Maximum training load, varied from 1x up to this value')
    parser.add_argument('--physical-cores', type=int, help='Maximum physical cores; SMT siblings are excluded')
    parser.add_argument('--cpus', help='Restrict selection to assigned Linux CPU IDs, e.g. 0-7,16-23')
    parser.add_argument('--build-jobs', type=int, default=2)
    parser.add_argument('--arch', default='native')
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--candidate', help='Use existing binaries; requires all four path overrides')
    parser.add_argument('--baseline')
    parser.add_argument('--fastchess')
    parser.add_argument('--book')
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.workers is not None:
        cfg['workers'] = args.workers
    if args.oversubscription is not None:
        cfg['oversubscription'] = args.oversubscription
    if args.physical_cores is not None:
        cfg['physical_cores'] = args.physical_cores
    validate_config(cfg)
    cpu_affinity = confine(cfg, args.cpus)
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / '.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Run directory already in use: {out}')
        overrides = [args.candidate, args.baseline, args.fastchess, args.book]
        if any(overrides) and not all(overrides):
            raise ValueError('Supply --candidate, --baseline, --fastchess, --book together')
        if all(overrides):
            env = dict(zip(('candidate', 'baseline', 'fastchess', 'book'), [str(Path(v).resolve()) for v in overrides]))
        else:
            env = setup(min(args.build_jobs, len(cpu_affinity)), args.arch)
        env['cpu_affinity'] = cpu_affinity
        env['binary_sha256'] = {k: sha256(env[k]) for k in ('candidate', 'baseline', 'fastchess')}
        env['book_sha256'] = sha256(env['book'])
        manifest = {'config': cfg, 'environment': env,
                    'pipeline_sha256': {p.name: sha256(p) for p in sorted((ROOT / 'tools').glob('*.py'))},
                    'python': platform.python_version(),
                    'platform': platform.platform(), 'sources': json.loads((ROOT / 'sources.json').read_text())}
        if (out / 'manifest.json').exists():
            old = json.loads((out / 'manifest.json').read_text())
            mismatches = [key for key in ('config', 'environment', 'pipeline_sha256')
                          if old.get(key) != manifest.get(key)]
            if mismatches:
                raise ValueError(f"Resume manifest mismatch in {', '.join(mismatches)} at {out}. "
                                 "Use a genuinely new --output directory or restore the "
                                 "original config/scripts/binaries/book for this run.")
        else:
            save(out / 'manifest.json', manifest)
            split_book(env['book'], out, cfg['seed'])
        if not args.setup_only:
            run(out, cfg, env)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped. Rerun the same command to resume completed matches.', file=sys.stderr)
        sys.exit(130)
