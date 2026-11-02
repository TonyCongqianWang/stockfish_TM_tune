"""Bounded asynchronous, orthogonal SPSA experiments.

Each lane owns a disjoint physical CPU set. Exactly four fastchess matches run
concurrently per experiment, with a fixed per-match game concurrency.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
from pathlib import Path
import random

from tune import integer_options, run_match, save, verify
from summarize import summarize


H4 = ((1, 1, 1, 1), (1, -1, 1, -1),
      (1, 1, -1, -1), (1, -1, -1, 1))


def directions(names, seed):
    """Four uniform-marginal hypercube vectors with |cross dot| <= d % 4."""
    rng = random.Random(seed)
    columns = [tuple(row[j] for row in H4) for _ in range(len(names) // 4) for j in range(4)]
    columns += [tuple(rng.choice((-1, 1)) for _ in range(4))
                for _ in range(len(names) % 4)]
    rng.shuffle(columns)
    columns = [tuple(v * rng.choice((-1, 1)) for v in column)
               for column in columns]
    return [{name: columns[j][i] for j, name in enumerate(names)}
            for i in range(4)]


def schedule(cfg, ordinal):
    """One permutation of every TC x objective x load combination per cycle."""
    levels = cfg['async_spsa']['loads']
    combinations = [(s['id'], objective, load)
                    for s in cfg['strata']
                    for objective in ('self', 'master')
                    for load in levels]
    block, offset = divmod(ordinal, len(combinations))
    random.Random(cfg['seed'] + 81000019 + block).shuffle(combinations)
    return combinations[offset]


def create_experiment(out, cfg, specs, theta, version, ordinal, lane):
    stratum_id, objective, load = schedule(cfg, ordinal)
    folder = out / f'iteration-{ordinal + 1:06d}'
    folder.mkdir(exist_ok=True)
    meta_path = folder / 'experiment.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['ordinal'] != ordinal or meta['lane'] != lane:
            raise ValueError('Experiment identity mismatch')
        return meta
    c = cfg['spsa']['perturbation'] / (ordinal + 1) ** cfg['spsa']['gamma']
    vectors = directions(list(theta), cfg['seed'] + 92000011 + ordinal)
    sampled = []
    for vector in vectors[:4 if objective == 'self' else 2]:
        plus = {k: min(1, max(0, theta[k] + c * vector[k])) for k in theta}
        minus = {k: min(1, max(0, theta[k] - c * vector[k])) for k in theta}
        sampled.append({'direction': vector, 'plus': integer_options(plus, specs),
                        'minus': integer_options(minus, specs)})
    meta = {'ordinal': ordinal, 'lane': lane, 'base_version': version,
            'base_theta': theta.copy(), 'tc_id': stratum_id, 'objective': objective,
            'load': load, 'c': c, 'samples': sampled}
    save(meta_path, meta)
    return meta


def evaluate(out, cfg, env, meta, cpus):
    stratum = next(s for s in cfg['strata'] if s['id'] == meta['tc_id'])
    folder = out / f"iteration-{meta['ordinal'] + 1:06d}"
    seed = cfg['seed'] + 100000 * (meta['ordinal'] + 1)
    jobs = []
    for i, sample in enumerate(meta['samples']):
        if meta['objective'] == 'self':
            jobs.append((i, 'self', sample['plus'], sample['minus'], False))
        else:
            jobs.append((i, 'plus', sample['plus'], {}, True))
            jobs.append((i, 'minus', sample['minus'], {}, True))
    if len(jobs) != 4:
        raise ValueError('Expected exactly four match jobs')
    def play(job):
        i, label, a, b, master = job
        return (i, label, run_match(folder / f'direction-{i}' / label, env, cfg,
                      stratum['tc'], 2, a, b, seed + i, out / 'training.epd',
                      versus_master=master, workers=meta['load'] // 4, cpus=cpus))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [f.result() for f in [pool.submit(play, job) for job in jobs]]
    per_direction = {}
    for i, label, result in results:
        per_direction.setdefault(i, {})[label] = result['score']
    responses = [(s['self'] - 0.5 if meta['objective'] == 'self'
                  else s['plus'] - s['minus'])
                 for i, s in sorted(per_direction.items())]
    return responses


def gradient(meta, responses):
    if len(responses) != len(meta['samples']):
        raise ValueError('Incomplete directional response')
    result = {k: 0.0 for k in meta['base_theta']}
    for sample, response in zip(meta['samples'], responses):
        for k, sign in sample['direction'].items():
            result[k] += response * sign / (2 * meta['c'] * len(responses))
    return result


def apply(theta, specs, grad, cfg, step_number, initial, previous_path):
    schedule = cfg['spsa']
    gain = schedule['learning_rate'] * (
        (schedule['stability'] + 1) / (schedule['stability'] + step_number)
    ) ** schedule['alpha']
    new = {k: min(1, max(0, v + max(-schedule['max_step'],
                                     min(schedule['max_step'], gain * grad[k]))))
           for k, v in theta.items()}
    differences = [new[k] - theta[k] for k in theta]
    step_l1 = sum(abs(x) for x in differences)
    step_l2 = math.sqrt(sum(x*x for x in differences))
    net_l2 = math.sqrt(sum((new[k]-initial[k])**2 for k in theta))
    old_int = integer_options(theta, specs)
    new_int = integer_options(new, specs)
    movement = {'step_l1': step_l1, 'step_l2': step_l2,
                'cumulative_l2': previous_path + step_l2, 'net_l2': net_l2,
                'max_coordinate': max(map(abs, differences), default=0),
                'changed_uci': sum(old_int[k] != new_int[k] for k in theta)}
    return new, movement


def run_async(out, cfg, env, specs, active):
    settings = cfg['async_spsa']
    cpus = env['cpu_affinity']
    if len(cpus) != 16 or cfg['workers'] != 16:
        raise ValueError('Async mode requires exactly 16 selected physical cores')
    if sorted(settings['loads']) != [8, 12, 16]:
        raise ValueError('Async load choices must be 8, 12, 16')
    if settings.get('max_staleness', 4) < 0:
        raise ValueError('Negative maximum staleness')
    lanes = (cpus[:8], cpus[8:])
    state_file = out / 'state.json'
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state.get('mode') != 'async':
            raise ValueError('Cannot resume non-async run in async mode')
    else:
        initial = {k: (s['default'] - s['min']) / (s['max'] - s['min'])
                   for k, s in active.items()}
        state = {'mode': 'async', 'theta': initial.copy(), 'initial_theta': initial,
                 'iteration': 0, 'issued': 0, 'history': [], 'inflight': {},
                 'cumulative_l2': 0.0}
        save(state_file, state)
    target = cfg['iterations']
    checkpoints = {target // 2, target}
    running = {}
    ready = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while state['iteration'] < target:
            # Recreate any launched-but-unapplied experiments after interruption.
            occupied = {meta['lane'] for meta in state['inflight'].values()}
            for key, meta in list(state['inflight'].items()):
                ordinal = int(key)
                if ordinal not in running and ordinal not in ready:
                    running[ordinal] = pool.submit(evaluate, out, cfg, env, meta,
                                                    lanes[meta['lane']])
            # Before checkpoints, do not launch past the barrier. This keeps
            # validation free of concurrent training CPU contention.
            next_checkpoint = min((c for c in checkpoints if c > state['iteration']),
                                  default=target)
            while len(state['inflight']) < 2 and state['issued'] < next_checkpoint:
                free = next(i for i in range(2) if i not in occupied)
                ordinal = state['issued']
                meta = create_experiment(out, cfg, active, state['theta'],
                                         state['iteration'], ordinal, free)
                state['inflight'][str(ordinal)] = meta
                state['issued'] += 1
                save(state_file, state)
                running[ordinal] = pool.submit(evaluate, out, cfg, env, meta, lanes[free])
                occupied.add(free)
            if running:
                done, _ = wait(list(running.values()), return_when=FIRST_COMPLETED)
                for ordinal, future in list(running.items()):
                    if future in done:
                        ready[ordinal] = future.result()
                        del running[ordinal]
            # Keep old gradients: never advance if an unfinished experiment
            # would become older than the configured version-lag bound.
            while ready:
                candidates = sorted(ready, key=lambda o: (state['inflight'][str(o)]['base_version'], o))
                ordinal = candidates[0]
                meta = state['inflight'][str(ordinal)]
                lag = state['iteration'] - meta['base_version']
                if lag > settings.get('max_staleness', 4):
                    raise RuntimeError('Staleness invariant violated')
                other_ages = [state['iteration'] - m['base_version']
                              for key, m in state['inflight'].items() if int(key) != ordinal]
                if other_ages and max(other_ages) >= settings.get('max_staleness', 4):
                    # Only the oldest snapshot can now be applied.
                    oldest = min(state['inflight'],
                                 key=lambda k: state['inflight'][k]['base_version'])
                    if int(oldest) != ordinal:
                        break
                grad = gradient(meta, ready.pop(ordinal))
                old = state['theta']
                new, move = apply(old, active, grad, cfg, state['iteration']+1,
                                  state['initial_theta'], state['cumulative_l2'])
                state['theta'], state['cumulative_l2'] = new, move['cumulative_l2']
                state['iteration'] += 1
                state['history'].append({'iteration': state['iteration'],
                    'experiment': ordinal, 'objective': meta['objective'], 'tc': meta['tc_id'],
                    'load': meta['load'], 'lag': lag, 'movement': move,
                    'parameters': integer_options(new, active)})
                del state['inflight'][str(ordinal)]
                save(state_file, state)
                save(out / 'current-options.json', integer_options(new, active))
                print(f"Update {state['iteration']}/{target} "
                      f"TC={meta['tc_id']} {meta['objective']} load={meta['load']} "
                      f"age={lag} step_L2={move['step_l2']:.6f} "
                      f"net_L2={move['net_l2']:.6f} path_L2={move['cumulative_l2']:.6f} "
                      f"changed_UCI={move['changed_uci']}/{len(new)}", flush=True)
                if state['iteration'] in checkpoints:
                    # At the checkpoint all earlier experiments have been issued.
                    # Only validate once no experiment is running.
                    if not state['inflight']:
                        verify(out, state['iteration'], new, active, env, cfg)
                if state['iteration'] >= target:
                    break
            # A resumed checkpoint may require verification even if the state
            # update was saved immediately before the verification call.
            if (state['iteration'] in checkpoints and not state['inflight']
                    and not (out / f"verify-{state['iteration']:06d}" / 'summary.json').exists()):
                verify(out, state['iteration'], state['theta'], active, env, cfg)
    save(out / 'final-options.json', integer_options(state['theta'], active))
    summarize(out)
