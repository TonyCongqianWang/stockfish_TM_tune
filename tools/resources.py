"""Linux CPU confinement and deterministic, balanced load variation."""
import math
import os
from pathlib import Path
import random


def parse_cpus(value):
    cpus = set()
    for item in value.split(','):
        bounds = item.split('-')
        first, last = int(bounds[0]), int(bounds[-1])
        if len(bounds) > 2 or first < 0 or last < first:
            raise ValueError('Use a CPU list such as 0-7,16-23')
        cpus.update(range(first, last + 1))
    return cpus


def select_cores(allowed, limit, topology_root=Path('/sys/devices/system/cpu')):
    """Use one permitted logical CPU per physical package/core pair; exclude SMT siblings."""
    representatives = {}
    for cpu in sorted(allowed):
        topology = topology_root / f'cpu{cpu}' / 'topology'
        try:
            key = tuple(int((topology / name).read_text()) for name in ('physical_package_id', 'core_id'))
        except (OSError, ValueError) as error:
            raise ValueError(f'Cannot determine physical topology for CPU {cpu}; refusing an unverified core cap') from error
        if min(key) < 0:
            raise ValueError(f'Invalid physical topology for CPU {cpu}')
        representatives.setdefault(key, cpu)
    return sorted(representatives.values())[:limit]


def confine(cfg, requested=None):
    allowed = set(os.sched_getaffinity(0))
    if requested is not None:
        selected = parse_cpus(requested)
        if not selected <= allowed:
            raise ValueError('Requested CPUs extend outside the current scheduler/affinity allocation')
        allowed = selected
    cpus = select_cores(allowed, min(cfg.get('physical_cores', cfg['workers']), cfg['workers']))
    if not cpus:
        raise ValueError('No physical cores available')
    os.sched_setaffinity(0, cpus)
    cfg['workers'] = min(cfg['workers'], len(cpus))
    cfg['effective_workers'] = min(cfg['max_workers'], math.floor(cfg['workers'] * cfg['oversubscription']))
    print(f"CPU affinity: {cpus} ({len(cpus)} physical cores, one hardware thread each); "
          f"training concurrency {cfg['workers']}..{cfg['effective_workers']}", flush=True)
    return cpus


def training_workers(cfg, iteration, stratum_index):
    # Each shuffled block visits every allowed integer concurrency once. Same
    # result on resume, and same load for all perturbation/match-type comparisons.
    levels = list(range(cfg['workers'], cfg['effective_workers'] + 1))
    index = (iteration - 1) * len(cfg['strata']) + stratum_index
    block, offset = divmod(index, len(levels))
    random.Random(cfg['seed'] + 70000001 + block).shuffle(levels)
    return levels[offset]
