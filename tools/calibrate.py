#!/usr/bin/env python3
"""Fit only the new nominal allocation to a declared master clock grid (no Elo claim)."""
import json
import math
import random


def samples():
    for base in (1.5, 3, 10, 30):
        for ratio in (0, .005, .01, .02, .04):
            for ply, pieces, fraction in ((0, 32, 1), (2, 32, .97), (6, 32, .92), (14, 32, .85), (30, 26, .7), (60, 18, .4), (100, 10, .2)):
                clock, inc, overhead = base * fraction * 1000, base * ratio * 1000, 10
                initial = max(1, base * 1000 + inc * 49 - overhead * 52)
                adjust = .3272 * math.log10(initial) - .4141
                mtg = 50 if clock >= 1000 else int(clock * .05)
                left = max(1, clock + inc * (mtg - 1) - overhead * (2 + mtg))
                constant = min(.0029869 + .00033554 * math.log10(clock / 1000), .004905)
                scale = min(.012112 + (ply + 3.22713) ** .46866 * constant, .19404 * clock / left) * adjust
                yield (clock, inc, ply, pieces, max(1, scale * left))


def budget(row, values):
    clock, inc, ply, pieces, _ = row
    rate, income, floor, plies = values
    bank = max(0, clock - 20)
    return max(1, (bank * rate / (10 + 2 * pieces) + income * (inc - 10))
               * (floor + (1 - floor) * math.tanh(ply / plies)))


def main():
    rows = list(samples())
    def loss(v):
        return sum(math.log(budget(r, v) / r[-1]) ** 2 for r in rows) / len(rows)
    rng = random.Random(20261101)
    best = [1.8, 1, .55, 18]
    bounds = [(.3, 4), (.3, 1.5), (.2, 1), (2, 60)]
    for step in (.3, .1, .03, .01, .003):
        for _ in range(3000):
            proposal = [min(hi, max(lo, x + rng.gauss(0, step * (hi - lo))))
                        for x, (lo, hi) in zip(best, bounds)]
            if loss(proposal) < loss(best):
                best = proposal
    ratios = sorted(budget(r, best) / r[-1] for r in rows)
    print(json.dumps({'grid': '4 bases x 5 increment ratios x 7 phase/clock states; overhead 10ms',
                      'parameters': dict(zip(('TMBankRate', 'TMIncRate', 'TMOpeningFloor', 'TMOpeningPlies'),
                                             [round(x * 1000) for x in best])),
                      'rms_log_error': math.sqrt(loss(best)),
                      'ratio_p10_p50_p90': [ratios[int((len(ratios)-1)*q)] for q in (.1,.5,.9)],
                      'limitation': 'Synthetic states and nominal budgets only; no strength evidence.'}, indent=2))

if __name__ == '__main__':
    main()
