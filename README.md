# Stockfish time-management SPSA

Clone this repository and run `./run.sh`. It builds pinned Stockfish candidate and
master binaries, builds fastchess, verifies and unpacks the opening book, then
runs a small two-worker Kaggle experiment. Python 3's standard library is enough;
there are no pip dependencies. Internet access is needed for initial setup.

```bash
git clone https://github.com/TonyCongqianWang/stockfish_TM_tune.git
cd stockfish_TM_tune
./run.sh --config configs/kaggle.json --output runs/kaggle
```

Enable Internet in Kaggle's notebook settings and select a CPU session. The
included `kaggle.ipynb` contains these commands. `--build-jobs 2` is the default.
Kaggle caps concurrent games at **two**, even if oversubscription is requested.
Each engine uses one thread. Fastchess keeps two engine processes per game, but
only the side to move searches. The limit is two concurrent games/searches.

The Kaggle preset is an integration check, **not a useful tune or Elo test**:

- Two SPSA iterations, using 1.5+0, 1.5+0.015, and 30+0.3 seconds.
- Each iteration: five self-play pairs and ten pairs against master.
- One validation pair per TC at halfway and at the end: **six verification pairs total**.
- In all, 36 pairs / 72 games. Setup and runtime depend on the CPU; 30+0.3 dominates.

For the shortest wiring test, use `configs/smoke.json`: 1.5+0.015 only, two
iterations and **eight pairs / sixteen games total**.

```bash
./run.sh --config configs/smoke.json --output runs/smoke
```

## Cluster preset

```bash
./run.sh --config configs/cluster.json --output runs/cluster --build-jobs 2
```

This exploratory preset uses 16 concurrent games, 400 SPSA iterations, and 20 TCs:
base times 1.5, 3, 10, and 30 seconds crossed with increment ratios 0%, 0.5%, 1%,
2%, and 4%. Thus 30+0.3 is included, along with 30+0 and 30+1.2.
The preset has **30,000 verification pairs across both checkpoints and all TCs**:
15,000 at iteration 200 and 15,000 at iteration 400, or 750 pairs per TC per check.
Set `verification_total_pairs` to 60000 if you want 30,000 at each checkpoint.
These are fixed-size tournaments against the pinned upstream master, not SPRTs.
Validation uses nominal worker count, without oversubscription.

Training pairs per TC per iteration are 32 / 24 / 16 / 8 for bases
1.5 / 3 / 10 / 30. With both match types, that is 1,200 pairs per iteration,
480,000 training pairs in the preset. These batch sizes and SPSA gains are starting
values, **not variance-calibrated recommendations**. Benchmark a pilot before a
full cluster run; the Kaggle run only verifies orchestration.

```bash
# Explicit stress experiment; keep separate from your nominal-speed run.
./run.sh --config configs/cluster.json --output runs/cluster-2x --oversubscription 2
```

Oversubscription permits up to 32 concurrent games on the 16-worker preset. It
changes scheduling latency and playing conditions as well as NPS. It is not a
clean equivalent of slower hardware. The runner records the setting; do not mix
such conditions silently or compare their raw scores as if the conditions matched.

## Objective and SPSA

Every update includes all configured TC strata. The `weight` fields define the
objective, independently of game counts. Faster TCs get more samples without
getting more objective weight. The default mix is 50% self-play, 50% master:

1. Play `theta+` against `theta-` with reversed colors per opening pair.
2. Play each perturbation against pinned master using identical opening seeds,
   colors, TCs and pair counts. Alternate which perturbation is tested first.
3. Combine `self_score - 0.5` and `plus_vs_master_score - minus_vs_master_score`,
   with the configured mix and stratum weights. Divide by each parameter's signed
   normalized perturbation distance to estimate its gradient.

The self-play response has the same first-order scale as the difference of master
scores under a local logistic strength model. This is a local approximation;
nontransitivity can make the two objectives disagree. We retain the constituent
match results so that disagreement can be inspected.

Parameters use normalized UCI ranges, a diminishing perturbation schedule,
projected gradient ascent and a per-update movement cap. Seeds and perturbations
are deterministic. Learning rates, perturbation widths and batch sizes must be
assessed on a pilot; they are not inferred from the smoke test.

All **43** allocation and dynamic time-use coefficients are exposed by the engine.
The runner discovers defaults and bounds from UCI. With the supplied sudden-death /
increment TCs, 37 are active. Five `TMCyclic*` parameters activate automatically if
a `moves/time` TC is added. `TMPonderBonus` is inactive because pondering is off.
Tuning coefficients with no effect would add random drift; they remain available
in UCI. `freeze` can explicitly exclude names from a run. Protocol safety guards,
minimum positive budgets, the 50-move cyclic horizon limit, mate stopping, and the
single-legal-move viewer cap are structural rules rather than SPSA coefficients.

## Model and baseline

The engine branch is `time-management-rewrite` in the Stockfish fork. It starts
directly from upstream `ff7cee4244ec96ea488c402669e3674c0574dce6`.
`sources.json` pins full candidate, baseline and fastchess revisions and a
checksummed official opening archive. Nothing follows a moving branch during a run.
Both engines use the same NNUE architecture and default network.

The new nominal budget is opening-adjusted net increment plus current bank draw
across a material-dependent horizon. Renewable income share controls both the
maximum multiplier and the exponent of the existing search-difficulty multiplier.
There is no initial-clock state, machine-speed estimator, or absolute-TC sigmoid.
The dynamic search coefficients initially match master; responsiveness defaults
to one. Cyclic allocation initially matches master. Maximum time remains subject
to a current-clock cap. Safety/clock-cap coefficients are tunable within bounds;
positive-budget and cap-ordering guards remain fixed.

`python3 tools/calibrate.py` reproduces `configs/calibration.json`. It fits four
nominal-allocation defaults to master's formula on a declared synthetic grid,
not game outcomes. Median new/master nominal budget is about 0.96, and the
10th–90th percentile range is about 0.71–1.37. This is an approximate behavioral
baseline, not demonstrated playing-strength parity. Scaling time can change
master's allocation even at the same relative increment; the new allocation is
scale-invariant apart from overhead and rounding, so it cannot exactly match
master at every TC.

## Outputs, resume and analysis

Rerun the **same command with the same output directory** to resume. A lock prevents
two orchestrators from using a run directory. Configuration, script, binary and book
fingerprints must match. Each completed fastchess match has an atomic result file;
SIGINT/SIGTERM stop the active match; interrupted matches are archived and restarted from their seed. SPSA updates are
atomic, and validation is independently resumable. A whole update is applied only
after every required match completes. No automatic crash recovery converts broken
engine/protocol matches into tuning evidence; real time forfeits remain results.

Opening positions are deduplicated and split deterministically into 80% training
and 20% held-out validation. Validation uses the same seeds at both checkpoints.
The two checks share positions, so their scores are correlated. The validation set
is held out from SPSA, but repeated human selection based on it can still overfit.

Outputs include:

- `manifest.json`: resolved configuration, pinned sources, binary/book hashes.
- `state.json`, `current-options.json`, `final-options.json`: resume state and UCI values.
- `iteration-*/`: perturbations, exact fastchess argv, logs, PGNs and results per TC/mode.
- `verify-*/summary.json`: per-TC WDL/pentanomial statistics and paired normal CIs,
  plus weighted aggregate score and uncertainty. Tiny test CIs are not meaningful.
- `summary.json`: per-match results and move-time/remaining-clock/NPS averages by
  stage, TC, matchup, engine and game-ply bucket; termination counts.

PGNs and logs stay uncompressed throughout tuning. At completion, the summarizer
streams the PGNs into averages, then retains the raw files as `.pgn.gz` and `.log.gz`.
Re-running summary is safe: `python3 tools/summarize.py runs/kaggle`.
Names A/B mean plus/minus in self-play, candidate/master in master games.

Override all four paths (`--candidate`, `--baseline`, `--fastchess`, `--book`) to
use existing local builds without setup. Use `--setup-only` to fetch and build
without playing. `--arch native` is default; builds are CPU-specific. System
packages are installed only when missing and root/passwordless sudo is available;
otherwise the entry point lists required packages. No shell environment or global
Python environment is changed. On clusters without outbound Internet, stage the
cache on a connected machine or supply local paths.

## Validation

```bash
python3 -m unittest discover -s tests -v
bash -n run.sh
# After setup/build, also check the compiled model:
python3 tools/check_engine.py
```

The tests cover paired-score variance, orientation of fastchess results, incomplete
matches, exact verification budgets, SPSA update direction/bounds, worker limits,
and parsing/compression of fastchess PGNs. A real 16-game fastchess smoke run covers
two iterations, both matchup types, both checkpoints, compression and resume.

Stockfish is GPLv3; fastchess is MIT licensed. Opening books retain their upstream
licenses. Dependencies are fetched rather than vendored; see their repositories.
