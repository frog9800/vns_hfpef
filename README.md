# MBRL for closed-loop VNS in hypertensive HFpEF


## Correspondence with DropletRunner

| DropletRunner | Here | Note |
|---|---|---|
| BRIO board + droplet | 32-state pulsatile plant, HFpEF phenotype | `hfpef_plant.py` |
| Overhead camera, 64x64x3 | none | no image branch; `encoder.cnn_keys='$^'` |
| 14-D vector: position, tilt, 5 waypoints | 8-D vector: HR, MAP, congestion, refs, previous action, T | `vns_env.py` |
| 20 Hz timer | one cardiac cycle | semi-Markov, decision interval varies |
| 2 motor currents, [-150,150] mA | I in [0,3] mA, f in [0,15] Hz | one-sided, so affine map not plain scaling |
| progress - deviation + terminal bonus | asymmetric tracking + energy + chatter, Eq. (3) | overshoot priced above untreated disease |
| ROS2 nodes | in-process | the plant is a Python object |
| PID waypoint baseline | fulcrum-clamped PI baseline | `policy_runner.py --policy fulcrum` |
| Kalman filter for glare dropouts | none | measurements are exact |
| operator places droplet | `env.reset()` | settled state is cached |

Unchanged: the three-stage loop, the random policy with 0.7 temporal
smoothing, the `episode_XXXX.npz` layout, the `{time}-{uuid}-{successor}-{length}.npz`
chunk format, and the manual offline training loop (`embodied.run.train()`
is for online training only).

## Files

Two entry points, four library modules.

```
train_offline_vns.py   EVERYTHING up to a trained checkpoint
                         [0] reachable set  [1] collect  [2] convert  [3] train
check_results.py       EVERYTHING after: closed-loop eval of each policy,
                       reward decomposition, both figures, summary JSON

hfpef_params.py        Table A1 phenotypes, VNS recruitment, randomization
hfpef_plant.py         plant, VNS coupling, per-cycle stepper, baseline solver
vns_env.py             per-cycle env, reward Eq. (3), shield, reachable set
dreamer_agent.py       shared spaces / config / checkpoint loading
train_dreamer_vns.slurm
```

## Setup

```bash
git clone https://github.com/rajneeshanand/DropletRunner   # for the dreamerv3 fork
export DREAMERV3_PATH=$HOME/cyberrunner/dreamerv3
```

Only `train_offline.py`, and `data_collector.py --mode explore`, and
`policy_runner.py --policy dreamer` import DreamerV3. Everything else runs
on numpy alone, so the plant, the environment, the classical baseline and
the figures all work before DreamerV3 is installed.

## Checking which version you have

Both entry points print `pipeline version` in their banner, and the version
is recorded in `run_manifest.json`, so a checkpoint always says which
pipeline built it. Current: **2026-09-22.6**.

```
2026-09-22.6  reachable set gains a therapeutic ceiling (HR,MAP <= baseline)
2026-09-22.5  Eq.(4) barrier ON by default on true LVEDP, lam_c=2.0
2026-09-22.4  stage 0 runs for every stage, not just --stage all
2026-09-22.3  gradient-steps-per-transition warning
2026-09-22.2  --fresh, --target_hr_mae; replay removed from the checkpoint
2026-09-22.1  first two-script version
```

`.5` and `.6` touch `vns_env.py`, `train_offline_vns.py` and
`check_results.py`, and `.6` invalidates any existing `reachable_set.npz`:
rebuild with `--stage reachable --force`.
`.1` to `.4` touched only `train_offline_vns.py`.

## Which command does what

There is ONE training script. `--stage` selects how far it goes.

| command | plant episodes | GPU training | where it runs |
|---|---|---|---|
| `train_offline_vns.py --stage reachable` | no | no | CPU, minutes |
| `train_offline_vns.py --stage collect` | **yes, this is collect** | no | CPU |
| `train_offline_vns.py --stage train` | no | **yes, this is train** | GPU |
| `train_offline_vns.py --stage all` (default) | yes | yes | needs both |
| `sbatch train_dreamer_vns.slurm` | no | yes | it is `--stage train` |
| `check_results.py` | yes, for evaluation | no | CPU |

So `sbatch train_dreamer_vns.slurm` and `train_offline_vns.py --stage train`
are the same operation. Open the .slurm file and you will see it is one
`python -u train_offline_vns.py --stage train ...` call wrapped in module
loads. Use sbatch on the cluster, the bare command anywhere with a GPU.

Stage 0 runs automatically whenever `reachable_set.npz` is missing,
whichever stage you asked for, because collecting without it produces
episodes whose correct action is no stimulation.

## Procedure

### First run

```bash
#  COLLECT: drives the plant, writes episodes/episode_XXXX.npz. No GPU.
python train_offline_vns.py --stage collect --episodes 50        # ~35 min

#  TRAIN: reads episodes/, fits world model + actor + critic. GPU.
sbatch train_dreamer_vns.slurm                                   # ~45-85 min
#  identical to, without SLURM:
#  python train_offline_vns.py --stage train --steps 50000 --decode_aux

#  CHECK: closed-loop evaluation and both figures. No GPU.
python check_results.py
```

Repeat collect -> train -> check as needed. Each collect appends episodes;
each train resumes from the checkpoint and rebuilds the replay buffer from
everything in `episodes/`.

### More training vs more data

These are not the same thing and they go in opposite directions.

| what you repeat | effect |
|---|---|
| `sbatch` alone, same episodes | gradient steps pile up on fixed data. Degrades past a point. |
| collect + train | episodes and steps grow together. Improves, then saturates. |

`train_offline_vns.py` prints gradient steps per transition each round and
warns above 5 cumulative. 50 episodes x 400 cycles is 20k transitions, so
the default 50k steps is 2.5. Running train three times on the same 50
episodes reaches 7.5 and warns; collecting 50 more each time holds it at
2.5. DropletRunner found 50k-150k steps on 50-100 episodes useful and 450k
on unchanged data harmful.

Reading the diagnostics figure:

- training loss still falling, eval HR MAE flat or rising -> overtraining.
  Collect, do not train longer.
- both flat, action-coverage hexbin dense everywhere -> saturated. More of
  the same data will not help; change the reference distribution or the
  reward weights.
- both flat, hexbin sparse near the fulcrum or the bounds -> under-explored.
  Collect in explore mode, or raise `--sigma` for another random batch.

There is also a floor. References are drawn from the reachable set, so they
are achievable in steady state, but the effector time constants run 2-20 s
against step changes in the reference, so transient error is irreducible.
The fulcrum-clamped PI baseline sits at 0.80 bpm HR MAE on this schedule.
Expect the agent to beat that, not to approach zero.

### If the policy is not good enough

Run the same collect command again. Episodes ACCUMULATE: the collector
numbers from the highest existing `episode_NNNN.npz`, and stage 2 rebuilds
the replay buffer from every episode in the directory. Training resumes from
the existing checkpoint rather than reinitializing, and the step counter
continues. So a second cycle gives 100 episodes and 100k cumulative
gradient steps, not a restart.

With a checkpoint present, `--rounds > 1` switches round 2 onward to
`explore` mode automatically: actions come from the learned policy plus its
own noise, which reaches the fulcrum region far more often than random
actions and puts the new data where the world model is weakest.

### Everything in one process

```bash
python train_offline_vns.py --rounds 3 --episodes 50 --steps 50000 \
    --decode_aux --target_hr_mae 2.0
python check_results.py
```

`--target_hr_mae` runs a short closed-loop check after each round and stops
once the policy is under that error, so the loop terminates on performance
rather than on a fixed round count. Without it the script runs exactly
`--rounds` rounds.

### Starting over

`--fresh` deletes `episodes/`, `logdir/` and `policy/` first. Without it
every invocation adds to what is already there, which is usually what you
want and occasionally not.

### What persists where

| directory | contents | on re-run |
|---|---|---|
| `episodes/` | raw plant episodes, one npz each | appended |
| `logdir/replay/` | embodied chunks | deleted and rebuilt from `episodes/` |
| `logdir/checkpoint.ckpt` | weights + step counter | resumed |
| `logdir/metrics.jsonl` | training curves | appended |
| `policy/` | checkpoint copy + manifest, what deployment reads | overwritten |

## Running

```bash
# laptop, CPU
python train_offline_vns.py --stage collect --episodes 50

# cluster, GPU
sbatch train_dreamer_vns.slurm

# laptop again
python check_results.py
```

Or everything in one process, including the guided-exploration loop:

```bash
python train_offline_vns.py --rounds 2 --episodes 50 --steps 50000 --decode_aux
python check_results.py
```

`train_offline_vns.py` writes `run_manifest.json` next to the checkpoint
recording every setting that must agree between training and deployment
(observation layout, normalization constants, congestion channel, action
bounds). `check_results.py` reads it and drops the dreamer policy from the
comparison rather than silently loading a checkpoint into a different model.

### Exploration

The random policy is AR(1), `a_t = s*a_{t-1} + (1-s)*N(0, sigma^2)`, with
stationary std `(1-s)*sigma/sqrt(1-s^2)`. DropletRunner's `sigma=0.533,
s=0.7` gives 0.224 on a [-1,1] range, which covers only I in [0.71, 2.29] mA
and f in [3.6, 11.4] Hz at 1st-99th percentile: 6.8 % of samples above 10 Hz
and 0.5 % in the high-amplitude, high-frequency corner. The default here is
`sigma=1.2` (stationary std 0.50), which reaches both bounds: 20 % above
10 Hz and 3 % in the corner. Raising sigma rather than lowering s keeps the
temporal correlation the 20 s venous effectors need.

## Congestion in the reward

Equation (4) is ON by default:

    r  <-  r - lam_c * [max(0, LVEDP - p_max)]^2     lam_c = 2.0, p_max = 16 mmHg

Two channels, deliberately independent:

| | flag | default | must be measurable? |
|---|---|---|---|
| what the POLICY SEES | `--congestion` | `papd` | yes |
| what the REWARD CHARGES | `--barrier_channel` | `lvedp` | no |

The reward is only ever evaluated while collecting data and fitting the
world model, so charging it on true LVEDP costs nothing at deployment. Same
asymmetry as the decoder-only aux targets.

### Setting lam_c

Set it by the exchange rate it implies, not directly. `RewardConfig.
exchange_rate()` returns the bpm of heart-rate error that one mmHg of LVEDP
is priced at. Over the admissible action set LVEDP spans only
15.69-16.85 mmHg, so:

| lam_c | 1 mmHg LVEDP costs | changes the chosen action? |
|---|---|---|
| 1.0 | 12 bpm | no |
| **2.0** | **23 bpm** | **no (default)** |
| 5.0 | 59 bpm | yes: +6.1 bpm HR error for 0.20 mmHg |
| 10.0 | 117 bpm | yes: +3.8 bpm HR error for 0.36 mmHg |
| 50-100 | - | flips SUB-FULCRUM (tachycardia hack) |

The barrier only redirects the policy at roughly 30+ bpm per mmHg, which is
not a trade a clinician would take. It CAN reach zero, by two routes, and
both are non-therapeutic:

| route | action | LVEDP | cost |
|---|---|---|---|
| overdose | 3.0 mA / 15 Hz | 14.39 (-2.08) | MAP 47.9, HR 33.7 |
| tachycardia | 0.75 mA / 15 Hz | 15.69 (-0.79) | HR 92.9, above baseline |

The first is blocked by the MAP tracking term, the shield rolloff and
termination at MAP < 60: commanded closed-loop with the shield on it lands
at 1.82 mA applied, MAP 80.6, LVEDP only -0.33; with the shield off the
episode terminates at cycle 2. The second passes both safety floors and is
blocked only by HR tracking, which is why the reference set now carries a
therapeutic ceiling.

So decongestion is not unreachable. Every route to it costs more than it
returns at any defensible exchange rate. That is Section 5 measured rather
than asserted, and it is the stronger claim.

Above lam_c ~50 the agent finds the sub-fulcrum route: tachycardia shortens
diastole, which lowers filling pressure without treating anything. The HR
tracking term is what blocks it. Re-run the sweep if you raise lam_c.

`--no_barrier` drops the term entirely.

## Congestion in the observation

`--congestion {papd,pla,lvedp}` selects which congestion signal enters the
8-D encoder input.

- `papd` (default) is pulmonary artery diastolic pressure, what a
  CardioMEMS-class implantable PA sensor reports. In this plant it tracks
  LVEDP to within 0.02 mmHg in both phenotypes, so the policy can observe
  congestion and still be deployable.
- `pla` and `lvedp` are simulation-only. A policy trained on either needs a
  sensor that no chronic implant provides.

Whichever channel is observed, the full hidden set (LVEDP, Pla, PAPd, PAPm,
EF, CO, EDV, ESV, Emax) is always written to `aux` and always read from the
plant during evaluation, so a run can be audited against true filling
pressure regardless of what the policy saw.

`--congestion_barrier` adds Equation (4) on the observed channel. It is a
separate switch on purpose: observing congestion changes what the agent
knows, rewarding it changes what the agent wants, and the two effects should
be measured apart.

`encoder.mlp_keys` is still `vector` and nothing else. `aux` may appear in
`decoder.mlp_keys` (`--decode_aux`) and never in the encoder. Nothing in
DreamerV3 warns you; the run trains fine and the result is undeployable.

## Notes that carry into the results

- The decision step costs about 44 ms of plant integration here, not the
  0.77 s quoted in the qualifying report. At that cost 1e6 decision steps is
  roughly 12 h rather than 200 h, so the "simulator too slow to train
  against directly" argument is weaker than stated and should be rechecked
  before it goes in a paper. The world model is still worth having, for
  belief-state estimation under partial observability rather than for speed.

- Actions stored in the episode files are the APPLIED actions, after the
  safety shield. The world model is therefore fit to the shielded system,
  which is the system that gets deployed.

- The 15 Hz bound exceeds the 10 Hz additivity limit that Table A2 cites
  Sadashivaiah et al. for. Above ~10 Hz evoked and physiological spikes
  collide and the additive form of Eq. (2) overstates the delivered rate, so
  the plant is optimistic about actuator authority in 10-15 Hz. See the note
  in `hfpef_params.py` for the two ways to fix it.

- Episodes store raw physiology (`meas`, `aux`), not only the scalar reward,
  so the reward can be redefined offline. Adding the congestion barrier of
  Eq. (4) later needs a relabeling pass, not a new data-collection campaign.

- References are drawn from the reachable set, the image of the action set
  under the plant's steady-state map, intersected with the therapeutic
  window (MAP >= 90 mmHg, HR >= 55 bpm). Sampling HR and MAP independently
  from a box puts most of the probability mass on targets no action can
  reach, since VNS pulls both down together.
