#!/usr/bin/env python3
"""
======================================================================
 OFFLINE TRAINING PIPELINE  --  closed-loop VNS in hypertensive HFpEF
======================================================================

One script, four stages, following the DropletRunner method:

  0  reachable set   open-loop sweep of the action set; the therapeutic
                     window is intersected with it to give references that
                     some constant action can actually reach. Cached.
  1  collect         episodes on the plant. Round 0 uses the random policy;
                     later rounds use the learned checkpoint in 'explore'
                     mode, which reaches the interesting part of the
                     dose-response far more often than random actions do.
  2  convert         episode_XXXX.npz -> embodied replay chunks
  3  train           manual offline loop over the replay buffer. The agent
                     never touches the plant during this stage.

With --rounds > 1 stages 1-3 repeat, resuming from the existing checkpoint
rather than reinitializing, which is DropletRunner's Stage 3 feedback.

    python train_offline_vns.py                        # defaults
    python train_offline_vns.py --episodes 50 --steps 50000 --rounds 2
    python train_offline_vns.py --stage collect        # data only, no GPU
    python train_offline_vns.py --stage train          # train on existing data

A run_manifest.json is written next to the checkpoint recording every
setting that has to agree between training and deployment (observation
layout, normalization constants, congestion channel, action bounds).
check_results.py reads it and refuses to run silently on a mismatch.
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*using stateful random seeds*")
warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

import numpy as np

import hfpef_params as HP
import vns_env as VE

PIPELINE_VERSION = "2026-09-22.6"
# 2026-09-22.6  reachable set gains a therapeutic ceiling (HR,MAP <= baseline)
# 2026-09-22.5  Eq.(4) barrier ON by default on true LVEDP, lam_c=2.0
# 2026-09-22.4  stage 0 runs for every stage, not just --stage all
# 2026-09-22.3  gradient-steps-per-transition warning
# 2026-09-22.2  --fresh, --target_hr_mae; replay removed from the checkpoint
# 2026-09-22.1  first two-script version


# ===================================================================
# Stage 1: exploration policy
# ===================================================================

class RandomPolicy:
    """
    AR(1) random exploration, a_t = s*a_{t-1} + (1-s)*N(0, sigma^2).

    The correlation matters for the same reason it does in DropletRunner:
    independent draws average to nothing over the effector time constants
    (2 s sympathetic to 20 s venous against a ~0.75 s decision interval), so
    an uncorrelated action sequence is invisible to them and the world model
    never sees the causal link between a sustained dose and the response.

    sigma is raised from DropletRunner's 0.533 because the two action
    ranges are used differently. The AR(1) stationary std is

        std = (1-s)*sigma / sqrt(1-s^2)

    and at s=0.7, sigma=0.533 that is 0.224 on a [-1,1] range: only 6.8 % of
    samples exceed 10 Hz and 0.5 % land in the high-amplitude, high-frequency
    corner. For the droplet that was fine, the interesting dynamics being
    mid-range. Here the fulcrum sits at 1.33 mA and the therapeutic edge sits
    near the bounds, so sigma=1.2 (stationary std 0.50, roughly full range at
    2 sigma) covers them. Raising sigma rather than lowering s keeps the
    temporal correlation intact.
    """

    def __init__(self, rng, sigma=1.2, smoothing=0.7):
        self.rng, self.sigma, self.s = rng, sigma, smoothing
        self.prev = np.zeros(2, dtype=np.float32)

    @property
    def stationary_std(self):
        return (1 - self.s) * self.sigma / np.sqrt(1 - self.s ** 2)

    def reset(self):
        self.prev = np.zeros(2, dtype=np.float32)

    def __call__(self, *_, **__):
        raw = self.rng.standard_normal(2).astype(np.float32) * self.sigma
        a = np.clip(self.s * self.prev + (1 - self.s) * raw, -1, 1).astype(np.float32)
        self.prev = a
        return a


def collect_episode(env, policy, seed, max_cycles):
    """
    One episode. Alignment follows embodied: index k holds the observation
    reached at step k, the reward received on arriving there, and the action
    taken from it.
    """
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    V, A, R, F, TM, X, RF, M, S, TR = [], [], [], [], [], [], [], [], [], []
    reward, terminal = 0.0, False

    for k in range(max_cycles):
        m = info["meas"]
        V.append(obs); R.append(reward); F.append(k == 0); TM.append(False)
        X.append(info["aux"]); RF.append(info["ref"])
        M.append([m["HR"], m["MAP"], m["SBP"], m["DBP"], m["T"]])
        S.append(info["shielded"])
        TR.append([info["terms"][t] for t in VE.REWARD_TERMS])

        a = policy(obs, reward, k == 0)
        obs, reward, terminal, info = env.step(a)
        A.append(info["applied"])          # store what the plant actually got
        if terminal:
            break

    TM[-1] = True
    return dict(
        vector=np.asarray(V, np.float32), action=np.asarray(A, np.float32),
        reward=np.asarray(R, np.float32), is_first=np.asarray(F, bool),
        is_terminal=np.asarray(TM, bool), aux=np.asarray(X, np.float32),
        ref=np.asarray(RF, np.float32), meas=np.asarray(M, np.float32),
        shielded=np.asarray(S, bool), terms=np.asarray(TR, np.float32),
    )


def stage_collect(args, env, mode, round_idx):
    os.makedirs(args.episodes_dir, exist_ok=True)
    start = len(glob.glob(os.path.join(args.episodes_dir, "episode_*.npz")))

    if mode == "explore":
        from dreamer_agent import load_agent, AgentPolicy
        print(f"  loading checkpoint -> EXPLORE mode")
        agent, _ = load_agent(args.policy_dir, VE.OBS_DIM, VE.ACT_DIM,
                              VE.AUX_DIM if args.decode_aux else None)
        policy = AgentPolicy(agent, VE.AUX_DIM if args.decode_aux else None,
                             mode="explore")
    else:
        policy = RandomPolicy(np.random.default_rng(args.seed + 977 * round_idx),
                              sigma=args.sigma, smoothing=args.smoothing)
        print(f"  RANDOM mode: sigma={args.sigma}, smoothing={args.smoothing}, "
              f"stationary std={policy.stationary_std:.3f}")

    t0 = time.time()
    allI, allf = [], []
    for i in range(args.episodes):
        idx = start + i
        ep = collect_episode(env, policy, seed=args.seed * 10000 + idx,
                             max_cycles=args.max_cycles)
        np.savez_compressed(
            os.path.join(args.episodes_dir, f"episode_{idx:04d}.npz"), **ep)
        I = 0.5 * HP.I_MAX * (ep["action"][:, 0] + 1)
        f = 0.5 * HP.F_MAX * (ep["action"][:, 1] + 1)
        allI.append(I); allf.append(f)
        print(f"  episode_{idx:04d}  {len(ep['reward']):4d} cyc  "
              f"return={ep['reward'].sum():9.1f}  "
              f"HR {ep['meas'][:,0].min():5.1f}-{ep['meas'][:,0].max():5.1f}  "
              f"MAP {ep['meas'][:,1].min():6.1f}-{ep['meas'][:,1].max():6.1f}  "
              f"LVEDP {ep['aux'][:,0].min():5.2f}-{ep['aux'][:,0].max():5.2f}")

    I, f = np.concatenate(allI), np.concatenate(allf)
    print(f"\n  {args.episodes} episodes in {(time.time()-t0)/60:.1f} min")
    print(f"  action coverage:  I p1-p99 {np.percentile(I,1):.2f}-"
          f"{np.percentile(I,99):.2f} mA   f p1-p99 {np.percentile(f,1):.2f}-"
          f"{np.percentile(f,99):.2f} Hz")
    print(f"                    {100*np.mean(I>2.0):.1f}% above 2 mA, "
          f"{100*np.mean(f>10):.1f}% above 10 Hz, "
          f"{100*np.mean((I>2.0)&(f>10)):.1f}% in both")


# ===================================================================
# Stage 2: conversion
# ===================================================================

def stage_convert(args):
    replay_dir = os.path.join(args.logdir, "replay")
    os.makedirs(replay_dir, exist_ok=True)
    for old in glob.glob(os.path.join(replay_dir, "*.npz")):
        os.remove(old)

    src = sorted(glob.glob(os.path.join(args.episodes_dir, "episode_*.npz")))
    if not src:
        raise FileNotFoundError(f"no episode_*.npz in {args.episodes_dir}")

    total = 0
    for i, path in enumerate(src):
        d = np.load(path)
        n = len(d["reward"])
        ep = {
            "vector": d["vector"].astype(np.float32),
            # already in [-1,1]: DropletRunner divides by 150 here because it
            # stores raw mA. The affine map lives in vns_env.to_physical and
            # nowhere else, so this is the identity and the clip is a guard.
            "action": np.clip(d["action"].astype(np.float32), -1, 1),
            "reward": d["reward"].astype(np.float32),
            "is_first": d["is_first"].astype(bool),
            "is_terminal": d["is_terminal"].astype(bool),
            "is_last": d["is_terminal"].astype(bool).copy(),
        }
        if args.decode_aux:
            ep["aux"] = d["aux"].astype(np.float32)
        # chunk name: {time}-{uuid}-{successor}-{length}.npz
        np.savez(os.path.join(replay_dir, f"{i:016d}-{i:016d}-{'0'*16}-{n}.npz"),
                 **ep)
        total += n
    print(f"  {len(src)} episodes -> {total} transitions in {replay_dir}")
    return total


# ===================================================================
# Stage 3: offline training
# ===================================================================

def stage_train(args):
    from dreamer_agent import add_dreamer_to_path, spaces, make_config
    add_dreamer_to_path(args.dreamer_path)
    import embodied
    from dreamerv3 import agent as agt

    aux_dim = VE.AUX_DIM if args.decode_aux else None
    config = make_config(args.logdir, VE.OBS_DIM, VE.ACT_DIM, aux_dim,
                         batch_size=args.batch_size,
                         batch_length=args.batch_length, size=args.size)
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / "config.yaml")

    print(f"  encoder.mlp_keys : {config.encoder.mlp_keys}")
    print(f"  decoder.mlp_keys : {config.decoder.mlp_keys}"
          + ("   <- aux is decoder-only" if aux_dim else ""))

    obs_space, act_space = spaces(VE.OBS_DIM, VE.ACT_DIM, aux_dim)
    replay = embodied.replay.Uniform(length=config.batch_length,
                                     capacity=config.replay_size,
                                     directory=os.path.join(args.logdir, "replay"))
    need = config.batch_size * config.batch_length
    print(f"  replay: {len(replay)} steps (need >= {need})")
    if len(replay) < need:
        print("  ERROR: not enough data. Collect more episodes or lower "
              "--batch_length.")
        sys.exit(1)

    step = embodied.Counter()
    agent = agt.Agent(obs_space, act_space, step, config)
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
    ])
    # The checkpoint carries the STEP COUNTER and the NETWORK WEIGHTS only,
    # deliberately not the replay buffer. DropletRunner registers the replay
    # too, which is safe online where the buffer only grows. Here stage 2
    # deletes and rewrites every chunk file each round, so a restored replay
    # state would point at item IDs that no longer exist. The buffer is fully
    # determined by episodes/ on disk, so checkpointing it is redundant as
    # well as unsafe. Consequence: resuming reloads weights and continues the
    # step count, and re-reads the whole accumulated dataset from scratch.
    ckpt = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    ckpt.step = step
    ckpt.agent = agent
    ckpt.load_or_save()
    if int(step) > 0:
        print(f"  resumed from step {int(step)} (weights kept, "
              f"replay rebuilt from {len(replay)} steps on disk)")

    # Gradient steps per transition. DropletRunner's ablation found
    # 50k-150k steps on 50-100 episodes useful and 450k on the same 100
    # episodes actively harmful: past some ratio the world model memorizes
    # individual episodes instead of generalizing. 50 episodes x 400 cycles
    # is 20k transitions here, so the default 50k steps is 2.5 per
    # transition, and a second collect+train cycle keeps it there. Running
    # train repeatedly WITHOUT collecting is what pushes it up.
    n_tr = max(len(replay), 1)
    ratio = args.steps / n_tr
    cum = (int(step) + args.steps) / n_tr
    print(f"  gradient steps per transition: {ratio:.2f} this round, "
          f"{cum:.2f} cumulative")
    if cum > 5.0:
        print("  *** WARNING: cumulative ratio above 5. Past this point more")
        print("      gradient steps on the same data degrade the policy.")
        print("      Collect more episodes instead:")
        print("        python train_offline_vns.py --stage collect --episodes 50")

    dataset = agent.dataset(replay.dataset)
    state, metrics = None, embodied.Metrics()
    last_log, t0 = time.time(), time.time()
    save_every = max(args.steps // 10, 1)

    for i in range(args.steps):
        batch = next(dataset)
        _, state, mets = agent.train(batch, state)
        metrics.add(mets, prefix="train")
        step.increment()
        if time.time() - last_log >= config.run.log_every:
            agg = metrics.result()
            rep = {k: v for k, v in agent.report(batch).items()
                   if "train/" + k not in agg}
            logger.add(agg); logger.add(rep, prefix="report")
            logger.add(replay.stats, prefix="replay"); logger.write(fps=True)
            last_log = time.time()
            el = time.time() - t0
            print(f"  step {i+1}/{args.steps} ({100*(i+1)/args.steps:.1f}%) "
                  f"[{el/60:.1f} min, eta {el/(i+1)*(args.steps-i-1)/60:.1f} min]")
        if (i + 1) % save_every == 0 or i == args.steps - 1:
            ckpt.save()
    ckpt.save(); logger.write()
    print(f"  done in {(time.time()-t0)/60:.1f} min")

    os.makedirs(args.policy_dir, exist_ok=True)
    import shutil
    for f in ("checkpoint.ckpt", "config.yaml"):
        p = os.path.join(args.logdir, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(args.policy_dir, f))
    print(f"  checkpoint -> {args.policy_dir}/")


# ===================================================================
# manifest
# ===================================================================

def write_manifest(args, cfg, env):
    man = dict(
        pipeline_version=PIPELINE_VERSION,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        phenotype=args.phenotype,
        congestion=args.congestion,
        congestion_deployable=VE.CONGESTION_DEPLOYABLE[args.congestion],
        congestion_barrier=cfg.congestion_barrier,
        barrier_channel=args.barrier_channel,
        decode_aux=args.decode_aux,
        obs_dim=VE.OBS_DIM, act_dim=VE.ACT_DIM, aux_dim=VE.AUX_DIM,
        aux_keys=VE.AUX_KEYS,
        obs_loc=VE.OBS_LOC.tolist(), obs_scale=VE.OBS_SCALE.tolist(),
        I_max=HP.I_MAX, f_max=HP.F_MAX,
        reward={k: getattr(cfg.reward, k) for k in
                ("hr_tol", "map_tol", "overshoot_ratio", "lam_u", "lam_d",
                 "lam_c", "p_max", "lam_f", "f_soft")},
        shield=dict(enabled=cfg.shield.enabled, hr_floor=cfg.shield.hr_floor,
                    map_floor=cfg.shield.map_floor, f_cap=cfg.shield.f_cap),
        max_cycles=args.max_cycles, episodes_per_round=args.episodes,
        rounds=args.rounds, steps=args.steps, sigma=args.sigma,
        smoothing=args.smoothing, randomize=args.randomize,
        baseline={k: float(v) for k, v in env.base.items()},
        reachable_points=(0 if env.reach is None else int(len(env.reach))),
    )
    os.makedirs(args.policy_dir, exist_ok=True)
    for d in (args.logdir, args.policy_dir):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "run_manifest.json"), "w") as fh:
            json.dump(man, fh, indent=2)
    return man


def eval_round(args, env):
    """Short closed-loop check between rounds. Returns HR MAE in bpm."""
    from check_results import load_dreamer, rollout, summarize
    pol = load_dreamer(args.policy_dir, args.decode_aux)
    saved, env.cfg.max_cycles = env.cfg.max_cycles, args.eval_cycles
    try:
        r = rollout(env, pol, args.eval_cycles, seed=12345)
    finally:
        env.cfg.max_cycles = saved
    return summarize(r)["hr_mae"]


# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=["all", "reachable", "collect", "train"],
                    default="all")
    ap.add_argument("--rounds", type=int, default=1,
                    help="collect/train cycles; round>0 collects in explore mode")
    # plant and problem
    ap.add_argument("--phenotype", default="hfpef")
    ap.add_argument("--congestion", choices=["papd", "pla", "lvedp"],
                    default="papd")
    ap.add_argument("--barrier_channel", choices=["lvedp", "pla", "papd"],
                    default="lvedp", help="congestion signal the REWARD "
                    "charges; need not be measurable")
    ap.add_argument("--no_barrier", action="store_true",
                    help="drop Equation (4) from the reward")
    ap.add_argument("--lam_c", type=float, default=None,
                    help="barrier weight; see RewardConfig for the exchange "
                         "rate each value implies")
    ap.add_argument("--p_max", type=float, default=None,
                    help="barrier ceiling in mmHg (default 16, HFA-PEFF)")
    ap.add_argument("--max_cycles", type=int, default=400)
    ap.add_argument("--randomize", action="store_true",
                    help="domain randomization per episode")
    # collection
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--smoothing", type=float, default=0.7)
    # training
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--batch_length", type=int, default=64)
    ap.add_argument("--size", default="small")
    ap.add_argument("--decode_aux", action="store_true",
                    help="reconstruct aux as a DECODER-ONLY target")
    ap.add_argument("--dreamer_path", default=None)
    # paths
    ap.add_argument("--episodes_dir", default="episodes")
    ap.add_argument("--logdir", default="logdir")
    ap.add_argument("--policy_dir", default="policy")
    ap.add_argument("--reachable", default="reachable_set.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild reachable set")
    ap.add_argument("--fresh", action="store_true",
                    help="delete episodes/, logdir/ and policy/ first; "
                         "otherwise every run ADDS to the existing dataset "
                         "and resumes from the existing checkpoint")
    ap.add_argument("--target_hr_mae", type=float, default=None,
                    help="evaluate after each round and stop early once the "
                         "policy is below this HR MAE in bpm; leave unset to "
                         "run exactly --rounds rounds")
    ap.add_argument("--eval_cycles", type=int, default=200)
    args = ap.parse_args()

    print("=" * 66)
    print(" OFFLINE TRAINING  --  VNS control in hypertensive HFpEF")
    print(f" pipeline version {PIPELINE_VERSION}")
    print("=" * 66)

    if args.fresh:
        import shutil
        for d in (args.episodes_dir, args.logdir, args.policy_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"  removed {d}/")

    # ---- stage 0 -----------------------------------------------------
    # Built for EVERY stage that needs an environment, not just --stage all.
    # Without the table the env has no reachable references and falls back to
    # the baseline itself, so a collect run would quietly produce episodes
    # whose correct action is no stimulation.
    if os.path.exists(args.reachable) and not args.force:
        print(f"\n[0] reachable set: {args.reachable} exists, skipping")
    else:
        print(f"\n[0] building reachable set (open-loop sweep, a few minutes)")
        VE.build_reachable_set(phenotype=args.phenotype,
                               out=args.reachable, verbose=False)
    if args.stage == "reachable":
        return

    # ---- environment --------------------------------------------------
    cfg = VE.EnvConfig(phenotype=args.phenotype, max_cycles=args.max_cycles,
                       randomize=args.randomize, reachable_file=args.reachable,
                       congestion=args.congestion,
                       congestion_barrier=not args.no_barrier,
                       barrier_channel=args.barrier_channel)
    if args.lam_c is not None:
        cfg.reward.lam_c = args.lam_c
    if args.p_max is not None:
        cfg.reward.p_max = args.p_max
    print("\n[env] settling plant to steady state...")
    env = VE.VNSEnv(cfg)
    dep = "deployable" if VE.CONGESTION_DEPLOYABLE[args.congestion] \
        else "NOT chronically measurable"
    print(f"  baseline   HR {env.base['HR']:.2f} bpm   MAP {env.base['MAP']:.2f} "
          f"mmHg   LVEDP {env.base['LVEDP']:.2f} mmHg   EF {100*env.base['EF']:.1f}%")
    print(f"  obs        {VE.OBS_DIM}-D, congestion = {args.congestion} "
          f"({VE.CONGESTION[args.congestion]}, {dep})")
    print(f"  action     I in [0,{HP.I_MAX}] mA, f in [0,{HP.F_MAX}] Hz")
    if cfg.congestion_barrier:
        print(f"  reward     barrier ON, Eq.(4) on "
              f"{VE.CONGESTION[cfg.barrier_channel]} above "
              f"{cfg.reward.p_max} mmHg, lam_c={cfg.reward.lam_c} "
              f"(1 mmHg priced at {cfg.reward.exchange_rate():.0f} bpm)")
    else:
        print("  reward     barrier OFF")
    print(f"             lam_f={cfg.reward.lam_f} (0 = no frequency penalty)")
    print(f"  references {0 if env.reach is None else len(env.reach)} "
          f"reachable points inside the therapeutic window")
    if env.reach is None:
        print("  *** WARNING: no reachable set loaded. Every reference will "
              "equal the untreated")
        print("      baseline, so the correct action is no stimulation and "
              "the episodes are useless.")
        print(f"      Fix: python train_offline_vns.py --stage reachable "
              f"--phenotype {args.phenotype}")

    man = write_manifest(args, cfg, env)
    print(f"  manifest   -> {args.policy_dir}/run_manifest.json")

    # ---- rounds -------------------------------------------------------
    for rnd in range(args.rounds):
        if args.rounds > 1:
            print("\n" + "-" * 66)
            print(f" ROUND {rnd+1}/{args.rounds}")
            print("-" * 66)

        if args.stage in ("all", "collect"):
            has_ckpt = os.path.exists(
                os.path.join(args.policy_dir, "checkpoint.ckpt"))
            mode = "explore" if (rnd > 0 and has_ckpt) else "random"
            print(f"\n[1] collecting {args.episodes} episodes [{mode}]")
            stage_collect(args, env, mode, rnd)
            if args.stage == "collect":
                continue

        if args.stage in ("all", "train"):
            print("\n[2] converting episodes to replay chunks")
            stage_convert(args)
            print("\n[3] offline training")
            stage_train(args)

            if args.target_hr_mae is not None:
                mae = eval_round(args, env)
                print(f"\n[eval] HR MAE {mae:.2f} bpm "
                      f"(target {args.target_hr_mae:.2f})")
                if mae <= args.target_hr_mae:
                    print("  target met; stopping early")
                    break
                if rnd < args.rounds - 1:
                    print("  below target; next round collects guided "
                          "episodes in explore mode")

    print("\n" + "=" * 66)
    print(" Next:  python check_results.py")
    print("=" * 66)


if __name__ == "__main__":
    main()
