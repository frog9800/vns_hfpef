"""
Per-cardiac-cycle VNS control environment for the hypertensive HFpEF plant.

Control is posed per beat: the decision interval is the cardiac period, every
action is held for exactly one cycle, and the reward quantities are defined
once per cycle. This is the semi-Markov structure of Section 3.3 of the
report, not a fixed-dt MDP.

WHAT THE POLICY SEES (8-D, encoder input)
    [HR, MAP, CONG, HR_ref, MAP_ref, I_prev, f_prev, T]
  HR and MAP are read by any rate-sensing implant. CONG is the congestion
  channel, selected by EnvConfig.congestion:

    "papd"   pulmonary artery diastolic pressure.  DEPLOYABLE.
             This is what a CardioMEMS-class implantable PA pressure sensor
             reports, and such devices are already used for ambulatory heart
             failure management. In this plant PAPd tracks LVEDP to within
             0.02 mmHg in both phenotypes (9.93 vs 9.95 healthy, 16.46 vs
             16.48 HFpEF), because with the mitral valve open in diastole the
             pulmonary bed and the left heart equilibrate. So congestion can
             be a genuine observation rather than a modeling convenience.

    "pla"    mean left atrial pressure, the PCWP analogue. Measurable by
             right-heart catheterization, not chronically.

    "lvedp"  the filling pressure of Equation (1) directly. NOT measurable by
             any chronic implant. Using it makes the trained policy a
             simulation result, not a device design.

  The references and the previous action are known by construction, and T is
  the decision interval itself, which a semi-Markov policy needs to interpret
  its own step. There is no congestion reference: the barrier is one-sided,
  so a fixed ceiling P_max replaces a tracked setpoint.

WHAT THE POLICY NEVER SEES (aux, logged for evaluation only)
    [LVEDP, Pla_mean, PAPd, PAPm, EDV, ESV, SV, EF, CO, Emaxlv, SBP, DBP]
  The full hidden set is kept regardless of which congestion channel is
  observed, so a run can always be audited against the true filling pressure
  even when the policy was reading a surrogate.

ACTION
    a in [-1,1]^2  ->  I_stim = 1.5*(a0+1) in [0,3] mA
                       f_stim = 7.5*(a1+1) in [0,15] Hz
  DreamerV3 actions are symmetric; the physical action set is one-sided, so
  the map is affine rather than the plain scaling DropletRunner uses.

REWARD, Equation (3)
    r = -( e^T We(e) e  +  lam_u a_hat^T Wu a_hat  +  lam_d ||delta a||^2 )
    e = y_ref - y
  e_i > 0 means the output was driven BELOW reference, i.e. overshoot toward
  bradycardia and hypoperfusion; that branch takes the larger weight w+.
  Residual error above reference is untreated disease and takes w-.

  With congestion_barrier=True the one-sided barrier of Equation (4) is
  added on the OBSERVED congestion channel:

      r  <-  r - lam_c * [max(0, CONG - P_max)]^2

  Observing congestion without this term changes what the agent KNOWS but
  not what it WANTS: the extra channel sharpens the belief state and can
  only help the world model, but with no reward on it the policy has no
  reason to act on congestion at all. The two switches are independent on
  purpose so the effect of each can be separated.
"""

import numpy as np
from dataclasses import dataclass, field

import hfpef_params as HP
import hfpef_plant as plant

OBS_DIM = 8
ACT_DIM = 2
AUX_KEYS = ["LVEDP", "Pla_mean", "PAPd", "PAPm", "EDV", "ESV", "SV", "EF",
            "CO", "Emaxlv", "SBP", "DBP"]
AUX_DIM = len(AUX_KEYS)

# congestion channel -> plant measurement key
REWARD_TERMS = ["track_hr", "track_map", "energy", "chatter", "barrier", "freq"]
CONGESTION = {"papd": "PAPd", "pla": "Pla_mean", "lvedp": "LVEDP"}
CONGESTION_DEPLOYABLE = {"papd": True, "pla": False, "lvedp": False}

# Observation normalization. These constants are part of the trained model:
# collection, training and deployment must all use the same ones. Index 2 is
# the congestion channel; all three options share a scale because they sit in
# the same 8-20 mmHg band in this plant.
OBS_LOC = np.array([70.0, 100.0, 16.0, 70.0, 100.0, 1.5, 7.5, 0.85],
                   dtype=np.float32)
OBS_SCALE = np.array([20.0, 20.0, 3.0, 20.0, 20.0, 1.5, 7.5, 0.20],
                     dtype=np.float32)


def to_physical(a):
    """[-1,1]^2 -> (I_stim mA, f_stim Hz)."""
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    return 0.5 * HP.I_MAX * (a[0] + 1.0), 0.5 * HP.F_MAX * (a[1] + 1.0)


def to_normalized(I, f):
    """(I_stim mA, f_stim Hz) -> [-1,1]^2."""
    return np.array([2.0 * I / HP.I_MAX - 1.0, 2.0 * f / HP.F_MAX - 1.0],
                    dtype=np.float32)


# --------------------------------------------------------------- config

@dataclass
class RewardConfig:
    """
    All six weights are UNSET in the report and tunable here. The ratio
    w_plus/w_minus is the one that carries meaning: it prices overshoot
    against untreated disease.
    """
    hr_tol: float = 5.0        # bpm, error scale
    map_tol: float = 5.0       # mmHg, error scale
    overshoot_ratio: float = 4.0   # w_plus / w_minus, both channels
    lam_u: float = 0.01        # stimulation energy
    lam_d: float = 0.05        # action chatter
    # Congestion barrier, Equation (4):  lam_c * [max(0, LVEDP - p_max)]^2
    #
    # p_max = 16 mmHg is the HFA-PEFF invasive confirmation threshold, so the
    # barrier reads "keep the patient out of the diagnostic range".
    #
    # lam_c is best set by the EXCHANGE RATE it implies, not chosen directly.
    # Matching marginal costs at the baseline operating point,
    #     d(barrier)/d(LVEDP) = 2*lam_c*over,   d(track_hr)/d(e) = 2*e/hr_tol^2
    # so one mmHg of filling pressure is priced at exchange_rate() bpm of
    # heart-rate error. Calibrated against the admissible action set
    # (MAP >= 90, HR >= 55), over which LVEDP spans only 15.69-16.85 mmHg:
    #
    #   lam_c   1 mmHg LVEDP costs   chosen action changes?
    #     0.5          6 bpm         no
    #     1.0         12 bpm         no
    #     2.0         23 bpm         no          <- default
    #     5.0         59 bpm         yes: +6.1 bpm HR error for 0.20 mmHg
    #    10.0        117 bpm         yes: +3.8 bpm HR error for 0.36 mmHg
    #    50-100        -             flips SUB-FULCRUM, see below
    #
    # Read that table before reporting anything. The barrier only redirects
    # the policy at exchange rates around 30+ bpm per mmHg, which is not a
    # trade a clinician would take. At any defensible rate it is a near
    # constant tax that does not change the chosen action. That is not a
    # tuning failure, it is Section 5 of the report measured: VNS moves
    # filling pressure only by unloading the whole circulation, so the
    # barrier cannot be earned at a price worth paying. A constant offset
    # does not change the optimal policy, only the value scale, so the
    # default is safe to leave on.
    #
    # The failure mode above lam_c ~50: sub-fulcrum stimulation drives
    # tachycardia, tachycardia shortens diastole, shorter diastole lowers
    # filling pressure. It earns the barrier without treating anything. The
    # HR tracking term is what holds that line, so re-run the sweep in
    # calibrate_barrier() if lam_c is raised.
    #
    # CORRECTION to an earlier note here: the barrier CAN reach zero. Two
    # routes exist, and both are non-therapeutic, which is why the rest of
    # the reward and the shield have to block them:
    #
    #   overdose        3.0 mA / 15 Hz  ->  LVEDP 14.39 (-2.08 mmHg), but
    #                   MAP 47.9 and HR 33.7. Blocked by the MAP tracking
    #                   term, the shield amplitude rolloff, and episode
    #                   termination at MAP < 60. Commanding it closed-loop
    #                   with the shield on lands at 1.82 mA applied, MAP
    #                   80.6, LVEDP -0.33; with the shield off the episode
    #                   terminates at cycle 2.
    #
    #   tachycardia     0.75 mA / 15 Hz ->  LVEDP 15.69 (-0.79 mmHg) with
    #                   MAP 117 and HR 92.9, so it passes the MAP and HR
    #                   FLOORS. Blocked only by the HR tracking term, and
    #                   formerly not blocked at all in the reference set,
    #                   which is what the therapeutic ceiling in
    #                   build_reachable_set now fixes.
    #
    # So the honest statement is not that decongestion is unreachable, it is
    # that every route to it costs more than it returns at any defensible
    # exchange rate.
    lam_c: float = 2.0
    p_max: float = 16.0        # mmHg

    # Optional one-sided penalty on frequency above f_soft. Off by default
    # (lam_f = 0). The 10 Hz figure is the additivity limit of Equation (2),
    # a bound on what the equation can model rather than a therapeutic or
    # device limit, so it is not charged to the agent unless asked for.
    lam_f: float = 0.0         # per Hz^2 above f_soft; 0 = no penalty
    f_soft: float = HP.F_ADDITIVITY_LIMIT    # 10 Hz

    def freq_penalty(self, f):
        if self.lam_f == 0.0:
            return 0.0
        over = max(0.0, float(f) - self.f_soft)
        return self.lam_f * over * over

    def exchange_rate(self, lvedp=16.47):
        """bpm of heart-rate error that one mmHg of LVEDP is priced at."""
        over = max(0.0, lvedp - self.p_max)
        return 2.0 * self.lam_c * over / (2.0 / self.hr_tol ** 2)

    def barrier(self, p):
        """One-sided penalty, Equation (4). Zero at or below the ceiling."""
        over = max(0.0, float(p) - self.p_max)
        return self.lam_c * over * over

    def weights(self, e):
        """We(e) = diag(w1, w2), branch on the sign of e = ref - y."""
        w = np.empty(2)
        for i, tol in enumerate((self.hr_tol, self.map_tol)):
            base = 1.0 / tol ** 2
            w[i] = base * (self.overshoot_ratio if e[i] > 0 else 1.0)
        return w


@dataclass
class SafetyShield:
    """
    Reflex shield. VNS past the fulcrum is unidirectional and can only pull
    HR and MAP down, so the failure mode is monotone: too much amplitude for
    too long. The shield scales the commanded amplitude back when either
    output falls through its floor. It acts on the plant, and the APPLIED
    action is what gets stored in the dataset, so the world model is fit to
    the shielded system that will actually be deployed.
    """
    enabled: bool = True
    hr_floor: float = 50.0     # bpm
    map_floor: float = 85.0    # mmHg
    hr_hard: float = 42.0
    map_hard: float = 70.0
    # Hard frequency cap, off by default. Set to 10.0 to keep the actuator
    # strictly inside the sourced additivity range, which is the alternative
    # to pricing the excursion in the reward: a cap removes the region, a
    # penalty lets the agent buy into it. Capping here rather than lowering
    # F_MAX keeps the action normalization unchanged, so a checkpoint trained
    # with one setting still loads under the other.
    f_cap: float = None

    def __call__(self, a, HR, MAP):
        if not self.enabled:
            return np.asarray(a, dtype=np.float32), False
        a = np.array(a, dtype=np.float32)
        capped = False
        if self.f_cap is not None:
            I0, f0 = to_physical(a)
            if f0 > self.f_cap:
                a = to_normalized(I0, self.f_cap)
                capped = True
        # linear rolloff between floor and hard limit
        g_hr = np.clip((HR - self.hr_hard) / (self.hr_floor - self.hr_hard), 0.0, 1.0)
        g_map = np.clip((MAP - self.map_hard) / (self.map_floor - self.map_hard), 0.0, 1.0)
        g = min(g_hr, g_map)
        if g >= 1.0:
            return a, capped
        I, f = to_physical(a)
        return to_normalized(I * g, f), True


@dataclass
class EnvConfig:
    phenotype: str = "hfpef"
    congestion: str = "papd"           # OBSERVED channel, see module docstring
    congestion_barrier: bool = True    # Equation (4) in the reward
    barrier_channel: str = "lvedp"     # REWARDED channel, independent of the above
    max_cycles: int = 400              # ~ 300 s at 80 bpm
    settle_seconds: float = 200.0      # run to steady state before episode 0
    dt: float = plant.DT
    randomize: bool = False            # domain randomization per episode
    reward: RewardConfig = field(default_factory=RewardConfig)
    shield: SafetyShield = field(default_factory=SafetyShield)
    # reachable-setpoint schedule
    n_segments: int = 4                # reference changes per episode
    reachable_file: str = "reachable_set.npz"
    map_ref_floor: float = 90.0        # therapeutic window, mmHg
    hr_ref_floor: float = 55.0         # bpm
    ref_jitter: float = 0.02           # fractional jitter on a drawn point
    terminate_hr: float = 40.0         # episode-ending safety violation
    terminate_map: float = 60.0


# --------------------------------------------------------------- env

class VNSEnv:
    """
    Minimal env, no gym dependency. Interface mirrors what data_collector and
    policy_runner need:

        obs, info = env.reset(seed=...)
        obs, reward, terminal, info = env.step(action)

    obs is the NORMALIZED 7-vector. info carries the raw measurements, the
    aux vector, the applied action and the current reference.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(0)
        for k in ("congestion", "barrier_channel"):
            if getattr(self.cfg, k) not in CONGESTION:
                raise ValueError(f"{k} must be one of {list(CONGESTION)}")
        self._P0 = HP.build(self.cfg.phenotype)
        self._cache = {}          # settled state per parameter signature
        self.reach = load_reachable(self.cfg)
        self.reset()

    # -------------------------------------------------- setpoints

    def _sample_schedule(self, base_hr, base_map):
        """
        Draw references from the reachable set intersected with the
        therapeutic window.

        Sampling HR and MAP independently from a box is wrong here: VNS pulls
        both down together, so most of the box is unreachable by any action
        and the agent would spend its training on impossible targets. The
        reachable set is the image of the action set under the plant's
        steady-state map, so every point in the table below is achievable by
        SOME constant (I, f), which makes it achievable by a policy.
        """
        c = self.cfg
        if self.reach is None:
            # no table available: fall back to the baseline itself, which is
            # always reachable (zero stimulation).
            return [(base_hr, base_map)] * c.n_segments
        idx = self.rng.integers(0, len(self.reach), size=c.n_segments)
        segs = []
        for i in idx:
            hr, mp = self.reach[i]
            j = 1.0 + c.ref_jitter * self.rng.standard_normal(2)
            segs.append((float(np.clip(hr * j[0], c.hr_ref_floor, base_hr)),
                         float(np.clip(mp * j[1], c.map_ref_floor, base_map))))
        return segs

    def _reference(self, k):
        seg = min(int(k / self.cfg.max_cycles * self.cfg.n_segments),
                  self.cfg.n_segments - 1)
        return self.schedule[seg]

    # -------------------------------------------------- lifecycle

    def reset(self, seed=None, schedule=None):
        c = self.cfg
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.P = HP.randomize(self._P0, self.rng) if c.randomize else dict(self._P0)

        sig = (round(self.P["Kelv"], 8), round(self.P["sEmax"], 6),
               round(self.P["Pn"], 6))
        if sig not in self._cache:
            st, base = plant.baseline(self.P, HP.REST, seconds=c.settle_seconds,
                                      dt=c.dt)
            self._cache[sig] = (st, base)
        st, base = self._cache[sig]

        self.state = st.copy()
        self.base = base
        self.schedule = schedule or self._sample_schedule(base["HR"], base["MAP"])

        self.k = 0
        self.terms = {k: 0.0 for k in ("track_hr", "track_map", "energy",
                                       "chatter", "barrier", "freq")}
        self.prev_a = np.zeros(2, dtype=np.float32)
        self.last = dict(base)
        self.t = 0.0
        return self._obs(), self._info(0.0, np.zeros(2, dtype=np.float32), False)

    def _cong(self):
        """Congestion the POLICY observes. Must be chronically measurable."""
        return float(self.last[CONGESTION[self.cfg.congestion]])

    def _barrier_val(self):
        """
        Congestion the REWARD charges. Defaults to true LVEDP.

        This may be a quantity no implant can measure, and that is not a
        deployment problem: the reward exists only while collecting data and
        fitting the world model. Nothing evaluates it at run time. So the
        barrier is charged on the clinical quantity, LVEDP, while the policy
        reads the deployable surrogate, PAPd. Same asymmetry as the
        decoder-only aux targets.
        """
        return float(self.last[CONGESTION[self.cfg.barrier_channel]])

    def _obs(self):
        ref = self._reference(self.k)
        v = np.array([self.last["HR"], self.last["MAP"], self._cong(),
                      ref[0], ref[1],
                      *to_physical(self.prev_a), self.last["T"]],
                     dtype=np.float32)
        return ((v - OBS_LOC) / OBS_SCALE).astype(np.float32)

    def _aux(self):
        return np.array([self.last[k] for k in AUX_KEYS], dtype=np.float32)

    def _info(self, reward, applied, shielded):
        ref = self._reference(self.k)
        return dict(meas=dict(self.last), aux=self._aux(),
                    ref=np.array(ref, dtype=np.float32),
                    applied=applied.astype(np.float32),
                    shielded=shielded, reward=float(reward),
                    terms=dict(self.terms), t=self.t, cycle=self.k)

    # -------------------------------------------------- step

    def step(self, action):
        c = self.cfg
        a_req = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        a, shielded = c.shield(a_req, self.last["HR"], self.last["MAP"])

        I, f = to_physical(a)
        self.state, meas = plant.step_cycle(self.state, (I, f), self.P,
                                            HP.REST, c.dt)
        self.last = meas
        self.t += meas["T"]
        self.k += 1

        ref = self._reference(self.k)
        e = np.array([ref[0] - meas["HR"], ref[1] - meas["MAP"]])
        w = c.reward.weights(e)
        a_hat = np.array([I / HP.I_MAX, f / HP.F_MAX])
        da = a - self.prev_a
        # Reward decomposed into named costs so the evaluation script can
        # report where the return is actually going. reward = -sum(terms).
        terms = {
            "track_hr": float(w[0] * e[0] ** 2),
            "track_map": float(w[1] * e[1] ** 2),
            "energy": float(c.reward.lam_u * (a_hat @ a_hat)),
            "chatter": float(c.reward.lam_d * (da @ da)),
            "barrier": (c.reward.barrier(self._barrier_val())
                        if c.congestion_barrier else 0.0),
            "freq": (c.reward.freq_penalty(f) if c.reward.lam_f else 0.0),
        }
        reward = -sum(terms.values())
        self.terms = terms

        self.prev_a = a
        terminal = bool(meas["HR"] < c.terminate_hr or
                        meas["MAP"] < c.terminate_map or
                        not np.isfinite(meas["HR"]))
        return self._obs(), reward, terminal, self._info(reward, a, shielded)

    @property
    def truncated(self):
        return self.k >= self.cfg.max_cycles


# --------------------------------------------------------------- reachable set

def build_reachable_set(phenotype="hfpef", n_amp=13, n_freq=4, seconds=90.0,
                        average_cycles=8, map_floor=90.0, hr_floor=55.0,
                        therapeutic_only=True,
                        out="reachable_set.npz", verbose=True):
    """
    Image of the action set under the plant's steady-state map, filtered to
    the therapeutic window. Run once per phenotype and cache.

    Returns (points, grid) where points is the (N,2) array of admissible
    (HR, MAP) and grid is the full (M,4) table [I, f, HR, MAP] including the
    rejected points, which is also the open-loop dose-response surface.
    """
    P = HP.build(phenotype)
    st0, base = plant.baseline(P, HP.REST, seconds=200.0)
    amps = np.linspace(0.0, HP.I_MAX, n_amp)
    freqs = np.linspace(HP.F_MAX / n_freq, HP.F_MAX, n_freq)
    grid = []
    for I in amps:
        for f in freqs:
            st = st0.copy()
            t, hist = 0.0, []
            while t < seconds:
                st, m = plant.step_cycle(st, (float(I), float(f)), P, HP.REST)
                t += m["T"]
                hist.append(m)
            tail = hist[-average_cycles:]
            hr = float(np.mean([h["HR"] for h in tail]))
            mp = float(np.mean([h["MAP"] for h in tail]))
            lv = float(np.mean([h["LVEDP"] for h in tail]))
            grid.append([I, f, hr, mp, lv])
            if verbose:
                keep = "keep" if (mp >= map_floor and hr >= hr_floor) else "  --"
                print(f"  I={I:4.2f} mA  f={f:5.2f} Hz  ->  HR={hr:6.2f}  "
                      f"MAP={mp:6.2f}  LVEDP={lv:5.2f}   [{keep}]")
    grid = np.asarray(grid, dtype=np.float32)
    mask = (grid[:, 3] >= map_floor) & (grid[:, 2] >= hr_floor)
    if therapeutic_only:
        # Floors alone are not enough. Sub-fulcrum stimulation RAISES heart
        # rate, and those points clear MAP >= 90 and HR >= 55 comfortably, so
        # without a ceiling they enter the reference set as legitimate
        # targets. On the 9x3 grid that was 9 of 16 references asking VNS to
        # make an HFpEF patient faster than untreated. VNS therapy is rate
        # and pressure REDUCTION, so a reference above baseline is not a
        # therapeutic target whatever the floors say.
        mask &= (grid[:, 2] <= base["HR"]) & (grid[:, 3] <= base["MAP"])
    points = grid[mask][:, 2:4]
    if out:
        np.savez(out, points=points, grid=grid,
                 base=np.array([base["HR"], base["MAP"], base["LVEDP"]],
                               dtype=np.float32),
                 phenotype=phenotype)
        if verbose:
            print(f"\n{len(points)}/{len(grid)} grid points inside the "
                  f"therapeutic window -> {out}")
    return points, grid


def load_reachable(cfg):
    import os
    if not cfg.reachable_file or not os.path.exists(cfg.reachable_file):
        return None
    d = np.load(cfg.reachable_file)
    if str(d["phenotype"]) != cfg.phenotype:
        return None
    pts = d["points"]
    return pts if len(pts) else None


# --------------------------------------------------------------- sweep

def fulcrum_sweep(phenotype="hfpef", fstim=10.0, amps=None, seconds=100.0,
                  average_cycles=8):
    """
    Open-loop amplitude sweep, used to locate the neural fulcrum and to bound
    the reachable set that _sample_schedule draws from. Mirrors Appendix B of
    the report: each point run from the converged baseline, averaged over the
    last cycles.
    """
    amps = np.arange(0.0, HP.I_MAX + 1e-9, 0.25) if amps is None else np.asarray(amps)
    P = HP.build(phenotype)
    st0, base = plant.baseline(P, HP.REST, seconds=200.0)
    rows = []
    for I in amps:
        st = st0.copy()
        t, hist = 0.0, []
        while t < seconds:
            st, m = plant.step_cycle(st, (float(I), float(fstim)), P, HP.REST)
            t += m["T"]
            hist.append(m)
        tail = hist[-average_cycles:]
        rows.append(dict(
            I=float(I), f=float(fstim),
            dHR=float(np.mean([h["HR"] for h in tail]) - base["HR"]),
            dMAP=float(np.mean([h["MAP"] for h in tail]) - base["MAP"]),
            dLVEDP=float(np.mean([h["LVEDP"] for h in tail]) - base["LVEDP"]),
            HR=float(np.mean([h["HR"] for h in tail])),
            MAP=float(np.mean([h["MAP"] for h in tail])),
            LVEDP=float(np.mean([h["LVEDP"] for h in tail])),
        ))
    return base, rows
