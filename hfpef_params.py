"""
Parameters for the 32-state pulsatile cardiovascular-baroreflex plant.

Self-contained: no imports from params_lsr_adjusted_decreasing_2.py, so this
directory can be dropped anywhere (Windows, Dropbox, HPC) and will run.

Two phenotypes:
  "healthy"  inherited resting baseline
  "hfpef"    hypertensive HFpEF, imposed in two layers per Table A1 of the
             2026 qualifying report

Table A1 factors were calibrated against the nonlinear-atrium plant of Adeodu
et al. This module applies them to the linear-atrium Park plant, so the
converged HFpEF operating point will differ somewhat from Table A3. Run
calibrate_phenotype.py to retune kE,lv and sEmax against the LVEDP target.
"""

import numpy as np

# ===================================================================
# Inherited constants (Ursino / Magosso / Park lineage, unchanged
# between phenotypes). Values follow params_lsr_adjusted_decreasing_2.py.
# ===================================================================

_INHERITED = dict(
    # systemic compliances
    Csa=0.28, Csp=2.05, Cep=1.36, Cmp=0.31, Csv=43.11, Cev=28.4,
    Cmv=6.6, Ctv=33.0,
    # pulmonary compliances
    Cpa=0.22, Cpp=1.865, Cpv=2.98,
    # unstressed volumes
    Vusa=0.0, Vusp=274.4, Vuep=274.1, Vump=62.5, Vutv=0.0,
    Vupa=0.0, Vupp=123.0, Vupv=120.0, Vura=25.0,
    # resistances
    Rsa=0.06, Rsv=0.038, Rev=0.0197, Rmv=0.0848, Rtv=0.0054,
    Rpa=0.08594273, Rpp=0.00513008, Rpv=0.00140656, Rra=0.0025,
    # inertances
    Lsa=0.00022, Lpa=0.00018,
    # right heart
    Cra=31.25, Porv=1.5, Kerv=0.011, Vurv=40.8, KRrv=0.0014,
    # misc
    P0=3.9, Tim=0.1, A=50.0, Tc=0.75,
    ksys=0.075, Tsys0=0.4, Vt=5300.0,
    # baroreceptor afferent
    fbrmin=2.52, fbrmax=47.78, ka=11.758, tauz=6.37, taup=2.067,
    # cardiopulmonary receptor
    fmaxl=20.0, Ptn=10.8, kl=11.758, taucp=2.0,
    # lung-stretch receptor
    taulung=2.0, Gal=12.0, vlung0=0.583,
    # sympathetic outflow
    fes0=16.11, fesmin=2.66, kes=0.0675,
    # left atrium (linear Ursino/Park compliance)
    Cla=19.23, Vula=25.0, Rla=0.0025,
    # left ventricle, Equation (1) of the report
    Polv=1.5, Vulv=16.77, KRlv=0.0004,
)

# effector time constants and pure delays
_TAUS = np.array([2., 2., 2., 6., 6., 6., 20., 20., 6.])
_TAUV = np.array([1.5, 2.0, 2.0])
_DS = np.array([2.0, 2.0, 2.0])
_DV = 0.2

# basal effector values theta0 =
#   [T0, Emaxlv0, Emaxrv0, Rsp0, Rep0, Rmp0, Vusv0, Vuev0, Vumv0]
_THETA0 = np.array([0.58, 1.283, 0.757, 2.49, 0.96, 4.13, 1435.4, 1247.0, 290.0])

# sympathetic effector gains
_GS = np.array([-0.13, -0.13, -0.22, 0.695, 0.653, 2.81, -265.4, -107.5, -25.0])

# vagal effector gains [T, Emax_lv, Emax_rv]
_GV = np.array([0.10353516, 0.23823242, 0.3666543])

# NTS afferent sigmoids, order (BR, CPR, LSR)
_FMIN = np.array([0.3, 0.451, 2.75])
_FMAX = np.array([21.5, 28.357, 31.57])
_MIDPT = np.array([44.3, 10.2, 10.0])
_KSIG = np.array([2.14, 1.636, 7.516])

# efferent sigmoids, order (NA, DMV, NActr)
_FOUTMIN = np.array([4.88, 2.59, 0.61])
_FOUTMAX = np.array([15.78, 6.66, 11.0])
_FOUTMIDPT = np.array([60.0, 43.1, 9.8])
_FOUTK = np.array([2.55, 1.24, 1.2])

# NTS -> (NA, DMV) weighting
_KRECEPTOR = np.array([[1., 1., -1.],
                       [0., 1., 1.]])

# afferent -> sympathetic weighting; entry [0,2] is -Galh
_GMAT = np.array([[1.0, 2.0, -1.541],
                  [1.0, 2.5, 0.33],
                  [1.0, 0.0, 0.0]])

# baroreceptor operating pressure
_PN = 92.0

# ===================================================================
# VNS fiber recruitment, Equation (2) and Table A2
# Index 0 = Adelta/B  (baroreceptor afferents + vagal efferent output)
# Index 1 = Abeta     (lung-stretch afferent)
# Getting this index backwards misplaces the neural fulcrum.
# ===================================================================

IMID = np.array([1.912, 0.637])      # mA
KREC = np.array([0.318, 0.106])      # mA
I_MAX = 3.0                          # mA, Adelta/B saturation

# Frequency bound. Table A2 of the report gives [0, 10] Hz, justified by
# Sadashivaiah et al. (2018): the additive form of Equation (2),
# f_phys + R*f_stim, holds only while evoked and physiological spikes rarely
# collide, which that study bounds at roughly 10 Hz for these fiber
# diameters. Above it the refractory period starts dropping spikes and the
# true resultant rate saturates below the additive prediction, so the plant
# OVERSTATES what the actuator can deliver in 10-15 Hz.
#
# F_MAX = 15 therefore extends the action set past the cited validity range.
# Two ways to make it defensible, neither implemented here:
#   1. cite a source that carries additivity to 15 Hz at this pulse width, or
#   2. replace Equation (2) with a saturating collision model, e.g.
#      f_r = f_phys + R*f_stim / (1 + f_stim/f_sat), and fit f_sat.
# Until one of those is done, treat any result that puts the policy above
# 10 Hz as optimistic about actuator authority.
F_MAX = 15.0                         # Hz
F_ADDITIVITY_LIMIT = 10.0            # Hz, the sourced bound

# ===================================================================
# Table A1: the hypertensive HFpEF phenotype
# ===================================================================

HFPEF_LAYER1 = dict(
    R_scale=1.45,      # Rsp0, Rep0, Rmp0
    Pn_shift=28.0,     # baroreceptor resetting, mmHg
    gv_scale=0.65,     # all three vagal effector gains
)

HFPEF_LAYER2 = dict(
    # Table A1 gives 1.61 (kE,lv 0.0140 -> 0.0225), calibrated against the
    # nonlinear-atrium plant. On this linear-atrium plant that lands LVEDP at
    # 15.6 mmHg, just short of the 16 mmHg HFA-PEFF confirmation threshold.
    # Rescaled to 1.66 (kE,lv = 0.02324) to clear it: LVEDP 16.5, EF 57.6 %,
    # Pla 16.9, everything else within ~1 % of Table A3. Set to 1.61 to
    # recover the report value exactly.
    kElv_scale=1.66,
    sEmax=1.95,        # multiplicative scale on the whole Emax,lv expression
)

# resting activity vector, no exercise, no muscle pump
#   (Tresp, Tinsp, Texp, fes_cc, fev_cc, Rd, Pthor_max, Pthor_min, pump)
REST = (4.0, 1.6, 1.4, 0.0, 0.0, 10000.0, -4.0, -9.0, 0.0)


def build(phenotype="healthy", overrides=None):
    """
    Assemble the parameter dict for one phenotype.

    overrides : optional dict applied last, for domain randomization or
                recalibration (e.g. {"Kelv": 0.0231, "sEmax": 1.88}).
    """
    P = dict(_INHERITED)
    P["taus"] = _TAUS.copy()
    P["tauv"] = _TAUV.copy()
    P["ds"] = _DS.copy()
    P["dv"] = _DV
    P["theta0"] = _THETA0.copy()
    P["Gs"] = _GS.copy()
    P["Gv"] = _GV.copy()
    P["fmin"] = _FMIN.copy()
    P["fmax"] = _FMAX.copy()
    P["midpt"] = _MIDPT.copy()
    P["k"] = _KSIG.copy()
    P["foutmin"] = _FOUTMIN.copy()
    P["foutmax"] = _FOUTMAX.copy()
    P["foutmidpt"] = _FOUTMIDPT.copy()
    P["foutk"] = _FOUTK.copy()
    P["kreceptor"] = _KRECEPTOR.copy()
    P["G"] = _GMAT.copy()
    P["Pn"] = _PN
    P["fesinf"] = 2.1
    P["Kelv"] = 0.014
    P["sEmax"] = 1.0
    P["Imid"] = IMID.copy()
    P["krec"] = KREC.copy()

    if phenotype == "hfpef":
        L1, L2 = HFPEF_LAYER1, HFPEF_LAYER2
        # Layer 1: hypertensive substrate
        P["theta0"][3] *= L1["R_scale"]      # Rsp,0  2.49 -> 3.61
        P["theta0"][4] *= L1["R_scale"]      # Rep,0  0.96 -> 1.39
        P["theta0"][5] *= L1["R_scale"]      # Rmp,0  4.13 -> 5.99
        P["Pn"] += L1["Pn_shift"]            # 92 -> 120
        P["Gv"] *= L1["gv_scale"]            # vagal withdrawal
        # Layer 2: ventricular remodeling
        P["Kelv"] *= L2["kElv_scale"]        # 0.0140 -> 0.0225
        P["sEmax"] = L2["sEmax"]
    elif phenotype != "healthy":
        raise ValueError(f"unknown phenotype {phenotype!r}")

    P["phenotype"] = phenotype
    if overrides:
        P.update(overrides)
    return P


# ===================================================================
# Domain randomization (Section 3.3.2 of the report: randomize disease
# parameters, baroreflex gains and electrode-nerve coupling). Not used
# by the default pipeline; wire it into data_collector with --randomize.
# ===================================================================

RANDOMIZATION = dict(
    Kelv=(0.90, 1.10),      # diastolic stiffness
    sEmax=(0.90, 1.10),     # systolic stiffness
    Pn=(0.97, 1.03),        # baroreflex operating point
    Imid0=(0.85, 1.15),     # Adelta/B half-recruitment current: interface drift
    krec0=(0.85, 1.15),     # Adelta/B recruitment sigmoid width
)


def randomize(P, rng):
    """Multiplicative jitter on the parameters most likely to drift."""
    P = dict(P)
    P["theta0"] = P["theta0"].copy()
    P["Imid"] = P["Imid"].copy()
    P["krec"] = P["krec"].copy()
    P["Kelv"] *= rng.uniform(*RANDOMIZATION["Kelv"])
    P["sEmax"] *= rng.uniform(*RANDOMIZATION["sEmax"])
    P["Pn"] *= rng.uniform(*RANDOMIZATION["Pn"])
    P["Imid"][0] *= rng.uniform(*RANDOMIZATION["Imid0"])
    P["krec"][0] *= rng.uniform(*RANDOMIZATION["krec0"])
    return P
