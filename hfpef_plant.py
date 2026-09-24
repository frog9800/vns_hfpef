"""
32-state pulsatile cardiovascular-baroreflex plant with VNS coupling.

AF-free: the stochastic Zeng-Glass AV-node filter is removed and the heart
period is a model output, T = T0 + dTs + dTv, as in Ursino/Park. HFpEF is
modeled in sinus rhythm.

Left atrium is the linear Ursino/Park compliance Pla = (Vla - Vula)/Cla.

Left ventricle, Equation (1) of the report:
    Plv = phi*Emax_lv*(Vlv - Vu_lv) + (1-phi)*P0_lv*(exp(kE_lv*Vlv) - 1)
with
    Emax_lv = sEmax * ( theta0_2 + 1/(dEls + dElv) )
The scale sEmax multiplies the WHOLE reciprocal expression so it sits outside
the reflex loop and cannot be compensated away.

VNS, Equation (2):
    R_j(I) = sigmoid((I - Imid_j)/k_j),   f_phys_j <- f_phys_j + R_j * f_stim
  j=0  Adelta/B  -> baroreceptor afferents + vagal efferent (NA) output
  j=1  Abeta     -> lung-stretch afferent
The opposition between these two curves is what produces the neural fulcrum.

State vector (32), ordering inherited from AF_5.py:
  0  u        cardiac phase [0,1]      1  zeta   respiratory phase [0,1]
  2  dTs      3  dTv
  4  dEls     5  dElv     6  dErs      7  dErv
  8  dRsp     9  dRep    10  dRmp
 11  dVusv   12  dVuev   13  dVumv
 14  Ppa     15  Fpa     16  Ppp      17  Ppv
 18  Psa     19  Fsa     20  Psp      21  Pev    22  Ptv   23  Pra
 24  Ptilde  25  Pl      26  flr
 27  Vmv     28  Vlv     29  Vrv      30  Vla    31  psi
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field

DT = 0.001
T_MIN, T_MAX = 0.30, 2.00
CYCLE_TIMEOUT = 3.0          # s; a cycle should never take this long


# ----------------------------------------------------------------- helpers

def _sigmoid(x, xmid, ymin, ymax, k):
    e = np.exp((x - xmid) / k)
    return (ymin + ymax * e) / (1.0 + e)


def _flow(pin, pout, R):
    return (pin - pout) / R if pin > pout else 0.0


def _phi(u, T, P):
    Tsys = P["Tsys0"] - P["ksys"] / T
    if u <= Tsys / T:
        return np.sin(np.pi * u * T / Tsys) ** 2
    return 0.0


def _pthor(z, act):
    Tresp, Tinsp, Texp, pmax, pmin = act[0], act[1], act[2], act[6], act[7]
    if z < Tinsp / Tresp:
        return pmax - (pmax - pmin) * (Tresp / Tinsp) * z
    if z < (Tinsp + Texp) / Tresp:
        return pmax - (pmax - pmin) / Texp * (Tinsp + Texp - z * Tresp)
    return pmax


def _pabd(z, act):
    Tresp, Tinsp, Texp = act[0], act[1], act[2]
    if z < (Tinsp / 2) / Tresp:
        return -2.5 * z * Tresp / (Tinsp / 2)
    if z < Tinsp / Tresp:
        return -2.5
    if z < (Tinsp + Texp) / Tresp:
        return -2.5 * (Tinsp + Texp - z * Tresp) / Texp
    return 0.0


def _sigma(Gs, fhist, fesmin):
    return Gs * np.log(fhist - fesmin + 1.0) if fhist > fesmin else 0.0


def recruit(I, P):
    """Fractional fiber recruitment, Equation (2). Returns [R_AdeltaB, R_Abeta]."""
    return np.array([
        _sigmoid(I, P["Imid"][0], 0.0, 1.0, P["krec"][0]),
        _sigmoid(I, P["Imid"][1], 0.0, 1.0, P["krec"][1]),
    ])


# ----------------------------------------------------------------- dynamics

def derivatives(x, ustar, act, hs1, hs2, hs3, hp1, hp2, P):
    (u, zeta, dTs, dTv, dEls, dElv, dErs, dErv, dRsp, dRep, dRmp,
     dVusv, dVuev, dVumv, Ppa, Fpa, Ppp, Ppv, Psa, Fsa, Psp, Pev,
     Ptv, Pra, Ptilde, Pl, flr, Vmv, Vlv, Vrv, Vla, psi) = x

    T = P["theta0"][0] + dTs + dTv
    T = min(max(T, T_MIN), T_MAX)

    Emaxlv = P["sEmax"] * (P["theta0"][1] + 1.0 / (dEls + dElv))
    Emaxrv = P["theta0"][2] + 1.0 / (dErs + dErv)

    Rsp = dRsp + P["theta0"][3]
    Rep = dRep + P["theta0"][4]
    Rmp = dRmp + P["theta0"][5]
    Vusv = dVusv + P["theta0"][6]
    Vuev = dVuev + P["theta0"][7]
    Vumv = dVumv + P["theta0"][8]

    Rlb = 1.0 / ((1.0 / Rmp) + (1.0 / act[5]))
    Pthor = _pthor(zeta, act)
    vlung = P["vlung0"] - 0.1 * Pthor
    Pabd = _pabd(zeta, act)
    Psp_trans = Psp - Pabd

    phi = _phi(u, T, P)
    Pim = P["A"] * (np.sin(np.pi * (P["Tim"] / P["Tc"]) * psi)
                    if psi <= P["Tc"] / P["Tim"] else 0.0)

    Pla = (Vla - P["Vula"]) / P["Cla"]

    Pmaxlv = phi * Emaxlv * (Vlv - P["Vulv"]) + \
        (1.0 - phi) * P["Polv"] * (np.exp(P["Kelv"] * Vlv) - 1.0)
    Rlv = P["KRlv"] * Pmaxlv
    Qol = _flow(Pmaxlv, Psa, Rlv)
    Plv = Pmaxlv - Rlv * Qol
    Qil = _flow(Pla, Plv, P["Rla"])

    Pmaxrv = phi * Emaxrv * (Vrv - P["Vurv"]) + \
        (1.0 - phi) * P["Porv"] * (np.exp(P["Kerv"] * Vrv) - 1.0)
    Rrv = P["KRrv"] * Pmaxrv
    Qor = _flow(Pmaxrv, Ppa, Rrv)
    Prv = Pmaxrv - Rrv * Qor
    Qir = _flow(Pra, Prv, P["Rra"])

    Vu = (P["Vusa"] + P["Vusp"] + P["Vump"] + P["Vuep"] + Vusv + Vuev +
          Vumv + P["Vutv"] + P["Vura"] + P["Vupa"] + P["Vupp"] + P["Vupv"])

    if Vmv >= Vumv:
        Pmv = (1.0 / P["Cmv"]) * (Vmv - Vumv) + Pim * act[8]
    else:
        Pmv = P["P0"] * (1.0 - (Vmv / Vumv) ** (-1.5)) + Pim * act[8]

    Psv = (1.0 / P["Csv"]) * (
        P["Vt"] - P["Csa"] * Psa - (P["Csp"] + P["Cep"] + P["Cmp"]) * Psp
        - P["Cev"] * Pev - P["Cmv"] * Pmv - P["Ctv"] * Ptv - P["Cra"] * Pra
        - P["Cpa"] * Ppa - P["Cpp"] * Ppp - P["Cpv"] * Ppv
        - Vla - Vu - Vlv - Vrv)

    Vom = (Pmv - Ptv) / P["Rmv"] if Pmv > Ptv else 0.0

    Gs, taus, tauv, Gv = P["Gs"], P["taus"], P["tauv"], P["Gv"]
    fesmin = P["fesmin"]
    dTsdt = (_sigma(Gs[0], hs1, fesmin) - dTs) / taus[0]
    dElsdt = (_sigma(Gs[1], hs1, fesmin) - dEls) / taus[1]
    dErsdt = (_sigma(Gs[2], hs1, fesmin) - dErs) / taus[2]
    dRspdt = (_sigma(Gs[3], hs2, fesmin) - dRsp) / taus[3]
    dRepdt = (_sigma(Gs[4], hs2, fesmin) - dRep) / taus[4]
    dRmpdt = (_sigma(Gs[5], hs2, fesmin) - dRmp) / taus[5]
    dVusvdt = (_sigma(Gs[6], hs3, fesmin) - dVusv) / taus[6]
    dVuevdt = (_sigma(Gs[7], hs3, fesmin) - dVuev) / taus[7]
    dVumvdt = (_sigma(Gs[8], hs3, fesmin) - dVumv) / taus[8]
    dTvdt = (Gv[0] * hp1 - dTv) / tauv[0]
    dElvdt = (Gv[1] * hp2 - dElv) / tauv[1]
    dErvdt = (Gv[2] * hp2 - dErv) / tauv[2]

    dudt = 1.0 / T
    dzetadt = 1.0 / act[0]
    dpsidt = 1.0 / P["Tim"]
    Ppadt = (1.0 / P["Cpa"]) * (Qor - Fpa)
    Fpadt = (1.0 / P["Lpa"]) * (Ppa - Ppp - P["Rpa"] * Fpa)
    Pppdt = (1.0 / P["Cpp"]) * (Fpa - (Ppp - Ppv) / P["Rpp"])
    Ppvdt = (1.0 / P["Cpv"]) * ((Ppp - Ppv) / P["Rpp"] - (Ppv - Pla) / P["Rpv"])
    Psadt = (1.0 / P["Csa"]) * (Qol - Fsa)
    Fsadt = (1.0 / P["Lsa"]) * (Psa - Psp - P["Rsa"] * Fsa)
    Pspdt = 1.0 / (P["Csp"] + P["Cep"] + P["Cmp"]) * (
        Fsa - (Psp_trans - (Psv - Pabd)) / Rsp
        - (Psp_trans - Pev) / Rep - (Psp - Pmv) / Rlb)
    Pevdt = (1.0 / P["Cev"]) * ((Psp_trans - Pev) / Rep
                                - (Pev - Ptv) / P["Rev"] - dVuevdt)
    Vmvdt = (Psp_trans - Pmv) / Rlb - Vom
    Ptvdt = (1.0 / P["Ctv"]) * (Vom + (Pev - Ptv) / P["Rev"]
                                + (Psv - Pabd - Ptv) / P["Rsv"]
                                - (Ptv - Pra) / P["Rtv"])
    Pradt = (1.0 / P["Cra"]) * ((Ptv - Pra) / P["Rtv"] - Qir)
    Vlvdt = Qil - Qol
    Vrvdt = Qir - Qor
    Ptildedt = (1.0 / P["taup"]) * (Psa + P["tauz"] * Psadt - Ptilde)
    Pldt = (1.0 / P["taucp"]) * (-Pl + Ppv - Pthor)
    flrdt = (1.0 / P["taulung"]) * (-flr + P["Gal"] * vlung)
    Vladt = (Ppv - Pla) / P["Rpv"] - Qil

    # ---- afferent limb, VNS injected here ----
    fbr = _sigmoid(Ptilde, P["Pn"], P["fbrmin"], P["fbrmax"], P["ka"])
    fcpr = P["fmaxl"] / (1.0 + np.exp((P["Ptn"] - Pl) / P["kl"]))
    I, fstim = ustar
    R = recruit(I, P)
    nfbr = fbr + R[0] * fstim
    nfcpr = fcpr + R[0] * fstim
    nflr = flr + R[1] * fstim
    afferent = np.array([nfbr, nfcpr, nflr])

    f = P["G"] @ afferent
    f[1] -= 25.0
    fes = act[3] + P["fesinf"] + (P["fes0"] - P["fesinf"]) * np.exp(-P["kes"] * f)

    # ---- vagal efferent limb ----
    frv = np.empty(3)
    for i in range(3):
        frv[i] = _sigmoid(afferent[i], P["midpt"][i], P["fmin"][i],
                          P["fmax"][i], P["k"][i])
    finput = P["kreceptor"] @ frv
    foNa = _sigmoid(finput[0], P["foutmidpt"][0], P["foutmin"][0],
                    P["foutmax"][0], P["foutk"][0]) - act[4]
    nfoNa = foNa + R[0] * fstim
    fDMV = _sigmoid(finput[1], P["foutmidpt"][1], P["foutmin"][1],
                    P["foutmax"][1], P["foutk"][1])
    fDMV_NActr = fDMV + _sigmoid(nfoNa, P["foutmidpt"][2], P["foutmin"][2],
                                 P["foutmax"][2], P["foutk"][2])

    dx = np.array([dudt, dzetadt, dTsdt, dTvdt, dElsdt, dElvdt, dErsdt,
                   dErvdt, dRspdt, dRepdt, dRmpdt, dVusvdt, dVuevdt,
                   dVumvdt, Ppadt, Fpadt, Pppdt, Ppvdt, Psadt, Fsadt,
                   Pspdt, Pevdt, Ptvdt, Pradt, Ptildedt, Pldt, flrdt,
                   Vmvdt, Vlvdt, Vrvdt, Vladt, dpsidt])
    aux = dict(T=T, Plv=Plv, Pla=Pla, Emaxlv=Emaxlv, Qol=Qol)
    return dx, fes[0], fes[1], fes[2], nfoNa, fDMV_NActr, aux


# ----------------------------------------------------------------- state

@dataclass
class PlantState:
    """Full plant state: 32 ODE states plus the five delay buffers."""
    x: np.ndarray
    hs1: deque
    hs2: deque
    hs3: deque
    hp1: deque
    hp2: deque

    def copy(self):
        return PlantState(self.x.copy(),
                          deque(self.hs1, maxlen=self.hs1.maxlen),
                          deque(self.hs2, maxlen=self.hs2.maxlen),
                          deque(self.hs3, maxlen=self.hs3.maxlen),
                          deque(self.hp1, maxlen=self.hp1.maxlen),
                          deque(self.hp2, maxlen=self.hp2.maxlen))


def default_x0():
    return np.array([
        0.0, 0.0,
        -0.10, 0.50,
        -0.12, 0.70,
        -0.20, 1.00,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        15.0, 80.0, 12.0, 8.0,
        95.0, 80.0, 85.0, 5.0, 4.0, 4.0,
        95.0, 8.0, 11.0,
        315.0, 120.0, 120.0, 60.0, 0.0,
    ])


def init_state(P, dt=DT, x0=None):
    ds, dv = P["ds"], P["dv"]
    n1, n2, n3, nv = (int(ds[0] / dt), int(ds[1] / dt), int(ds[2] / dt),
                      int(dv / dt))
    return PlantState(
        x=default_x0() if x0 is None else x0.copy(),
        hs1=deque([20.0] * n1, maxlen=n1),
        hs2=deque([3.0] * n2, maxlen=n2),
        hs3=deque([3.0] * n3, maxlen=n3),
        hp1=deque([10.0] * nv, maxlen=nv),
        hp2=deque([10.0] * nv, maxlen=nv),
    )


# ----------------------------------------------------------------- steppers

def step_cycle(st, ustar, P, act, dt=DT):
    """
    Integrate exactly one cardiac cycle, holding ustar = (I_stim, f_stim)
    constant. This is one decision step of the semi-Markov control problem.

    Returns (new_state, measurements). Measurements split into what an
    implant can read and what only the simulator knows:

      observable   HR, MAP, SBP, DBP, T
      hidden       LVEDP, Pla_mean, EDV, ESV, SV, EF, CO, Emaxlv

    LVEDP is read at end-diastole, where phi = 0 and Equation (1) collapses
    to the EDPVR term, so it is exactly P0_lv*(exp(kE_lv*V_ED) - 1).
    """
    x = st.x
    hs1, hs2, hs3, hp1, hp2 = st.hs1, st.hs2, st.hs3, st.hp1, st.hp2

    psa, vlv, plv, pla, ppa, emax, tser = [], [], [], [], [], [], []
    nmax = int(CYCLE_TIMEOUT / dt)
    done = False

    for _ in range(nmax):
        dx, f1, f2, f3, p1, p2, aux = derivatives(
            x, ustar, act, hs1[0], hs2[0], hs3[0], hp1[0], hp2[0], P)

        psa.append(x[18]); vlv.append(x[28]); ppa.append(x[14])
        plv.append(aux["Plv"]); pla.append(aux["Pla"])
        emax.append(aux["Emaxlv"]); tser.append(aux["T"])

        x = x + dt * dx
        if x[1] >= 1.0:
            x[1] = 0.0
        if x[31] >= 1.0:
            x[31] = 0.0
        hs1.append(f1); hs2.append(f2); hs3.append(f3)
        hp1.append(p1); hp2.append(p2)

        if x[0] >= 1.0:
            x[0] = 0.0
            done = True
            break

    if not np.all(np.isfinite(x)):
        raise RuntimeError("plant diverged")

    psa = np.asarray(psa); vlv = np.asarray(vlv)
    plv = np.asarray(plv); pla = np.asarray(pla); ppa = np.asarray(ppa)

    EDV = float(vlv.max())
    ESV = float(vlv.min())
    SV = EDV - ESV
    T = float(np.mean(tser))
    HR = 60.0 / T
    SBP = float(psa.max())
    DBP = float(psa.min())
    LVEDP = float(P["Polv"] * (np.exp(P["Kelv"] * EDV) - 1.0))

    # Pulmonary artery pressures. PAPd is the congestion signal a CardioMEMS
    # class implant actually reports, and it tracks PCWP/LVEDP without needing
    # a sensor in the left heart. It is the only congestion channel here with
    # an existing chronic implantable device behind it.
    PAPs = float(ppa.max())
    PAPd = float(ppa.min())

    meas = dict(
        HR=HR, MAP=(SBP + 2.0 * DBP) / 3.0, SBP=SBP, DBP=DBP, T=T,
        LVEDP=LVEDP, Pla_mean=float(pla.mean()), Plv_ed=float(plv[vlv.argmax()]),
        PAPd=PAPd, PAPs=PAPs, PAPm=(PAPs + 2.0 * PAPd) / 3.0,
        EDV=EDV, ESV=ESV, SV=SV, EF=SV / max(EDV, 1e-6),
        CO=SV * HR / 60.0 / 1000.0 * 60.0,     # L/min
        Emaxlv=float(np.mean(emax)),
        n_steps=len(psa), complete=done,
    )
    return PlantState(x, hs1, hs2, hs3, hp1, hp2), meas


def settle(P, act, seconds=200.0, dt=DT, ustar=(0.0, 0.0), st=None,
           verbose=False):
    """Run to steady state with no stimulation and return (state, baseline)."""
    st = init_state(P, dt) if st is None else st
    t, last = 0.0, None
    while t < seconds:
        st, meas = step_cycle(st, ustar, P, act, dt)
        t += meas["T"]
        last = meas
        if verbose and int(t) % 50 == 0:
            print(f"  t={t:6.1f}s  HR={meas['HR']:5.1f}  "
                  f"MAP={meas['MAP']:6.1f}  LVEDP={meas['LVEDP']:5.2f}")
    return st, last


def baseline(P, act, seconds=200.0, average_cycles=10, dt=DT):
    """Converged operating point, averaged over the last N cycles."""
    st = init_state(P, dt)
    t = 0.0
    hist = []
    while t < seconds:
        st, meas = step_cycle(st, (0.0, 0.0), P, act, dt)
        t += meas["T"]
        hist.append(meas)
    tail = hist[-average_cycles:]
    keys = ["HR", "MAP", "SBP", "DBP", "LVEDP", "Pla_mean", "PAPd", "PAPs",
            "PAPm", "EDV", "ESV", "SV", "EF", "CO", "Emaxlv", "T"]
    return st, {k: float(np.mean([h[k] for h in tail])) for k in keys}
