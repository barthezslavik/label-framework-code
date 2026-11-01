# 123: thermodynamic and nuclear consequences of the label sector. (i) Is the scalar phi of m(phi) or the label dust ever in thermal
# equilibrium with the SM plasma? phi couples with gravitational strength, so Gamma/H ~ (T/M_P)^3; dust may couple through even scalars
# (doc 122) with label coefficients. (ii) Number-changing equilibrium needs annihilation, absent when the label count is constant (doc 125):
# no freeze-out, abundance = label; phi adds no radiation: N_eff = 3.044. (iii) Varying masses: with a universal baryon coupling beta_b, m_b ~ exp(beta_b phi) drifts with the
# background phi of m(phi); computed: Delta phi from a = 1e-3 to today and today's rate d phi/dt for the beta that S8 allows (doc 117),
# hence bounds on Delta ln(G m_b^2) and on the present effective G-dot/G (lunar ranging); mass ratios do not move under a universal coupling. Units 8 pi G = 1. TN1-TN5.
# Run: ~/.venvs/jev/bin/python ga_thermal_nuclear_check.py
import numpy as np
from scipy.integrate import solve_ivp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

Mp = 2.435e18                                                        # reduced Planck mass, GeV
g_star = 106.75
for T in (1e3, 1e9, 1e15):
    H = np.sqrt(np.pi**2 * g_star / 90) * T**2 / Mp
    Gam = T**5 / Mp**4                                                # gravitational-strength 2 -> 2 rate
    print(f"   T = {T:.0e} GeV: Gamma/H = {Gam / H:.1e}")
ok("TN1 (conditional) gravitational scattering of phi (Gamma = T^5/M_P^4) gives Gamma/H ~ (T/M_P)^3 < 1 below the Planck scale -- this bound "
   "is for that channel only (the baryon channel is TN1b); a coupling a(phi) F^2 has a free (label) slope, so N_eff = 3.044 holds if that slope is "
   "Planck-suppressed and phi is absent from the initial data -- the photon channel is not computed",
   (1e15**5 / Mp**4) / (np.sqrt(np.pi**2 * g_star / 90) * 1e15**2 / Mp) < 1e-6)
# TN1b (review r6) the baryon channel: Yukawa y = beta_b m_N / M_P (reduced Planck mass), Compton-like gamma N -> N phi with
#     sigma ~ alpha y^2 / max(T, m_N)^2 (order of magnitude), n_N ~ T^3 above m_N and eta T^3 below; Gamma/H scanned over 1e-3 GeV - 1e15 GeV
alpha, mN, eta, bb_ = 1 / 137, 0.938, 6e-10, 2.4e-3
yN = bb_ * mN / Mp
Ts = np.logspace(-3, 15, 400)
nN = np.where(Ts > mN, Ts**3, eta * Ts**3)
Gb = nN * alpha * yN**2 / np.maximum(Ts, mN)**2
Hs = np.sqrt(np.pi**2 * g_star / 90) * Ts**2 / Mp
ok(f"TN1b baryon channel: max Gamma/H = {np.max(Gb / Hs):.1e} (at T = {Ts[np.argmax(Gb / Hs)]:.1e} GeV): phi does not thermalise through baryons either "
   "(order-of-magnitude rate; the photon channel a(phi) F^2 has a label slope and is not computed)", np.max(Gb / Hs) < 1e-10)
ok("TN2 (rewritten after review r1) label dust may reach kinetic contact with the plasma through the even scalar couplings of doc 122 "
   "((J.J) S, (J.J) H^+H), whose coefficients are labels -- not computed; what does not happen is number-changing equilibrium: with the label "
   "count constant (doc 125, embeddings) there is no freeze-out, the abundance is a label and has no relation to a cross-section", True)
def run(beta, ai=1e-3, target=0.3):
    def sol(V0):
        def f(N, y):
            ph, pd = y
            rm = np.exp(-3 * (N - np.log(ai)) + beta * ph); h = np.sqrt((rm + 0.5 * pd**2 + V0) / 3)
            return [pd / h, (-3 * h * pd - beta * rm) / h]
        s = solve_ivp(f, [np.log(ai), 0.0], [0, 0], rtol=1e-10, atol=1e-13, dense_output=True)
        ph, pd = s.sol(0.0); rm = np.exp(3 * np.log(ai) + beta * ph); h2 = (rm + 0.5 * pd**2 + V0) / 3
        return ph, pd, h2, rm
    lo, hi = 1e-9, 1e-4
    for _ in range(60):
        V0 = np.sqrt(lo * hi); ph, pd, h2, rm = sol(V0)
        lo, hi = (V0, hi) if rm / (3 * h2) > target else (lo, V0)
    return ph, pd, np.sqrt(h2)                                       # phi, d phi/dt (units of the code), H0 (code)
beta = 0.036
ph, pd, H0c = run(beta)
dphi = abs(ph); rate = abs(pd) / H0c                                 # d phi / d(H0 t)
H0_yr = 70 / 3.086e19 * 3.156e7
bb = 2.4e-3
ok(f"TN3 m(phi) at beta = {beta}: Delta phi (a = 1e-3 -> 1) = {dphi:.2e} (reduced Planck units), today d phi/d(H0 t) = {rate:.2e}", dphi > 0)
Gdot = 2 * bb * rate * H0_yr
ok(f"TN4 universal baryon coupling at the bound beta_b = {bb}: baryon masses move together; mass ratios stay constant only if every compared "
   f"mass, the electron included, couples alike (assumption, not checked here); what moves is G m_b^2: Delta ln(G m_b^2) since recombination <= {2 * bb * dphi:.1e}, present |d ln(G m^2)/dt| <= {Gdot:.1e} per "
   f"year; lunar ranging gives G-dot/G = (-5.0 +- 9.6)e-15 per year (PDG 2025), comparable precision; the mapping of d ln(G m^2)/dt onto "
   "the ephemeris is not done", Gdot < 1e-13)
ok("TN5 recorded (negative): alpha is a label; a phi-dependent coupling a(phi) F^2 is K-blind as well (phi is dimensionless), so the "
   "framework neither forbids nor predicts a varying alpha: its slope d ln alpha/d phi would be one more label", True)
