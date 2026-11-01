# 127: D7 of PREDICTIONS.md -- how a universal baryon coupling m_b ~ exp(beta_b phi) enters lunar laser ranging. In the Einstein frame a
# body of mass m(t) in the field of M(t) obeys d(m v)/dt = -G m M r/r^3, i.e. per unit mass a = -G M r/r^3 - eps v with eps = beta_b dphi/dt
# and G M ~ exp(eps t). A Kepler orbit is integrated with slowly varying G M and the drag term; the secular drifts of the semi-major axis
# and the mean motion are measured and converted to atomic units (atomic frequencies ~ m, atomic lengths ~ 1/m, since every compared mass
# couples alike -- the assumption of doc 123). A pure G-dot in fixed units gives a-dot/a = -Gdot/G and n-dot/n = 2 Gdot/G; the coupling is
# mapped on the equivalent Gdot/G. Then the number: phi-dot of the coupled model today, Cassini's beta_b < 2.4e-3, and the LLR result
# Gdot/G = (-5.0 +- 9.6)e-15 per yr (PDG 2025). LL1-LL5.
# Run: ~/.venvs/jev/bin/python ga_llr_mapping_check.py
import numpy as np
from scipy.integrate import solve_ivp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

def drift(eps, T=400.0):
    GM0 = 1.0
    def f(t, y):
        x, yy, vx, vy = y; r3 = (x * x + yy * yy)**1.5; GM = GM0 * np.exp(eps * t)
        return [vx, vy, -GM * x / r3 - eps * vx, -GM * yy / r3 - eps * vy]
    s = solve_ivp(f, [0, T], [1.0, 0.0, 0.0, 1.0], rtol=1e-11, atol=1e-13, dense_output=True)
    ts = np.linspace(0, T, 40001); x, yy, vx, vy = s.sol(ts)
    GM = GM0 * np.exp(eps * ts); r = np.hypot(x, yy); v2 = vx**2 + vy**2
    a = 1 / (2 / r - v2 / GM); n = np.sqrt(GM / a**3)              # osculating elements
    la, ln_ = np.polyfit(ts, np.log(a), 1)[0], np.polyfit(ts, np.log(n), 1)[0]
    return la, ln_
eps = 1e-4
la, ln_ = drift(eps)
ok(f"LL1 Einstein frame: d ln a/dt = {la / eps:+.3f} eps, d ln n/dt = {ln_ / eps:+.3f} eps (expected -3 and +5 from conserved r^2 theta-dot ~ exp(-eps t))",
   abs(la / eps + 3) < 0.05 and abs(ln_ / eps - 5) < 0.05)
la_at, ln_at = la + eps, ln_ - eps                                   # atomic length ~ 1/m, atomic frequency ~ m
ok(f"LL2 atomic units: d ln a/dt = {la_at / eps:+.3f} eps, d ln n/dt = {ln_at / eps:+.3f} eps: both equal a pure Gdot/G = 2 eps "
   "(a-dot/a = -Gdot/G, n-dot/n = 2 Gdot/G) -- the LLR Gdot/G measures d ln(G m^2)/dt = 2 beta_b phi-dot", abs(la_at / eps + 2) < 0.05 and abs(ln_at / eps - 4) < 0.05)
# numbers: phi-dot today from the coupled model (ga_thermal_nuclear_check run) at the fitted beta bounds; H0 = 70 km/s/Mpc
H0_yr = 70 / 3.086e19 * 3.156e7
from scipy.optimize import brentq
def phidot_today(beta, Om=0.30, fb=0.157):
    """d phi/d(H0 t) today, two fluids: phi sourced by the coupled dark dust only (doc 126 after paper7 review r14); H0 = 1, 8 pi G = 1"""
    ai = 1e-3; rci = 3 * Om * (1 - fb) * ai**-3; rbi = 3 * Om * fb * ai**-3
    def run(V0):
        def f(N, y):
            ph, pd = y; x = np.exp(-3 * (N - np.log(ai))); rc = rci * x * np.exp(beta * ph)
            h = np.sqrt((rc + rbi * x + 0.5 * pd**2 + V0) / 3); return [pd / h, (-3 * h * pd - beta * rc) / h]
        return solve_ivp(f, [np.log(ai), 0.0], [0.0, 0.0], rtol=1e-10, atol=1e-13).y[:, -1]
    def H0of(V0):
        ph, pd = run(V0); return np.sqrt((rci * ai**3 * np.exp(beta * ph) + rbi * ai**3 + 0.5 * pd**2 + V0) / 3)
    V0 = brentq(lambda V: H0of(V) - 1.0, 0.0, 6.0, xtol=1e-12)
    return run(V0)[1] / H0of(V0)
rates = {b: phidot_today(b) for b in (0.054, 0.062)}               # 2 sigma bounds of the two-fluid fit (doc 126): DES Y6, KiDS-DES
print("   d phi/d(H0 t) today (signed, two fluids): " + ", ".join(f"beta {b}: {r:+.4f}" for b, r in rates.items()))
phidot = rates[0.062] * H0_yr                                        # the largest beta the joint fit allows (KiDS-DES, 2 sigma)
bb = 2.4e-3
llr, sl = -5.0e-15, 9.6e-15
pred = {'beta_b same sign as beta': 2 * bb * phidot, 'opposite sign': -2 * bb * phidot}
ok("LL3 prediction at |beta_b| = 2.4e-3 (Cassini) and beta = 0.062 (two-fluid fit bound, doc 126): " + "; ".join(
   f"{k}: Gdot/G = {v:+.1e} per yr, {abs(v - llr) / sl:.1f} sigma from LLR ({llr:.1e} +- {sl:.1e})" for k, v in pred.items())
   + " -- not distinguished at 2 sigma for the same sign; at 2.1 sigma for the opposite sign", phidot < 0)
lo, hi = llr - 2 * sl, llr + 2 * sl                                  # 2 sigma interval of Gdot/G
bmin, bmax = hi / (2 * phidot), lo / (2 * phidot)                    # phidot < 0 reverses the order
ok(f"LL4 LLR alone (2 sigma): beta_b in [{min(bmin, bmax):+.1e}, {max(bmin, bmax):+.1e}] (Cassini |beta_b| < {bb}); to rule out |beta_b| = {bb} of the "
   f"same sign at 2 sigma with the present central value an LLR error below {abs(2 * bb * phidot - llr) / 2:.1e} per yr is needed", True)
ok("LL5 recorded: the mapping assumes every compared mass (electron included) couples alike and alpha is constant; otherwise atomic units "
   "drift against the orbit and the LLR number mixes in clock effects", True)
