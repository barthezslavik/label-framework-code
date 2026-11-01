# 86: program item 2 -- growth of dust perturbations with the only lowest-order dust-scalar coupling m(phi) = m0 e^(beta phi) (paper1 v2),
# and the test chain: DESI phantom -> m(phi) -> S8. GC1-GC4. Units 8 pi G = 1.
# Background: phi'' + 3H phi' = -beta rho_m, rho_m ~ a^-3 e^(beta phi), 3H^2 = rho_m + phi'^2/2 + V0 (V0 tuned to Omega_m = 0.3 today).
# Sub-horizon growth of dust (light phi): delta'' + (2H + beta phi') delta' = (1 + 2 beta^2) rho_m delta / 2 (momentum m v with varying m gives
# the extra friction beta phi'; scalar exchange gives G_eff = G (1 + 2 beta^2), 82 CC3). Same early amplitude (CMB) as beta = 0.
# Run: ~/.venvs/jev/bin/python ga_growth_coupling_check.py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

def model(beta, ai=1e-3, rmi=1.0, target=0.3):
    def run(V0):
        def f(N, y):
            ph, pd, d, dd = y
            rm = rmi * np.exp(-3 * (N - np.log(ai)) + beta * ph); h = np.sqrt((rm + 0.5 * pd**2 + V0) / 3)
            return [pd / h, (-3 * h * pd - beta * rm) / h, dd / h, (-(2 * h + beta * pd) * dd + 0.5 * rm * (1 + 2 * beta**2) * d) / h]
        s = solve_ivp(f, [np.log(ai), 0.0], [0, 0, ai, ai * np.sqrt(rmi / 3)], rtol=1e-9, atol=1e-12, dense_output=True)
        N = np.linspace(np.log(ai), 0, 4001); ph, pd, d, dd = s.sol(N)
        rm = rmi * np.exp(-3 * (N - np.log(ai)) + beta * ph); h2 = (rm + 0.5 * pd**2 + V0) / 3
        return N, rm, h2, d
    lo, hi = 1e-9, 1e-4
    for _ in range(60):
        V0 = np.sqrt(lo * hi); N, rm, h2, d = run(V0)
        lo, hi = (V0, hi) if rm[-1] / (3 * h2[-1]) > target else (lo, V0)
    rde = 3 * h2 - rmi * np.exp(-3 * (N - np.log(ai)))
    w = -1 - np.gradient(np.log(np.abs(rde)), N) / 3
    at = lambda z: w[np.argmin(abs(N + np.log(1 + z)))]
    return d[-1], at(0.0), at(0.5), at(1.0)

D0 = model(0.0)[0]
res = {b: model(b) for b in (0.03, 0.05, 0.1)}
for b, (D, w0, w05, w1) in res.items():
    print(f"   beta = {b}: sigma8/sigma8_LCDM = {D / D0:.4f}, inferred w(z=0, 0.5, 1) = {w0:.3f}, {w05:.3f}, {w1:.3f}")

# GC1 beta = 0 reproduces LCDM (w = -1, same growth) -- pipeline check.
D00, w00, _, w10 = model(0.0)
ok(f"GC1 beta = 0: w = {w00:.4f}, {w10:.4f}; growth ratio 1", abs(w00 + 1) < 1e-3 and abs(w10 + 1) < 1e-3 and abs(D00 / D0 - 1) < 1e-9)

# GC2 the coupling that makes dark energy look phantom also enhances growth: every beta > 0 gives sigma8 above LCDM at fixed early amplitude.
ok("GC2 apparent phantom and enhanced growth go together: sigma8 ratio > 1 and w(z=1) < -1 for every beta tested",
   all(D / D0 > 1 and w1 < -1 for D, w0, w05, w1 in res.values()))

# GC3 DESI depth: CPL best fits give w(z=1) = w0 + wa/2 = -1.11 (DESI+CMB+Pantheon+, -0.85/-0.52) to -1.14 (+DES, -0.82/-0.64) and -1.28
#     (DESI+CMB, -0.44/-1.67) (doc 70). The beta that gives w(z=1) = -1.11 raises sigma8 by the factor printed.
b11 = brentq(lambda b: model(b)[3] + 1.11, 0.02, 0.1, xtol=1e-4)
r11 = model(b11)[0] / D0
b128 = brentq(lambda b: model(b)[3] + 1.28, 0.02, 0.12, xtol=1e-4)
r128 = model(b128)[0] / D0
print(f"   w(z=1) = -1.11 needs beta = {b11:.3f}, sigma8 x{r11:.3f}; w(z=1) = -1.28 needs beta = {b128:.3f}, sigma8 x{r128:.3f}")
ok(f"GC3 matching DESI's past phantom depth with m(phi) raises sigma8 by {100 * (r11 - 1):.1f}%..{100 * (r128 - 1):.1f}%", r11 > 1.03 and r128 > r11)

# GC4 test: weak lensing (KiDS-Legacy + DES Y3 joint, arXiv:2503.19442) S8 = 0.818 +0.012 -0.014, consistent with Planck LCDM (~0.83) at ~1 sigma.
#     Scaling the Planck-LCDM S8 ~ 0.83 by the growth factor (same Omega_m) gives the coupled prediction; deviation in units of the lensing error.
S8L, eS8 = 0.818, 0.013
pred = (0.83 * r11, 0.83 * r128)
dev = ((pred[0] - S8L) / eS8, (pred[1] - S8L) / eS8)
ok(f"GC4 coupled S8 = {pred[0]:.3f}..{pred[1]:.3f} vs lensing {S8L} +- {eS8}: {dev[0]:.1f}..{dev[1]:.1f} sigma high -- the native explanation of a DESI phantom is disfavoured by lensing",
   dev[0] > 2)
