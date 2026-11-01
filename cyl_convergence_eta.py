# 137 RE8 convergence (2026-11-01): cylinder vs flat control with Gauss-Legendre quadrature in ln k on [1e-3, 60], separately in the number
# of nodes (30, 60) and the radial lattice (150, 250 sites), R = 10.5, 20.5, 40.5. Outside run_all (each run several minutes).
# Results in cyl_convergence_eta.out. Supersedes the trapezoid-grid runs (cyl_dense_eta.py and the first version of this file), whose
# flat control clamped S_1 beyond k = 60 or used unconverged weights.
# Run: ~/.venvs/jev/bin/python cyl_convergence_eta.py grid | radius | box | box20
import numpy as np, time, sys
from numpy.polynomial.legendre import leggauss
import os; src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ga_rindler_eta_check.py')).read()
exec(src[src.index("def chain_half"):src.index("# ---------- RE1")])
def S_rad(K, n):
    w, V = np.linalg.eigh(K)
    X = 0.5 * (V / np.sqrt(w)) @ V.T; P = 0.5 * (V * np.sqrt(w)) @ V.T
    nu = np.sqrt(np.clip(np.linalg.eigvals(X[:n, :n] @ P[:n, :n]).real, 0.25, None)); nu = nu[nu > 0.5 + 1e-12]
    return np.sum((nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5))
ktab = np.geomspace(1e-4, 80, 260); Stab = np.array([chain_half(k) for k in ktab])
p_tab = np.polyfit(np.log(ktab[-20:]), np.log(Stab[-20:]), 1)[0]
def S1i(k):
    k = np.asarray(k, float)
    inside = np.exp(np.interp(np.log(np.minimum(np.maximum(k, ktab[0]), ktab[-1])), np.log(ktab), np.log(Stab)))
    return np.where(k > ktab[-1], Stab[-1] * (k / ktab[-1])**p_tab, inside)
def run(Ns, npts, n, lo=1e-3, hi=60.0):
    x, w = leggauss(npts); a, b = np.log(lo), np.log(hi)
    kc = np.exp(0.5 * (b - a) * x + 0.5 * (b + a)); wc = 0.5 * (b - a) * w * kc     # dk = k d ln k
    R = n + 0.5
    jc = np.arange(1, Ns + 1, dtype=float); bd = ((jc + 0.5) + (jc - 0.5)) / jc; off = -(jc[:-1] + 0.5) / np.sqrt(jc[:-1] * jc[1:])
    def per_area(f):
        tot = 0.0
        for m in range(0, 6000):
            sm = np.sum(wc * np.array([f(m, k) for k in kc])) / np.pi
            tot += sm * (1 if m == 0 else 2)
            if m > 20 and sm < 1e-9 * tot: break
        return tot / (2 * np.pi * R)
    fl = per_area(lambda m, k: S1i(np.hypot(m / R, k)))
    cy = per_area(lambda m, k: S_rad(np.diag(bd + m**2 / jc**2 + k**2) + np.diag(off, 1) + np.diag(off, -1), n))
    return fl, cy
mode = sys.argv[1]
t = time.time()
if mode == 'grid':
    for npts in (30, 60, 120):
        fl, cy = run(150, npts, 20); print(f"R=20.5 Ns=150 GL{npts}: flat {fl:.7f} cyl {cy:.7f} rel {cy/fl-1:.3e} t={time.time()-t:.0f}", flush=True)
    fl, cy = run(150, 60, 20, lo=3e-4); print(f"R=20.5 Ns=150 GL60 lo=3e-4: rel {cy/fl-1:.3e}", flush=True)
elif mode == 'radius':
    for n in (10, 40):
        fl, cy = run(150, 60, n); print(f"R={n+0.5} Ns=150 GL60: flat {fl:.7f} cyl {cy:.7f} rel {cy/fl-1:.3e} t={time.time()-t:.0f}", flush=True)
if mode == 'box':
    for Ns in (250,):
        fl, cy = run(Ns, 30, 40); print(f"R=40.5 Ns={Ns} GL30: flat {fl:.7f} cyl {cy:.7f} rel {cy/fl-1:.3e} t={time.time()-t:.0f}", flush=True)
if mode == 'box20':
    fl, cy = run(250, 30, 20); print(f"R=20.5 Ns=250 GL30: rel {cy/fl-1:.3e} t={time.time()-t:.0f}", flush=True)
