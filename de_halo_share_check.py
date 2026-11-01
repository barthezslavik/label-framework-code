# 99: the test a0 = g/f without rotation curves (doc 96 KA1). Halo masses of blue (disc) central galaxies from weak lensing (Mandelbaum et
# al. 2016, Table B1, transcribed into DATA_ROOT/mandelbaum2016/table_B1_blue.txt), gas from SPARC (mean log M_b/M_* vs log M_*, M_b =
# M_* + 1.33 M_HI, Upsilon_3.6 = 0.5 -- a Chabrier-like normalisation). For each bin: r200m from 200 rho_m(z_eff), g = G M200m/r200m^2,
# f = M_b/M200m, and a0_test = g/f (x 1.18 if the measured speed is the Emden plateau rather than v_c(r200m), KA7). HS1-HS4.
# Run: ~/.venvs/jev/bin/python -I de_halo_share_check.py DATA_ROOT SPARC_TABLE
import sys, os
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

root, sparc = sys.argv[1], sys.argv[2]
t = np.loadtxt(os.path.join(root, 'mandelbaum2016', 'table_B1_blue.txt'))
lms, lmh, ep, em, zeff = t[:, 0], t[:, 2], t[:, 3], t[:, 4], t[:, 5]
h, Om, fcos = 0.673, 0.315, 0.0493 / 0.315
# gas fraction from SPARC: fit log(M_b/M_*) = a + b (log M_* - 10)
L, MHI = [], []
for l in open(sparc):
    p = l.split()                                                 # data rows: name T D e_D f_D Inc e_Inc L e_L Reff SBeff Rdisk SBdisk MHI ...
    if len(p) < 18: continue
    try:
        Lv, Mh = float(p[7]), float(p[13]); int(p[1])
    except ValueError:
        continue
    L.append(Lv); MHI.append(Mh)
L, MHI = np.array(L) * 1e9, np.array(MHI) * 1e9
Ms = 0.5 * L; Mb = Ms + 1.33 * MHI
sel = Ms > 3e9
cg = np.polyfit(np.log10(Ms[sel]) - 10, np.log10(Mb[sel] / Ms[sel]), 1)
gas = lambda lm: 10**np.polyval(cg, lm - 10)
print(f"   SPARC ({sel.sum()} galaxies with M_* > 3e9): log M_b/M_* = {cg[1]:.3f} + {cg[0]:.3f} (log M_* - 10)")
ok(f"HS1 gas correction from SPARC: M_b/M_* = {gas(10.24):.2f} at log M_* = 10.24, {gas(11.10):.2f} at 11.10", 1.0 < gas(11.1) < gas(10.24) < 3)

G = 4.301e-9; conv = 1e6 / 3.0857e22
Mh = 10**lmh / h                                                  # Msun
rho_m = 2.775e11 * h**2 * Om * (1 + zeff)**3                     # Msun/Mpc^3
r200 = (3 * Mh / (4 * np.pi * 200 * rho_m))**(1 / 3)
g200 = G * Mh / r200**2 * conv
Mbar = 10**lms * gas(lms)
f = Mbar / Mh
a0t = g200 / f
for i in range(len(lms)):
    print(f"   log M_* {lms[i]:.2f}: log M200m {np.log10(Mh[i]):.2f} Msun, g200 = {g200[i]:.2e}, f = M_b/M200m = {f[i]:.3f} (r_b = {f[i] / fcos:.2f}), "
          f"a0_test = g/f = {a0t[i]:.2e} (+{(10**(4 / 3 * ep[i]) - 1) * 100:.0f}% / -{(1 - 10**(-4 / 3 * em[i])) * 100:.0f}% from the halo mass)")
# g/f scales as M200m^(2/3)/M200m^(-1)... a0 = G M_b ... : a0_test = G Mh/r^2 * Mh/Mb ~ Mh^(4/3)/Mb, so its error is (4/3) of the log halo-mass error
w = 1 / ((4 / 3 * (ep + em) / 2)**2)
la = np.sum(w * np.log10(a0t)) / np.sum(w); ela = 1 / np.sqrt(np.sum(w))
ok(f"HS2 weighted mean a0_test = g/f = {10**la:.2e} m/s^2 (x/ {10**ela:.2f}); with the Emden plateau factor 1.18: {1.18 * 10**la:.2e}; observed a0 = 1.2e-10",
   True)
dev = (la - np.log10(1.2e-10)) / ela; dev2 = (la + np.log10(1.18) - np.log10(1.2e-10)) / ela
ok(f"HS3 deviation from 1.2e-10: {dev:+.1f} sigma (v_c(r200m)), {dev2:+.1f} sigma (plateau); retained share r_b = f/f_cos = "
   f"{', '.join(f'{x:.2f}' for x in f / fcos)} (KA6 required 0.20-0.86)", True)
ok(f"HS4 recorded: the test assumes the circular speed stays flat to r200m today; a declining outer curve raises the true v_flat/v_c(r200m) "
   f"factor above 1.18 and a0_test with it", True)

# HS5 the lensing virial acceleration against the spectrum (LB1/KA4, nu = 1 shell collapse at the same halo mass; definitions differ:
#     200 rho_m today vs 18 pi^2 rho_m at collapse, so only the order and the flatness in mass are compared)
import contextlib, io
kc = {'__name__': 'kc', '__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read(), kc)
gpred = np.array([kc['gv'](M, 1.0)[1] for M in Mh])
print("   " + ", ".join(f"log M {np.log10(M):.2f}: lensing {g:.2e} vs spectrum {gp:.2e}" for M, g, gp in zip(Mh, g200, gpred)))
ok(f"HS5 lensing g200 / spectrum g_v (nu = 1): {', '.join(f'{x:.2f}' for x in g200 / gpred)} -- same order; the mass trend is not comparable: "
   f"at fixed overdensity g200 ~ M^(1/3) by definition ({np.log10(g200.max() / g200.min()):.2f} dex over {np.log10(Mh.max() / Mh.min()):.2f} dex)",
   0.3 < np.min(g200 / gpred) and np.max(g200 / gpred) < 3)
# HS6 systematics budget (recorded): a0_test ~ M200m^(4/3)/M_b; stellar-mass systematics (IMF, Upsilon) ~0.15 dex move a0_test by 0.15 dex,
#     the halo-mass definition (mean vs typical, NFW correction 1.1-1.5 in Table B1) by up to (4/3) 0.15 = 0.2 dex.
ok(f"HS6 systematics: 0.15 dex (stellar mass) and up to 0.2 dex (halo-mass correction) against the {abs(la - np.log10(1.2e-10)):.2f} dex offset "
   f"(v_c(r200m)) / {abs(la + np.log10(1.18) - np.log10(1.2e-10)):.2f} dex (plateau)", True)

# HS7 consistency of the bins with one value, and the trend with stellar mass (review 2026-10-22, M4)
chi2c = float(np.sum(w * (np.log10(a0t) - la)**2)); A_ = np.vstack([np.ones_like(lms), lms - 10.8]).T
cov = np.linalg.inv(A_.T @ (w[:, None] * A_)); beta = cov @ (A_.T @ (w * np.log10(a0t)))
ok(f"HS7 the five bins are not consistent with one value: chi2 = {chi2c:.1f} for {len(lms) - 1} dof; trend d log(g/f)/d log M_* = "
   f"{beta[1]:+.2f} +- {np.sqrt(cov[1, 1]):.2f}; scatter-inflated error of the mean x/ {10**(ela * np.sqrt(chi2c / (len(lms) - 1))):.2f}", chi2c > len(lms) - 1)
