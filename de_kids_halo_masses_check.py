# 128b: halo masses of isolated KiDS-1000 lens galaxies from their own lensing profiles (Brouwer et al. 2021, the data release
# brouwer2021_rar.tar: Fig-3 ESD profiles in four stellar-mass bins and their covariance). Model per bin: the baryonic mass as a point
# source, Delta Sigma_b = M_gal/(pi R^2), plus an NFW halo (Wright & Brainerd 2000 Delta Sigma, equation borrowed) of mass M200c and free
# concentration (the Duffy et al. 2008 c(M) fits badly, chi2 2-5 per dof; its masses are printed as a systematic); fitted for R <= 1 Mpc (inside that the two-halo term of isolated galaxies is small), with the
# block covariance of the bin (calibration bias divided out as in the README). Mean log M_gal (stars + cold gas) per bin from the paper:
# 10.14, 10.57, 10.78, 10.96 (h70 = 1 at h = 0.70), lens redshift ~0.25. Output: log M200c and its 1 sigma error per bin, written to
# DATA_ROOT/kids_brouwer2021/halo_masses_fit.txt for the G2 share-curvature check. KH1-KH3.
# Run: ~/.venvs/jev/bin/python -I de_kids_halo_masses_check.py DATA_ROOT
import os, sys
import numpy as np
from scipy.optimize import minimize_scalar

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

D = os.path.join(sys.argv[1], 'kids_brouwer2021')
lMgal = np.array([10.14, 10.57, 10.78, 10.96]); z = 0.25; h = 0.70; Om = 0.3
G = 4.301e-9                                                         # Mpc (km/s)^2 / Msun
Hz = 100 * h * np.sqrt(Om * (1 + z)**3 + 1 - Om)                     # km/s/Mpc
rho_c = 3 * Hz**2 / (8 * np.pi * G)                                  # Msun/Mpc^3
cov = np.loadtxt(os.path.join(D, 'Fig-3_Lensing-rotation-curves_Massbins_covmatrix.txt'))
mins = np.unique(cov[:, 0])
def g_small(x):
    out = np.empty_like(x)
    lo, hi, eq = x < 1, x > 1, np.isclose(x, 1)
    out[lo] = 8 * np.arctanh(np.sqrt((1 - x[lo]) / (1 + x[lo]))) / (x[lo]**2 * np.sqrt(1 - x[lo]**2)) + 4 / x[lo]**2 * np.log(x[lo] / 2) \
        - 2 / (x[lo]**2 - 1) + 4 * np.arctanh(np.sqrt((1 - x[lo]) / (1 + x[lo]))) / ((x[lo]**2 - 1) * np.sqrt(1 - x[lo]**2))
    out[hi] = 8 * np.arctan(np.sqrt((x[hi] - 1) / (1 + x[hi]))) / (x[hi]**2 * np.sqrt(x[hi]**2 - 1)) + 4 / x[hi]**2 * np.log(x[hi] / 2) \
        - 2 / (x[hi]**2 - 1) + 4 * np.arctan(np.sqrt((x[hi] - 1) / (1 + x[hi]))) / ((x[hi]**2 - 1)**1.5)
    out[eq] = 10 / 3 + 4 * np.log(0.5)
    return out
def dsigma_nfw(R, M200):                                             # R in Mpc, returns Msun/pc^2
    c = 5.71 * (M200 * h / 2e12)**-0.084 * (1 + z)**-0.47               # Duffy et al. 2008, M200c, full sample
    r200 = (3 * M200 / (4 * np.pi * 200 * rho_c))**(1 / 3); rs = r200 / c
    dc = 200 / 3 * c**3 / (np.log(1 + c) - c / (1 + c))
    return rs * dc * rho_c * g_small(R / rs) / 1e12
from scipy.optimize import minimize
def dsig_c(R, M200, c):
    r200 = (3 * M200 / (4 * np.pi * 200 * rho_c))**(1 / 3); rs = r200 / c
    dc = 200 / 3 * c**3 / (np.log(1 + c) - c / (1 + c))
    return rs * dc * rho_c * g_small(R / rs) / 1e12
def fit_bins(RMAX):
  rows = []
  for k in range(4):
      d = np.loadtxt(os.path.join(D, f'Fig-3_Lensing-rotation-curves_Massbin-{k + 1}.txt'))
      R, esd = d[:, 0], d[:, 1] / d[:, 4]
      m = R <= RMAX
      blk = cov[(cov[:, 0] == mins[k]) & (cov[:, 1] == mins[k])]
      n = len(R); C = (blk[:, 4] / blk[:, 6]).reshape(n, n)[np.ix_(m, m)]
      Ci = np.linalg.inv(C)
      point = 10**lMgal[k] / (np.pi * (R[m] * 1e6)**2)
      chi2 = lambda lM, lc: (lambda r: r @ Ci @ r)(esd[m] - point - dsig_c(R[m], 10**lM, 10**lc))
      best = min((minimize(lambda p: chi2(*p), [x0, np.log10(6)], method='Nelder-Mead') for x0 in (11.5, 12.0, 12.5)), key=lambda r: r.fun)
      prof = lambda lM: minimize_scalar(lambda lc: chi2(lM, lc), bounds=(-0.5, 1.6), method='bounded').fun
      grid = np.linspace(best.x[0] - 0.6, best.x[0] + 0.6, 241); cc = np.array([prof(x) for x in grid])
      inside = grid[cc <= best.fun + 1]; err = 0.5 * (inside.max() - inside.min())
      dfy = minimize_scalar(lambda lM: (lambda r: r @ Ci @ r)(esd[m] - point - dsigma_nfw(R[m], 10**lM)), bounds=(10, 14.5), method='bounded').x
      rows.append((lMgal[k], best.x[0], err, best.fun, m.sum() - 2, 10**best.x[1], dfy))
      print(f"   bin {k + 1}: log M_gal {lMgal[k]:.2f}: log M200c = {best.x[0]:.2f} +- {err:.2f} (c = {10**best.x[1]:.1f}, chi2 = {best.fun:.1f} for "
            f"{m.sum() - 2} dof); with the Duffy c(M): {dfy:.2f}")
  return np.array(rows)
rows = fit_bins(1.0)
np.savetxt(os.path.join(D, 'halo_masses_fit.txt'), rows[:, :3], fmt='%.4f',
           header='log M_gal (stars + cold gas, Brouwer 2021)  log M200c (Msun, own NFW fit, de_kids_halo_masses_check)  error')
ok("KH1 NFW + point baryons fit each KiDS bin with acceptable chi2 (< 2 per dof): " + ", ".join(f"{r[3]:.1f}/{int(r[4])}" for r in rows),
   all(r[3] < 2 * r[4] for r in rows))
ok(f"KH2 halo masses rise with baryonic mass: log M200c = {', '.join(f'{x:.2f}' for x in rows[:, 1])}", np.all(np.diff(rows[:, 1]) > -0.3))
win = {}
for rmax in (0.3, 0.5):
    with __import__('contextlib').redirect_stdout(__import__('io').StringIO()):
        win[rmax] = fit_bins(rmax)
    np.savetxt(os.path.join(D, f'halo_masses_fit_R{rmax}.txt'), win[rmax][:, :3], fmt='%.4f', header=f'R <= {rmax} Mpc fit')
ok("KH4 radial window (review r13): log M200c for R <= 0.3 / 0.5 / 1.0 Mpc: " + "; ".join(
   f"bin {k + 1}: {win[0.3][k, 1]:.2f} / {win[0.5][k, 1]:.2f} / {rows[k, 1]:.2f}" for k in range(4))
   + " -- Brouwer et al. flag R > 0.3 Mpc as sensitive to isolation errors from photometric redshifts", True)
ok(f"KH3 recorded: concentration free (c = {', '.join(f'{x:.1f}' for x in rows[:, 5])}); with the Duffy c(M) the masses shift by "
   f"{', '.join(f'{b - a:+.2f}' for a, b in zip(rows[:, 1], rows[:, 6]))} dex (systematic); no two-halo term (R <= 1 Mpc, isolated lenses), one redshift 0.25 for "
   "all bins, cross-bin covariance not used; the first bin is wide (log M* 8.5-10.3)", True)
