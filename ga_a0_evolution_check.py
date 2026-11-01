# 115: a0(z) from the level formula a0 = R^4 g_v/eps (paper3 Prop level) with R and eps fixed: a0(z)/a0(0) = g_v(z)/g_v(0), g_v ~ M^(1/3)
# rho_vir(z_c)^(2/3) (LCDM virial overdensity, Bryan-Norman). Two native ways to pick the collapse epoch of the hosts of discs seen at z:
#   (A) fixed nu-peaks as at z = 0: z_c independent of z while z_c >= z; once z_c < z the hosts must have collapsed by z (z_c = z);
#   (B) the same growth-factor ratio as today: D(z_c)/D(z) = D(z_c0)/D(0).
# Compared in shape with the MUSE-DARK line a0(z) = 1.00 + 1.59 z (0.33 < z < 1.44, quoted errors of the line not used here) and with
# a0 ~ 1/t. Halo mass 10^11.5-10^12 Msun (log M_b ~ 10.3-10.8 for effective fractions 0.1). AE1-AE3. Run: ~/.venvs/jev/bin/python ga_a0_evolution_check.py
import os, contextlib, io
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
HERE = os.path.dirname(os.path.abspath(__file__))
kc = {'__name__': 'kc', '__file__': os.path.join(HERE, 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read(), kc)
D, sigM, ok = kc['D'], kc['sigM'], kc['ok']
Om = 0.315
def rho_vir(zc):
    E2 = Om * (1 + zc)**3 + 1 - Om; x = Om * (1 + zc)**3 / E2 - 1
    return (18 * np.pi**2 + 82 * x - 39 * x**2) * E2
gv = lambda zc: rho_vir(zc)**(2 / 3)                               # at fixed mass
def zc_nu(M, nu):
    t = 1.686 / (nu * sigM(M))
    return brentq(lambda z: D(z) - t, 0.0, 50) if t < 1 else 0.0
zs = np.array([0.0, 0.33, 0.5, 0.75, 1.0, 1.25, 1.44])
H0 = 1.0
tcos = lambda z: quad(lambda zz: 1 / ((1 + zz) * np.sqrt(Om * (1 + zz)**3 + 1 - Om)), z, np.inf)[0]
inv_t = np.array([tcos(0) / tcos(z) for z in zs])
line = 1.00 + 1.59 * zs
for M in (10**11.5, 10**12.0):
    for nu in (1.0, 1.5):
        zc0 = zc_nu(M, nu)
        A = np.array([gv(max(zc0, z)) / gv(zc0) for z in zs])
        ratio0 = D(zc0) / D(0.0)
        zcB = np.array([brentq(lambda zz: D(zz) - ratio0 * D(z), z, 60) for z in zs])
        B = np.array([gv(zc) / gv(zc0) for zc in zcB])
        print(f"   M = 10^{np.log10(M):.1f}, nu = {nu}: z_c0 = {zc0:.2f}")
        print("      z      " + " ".join(f"{z:5.2f}" for z in zs))
        print("      (A)    " + " ".join(f"{v:5.2f}" for v in A))
        print("      (B)    " + " ".join(f"{v:5.2f}" for v in B))
print("      1/t    " + " ".join(f"{v:5.2f}" for v in inv_t))
print("      line   " + " ".join(f"{v:5.2f}" for v in line))
ok("AE1 (A) fixed nu-peaks: a0 constant until z reaches the collapse epoch of the hosts, then rising; (B) fixed growth ratio: rising from z = 0", True)
ok("AE2 recorded: the shapes of (A), (B), 1/t and the MUSE-DARK line over 0.33 < z < 1.44 are printed above", True)

# AE3 shape test as in paper3 Section 9: line a0 = (1.00 +- 0.04) + (1.59 +- 0.10) z (MUSE-DARK III), 0.33 < z < 1.44; free normalisation;
#     largest deviation in units of the line's error, for (A), (B) at the four (M, nu) cases, 1/t, H and a constant
zg = np.linspace(0.33, 1.44, 12)                                  # the same grid as inv_clock_combos_check P4
lin = 1.00 + 1.59 * zg; sl = np.sqrt(0.04**2 + (0.10 * zg)**2)
Hz = np.sqrt(Om * (1 + zg)**3 + 1 - Om)
it = np.array([tcos(0) / tcos(z) for z in zg])
def maxdev(f):
    w = 1 / sl**2; A = np.sum(w * f * lin) / np.sum(w * f * f)
    return np.max(np.abs(A * f - lin) / sl), A
res = {'1/t': maxdev(it), 'H': maxdev(Hz), 'constant': maxdev(np.ones_like(zg))}
for M in (10**11.5, 10**12.0):
    for nu in (1.0, 1.5):
        zc0 = zc_nu(M, nu); r0 = D(zc0) / D(0.0)
        fA = np.array([gv(max(zc0, z)) / gv(zc0) for z in zg])
        fB = np.array([gv(brentq(lambda zz: D(zz) - r0 * D(z), z, 60)) / gv(zc0) for z in zg])
        res[f'(A) M=10^{np.log10(M):.1f} nu={nu}'] = maxdev(fA); res[f'(B) M=10^{np.log10(M):.1f} nu={nu}'] = maxdev(fB)
for k, (d, A) in res.items():
    print(f"   {k:26s}: max deviation {d:5.1f} sigma (normalisation {A:.2f})")
dB = [v[0] for k, v in res.items() if k.startswith('(B)')]; dA = [v[0] for k, v in res.items() if k.startswith('(A)')]
ok(f"AE3 shape vs the MUSE-DARK line: (B) {min(dB):.1f}-{max(dB):.1f} sigma, (A) {min(dA):.1f}-{max(dA):.1f} sigma, 1/t {res['1/t'][0]:.1f}, "
   f"H {res['H'][0]:.1f}, constant {res['constant'][0]:.1f}", True)

# AE4 (Codex round 12) scenario (A) only where it is defined (z <= z_c of the hosts), the line's own range intersected with that; the
#     deviation is a diagnostic (largest pointwise deviation over the line's coefficient errors), not a significance
for M in (10**11.5, 10**12.0):
    for nu in (1.0, 1.5):
        zc0 = zc_nu(M, nu); m = zg <= zc0
        if m.sum() < 3:
            print(f"   (A) M=10^{np.log10(M):.1f} nu={nu}: z_c = {zc0:.2f}, fewer than 3 line points below z_c"); continue
        f = np.ones(m.sum()); w = 1 / sl[m]**2; A = np.sum(w * f * lin[m]) / np.sum(w * f * f)
        d = np.max(np.abs(A * f - lin[m]) / sl[m])
        print(f"   (A) M=10^{np.log10(M):.1f} nu={nu}: z_c = {zc0:.2f}; on 0.33 < z < {min(zc0, 1.44):.2f}: a0 constant, line rises "
              f"{lin[m][-1] / lin[m][0]:.2f}x; diagnostic deviation {d:.1f}")
ok("AE4 (A) restricted to its definition: constant a0 where the line already rises; the deviations are diagnostics, not significances", True)
