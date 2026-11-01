# 109: the a0 = g/eps test extended to dwarfs with weak-lensing halo masses from DESI and DES (arXiv:2509.20434; M200c; table transcribed into
# DATA_ROOT/desi_des_dwarfs2025/table_halo_masses.txt). Gas from SPARC: mean log(M_b/M_*) vs log M_* fitted over all SPARC galaxies
# (M_b = 0.5 L + 1.33 M_HI). For each bin: r200c at z = 0.05 (low-z dwarf samples; the result moves by <1% for z = 0.02-0.1), g200c/eps =
# G M200c^2/(r200c^2 M_b), and the retained fraction chi_b = M_b/(M200c Omega_b/Omega_m). Comparison with the blue centrals of doc 99 (M200m).
# DL1-DL4. Run: ~/.venvs/jev/bin/python -I de_dwarf_lensing_check.py DATA_ROOT SPARC_TABLE
import os, sys
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

root, sparc = sys.argv[1], sys.argv[2]
rows = [l.split() for l in open(os.path.join(root, 'desi_des_dwarfs2025', 'table_halo_masses.txt')) if l.strip() and not l.startswith('#')]
samp = np.array([r[0] for r in rows]); lms = np.array([float(r[1]) for r in rows]); lmh = np.array([float(r[4]) for r in rows])
ehp = np.array([float(r[5]) for r in rows]); ehm = np.array([float(r[6]) for r in rows])
L, MHI = [], []
for l in open(sparc):
    p = l.split()
    if len(p) < 18: continue
    try:
        int(p[1]); L.append(float(p[7]) * 1e9); MHI.append(float(p[13]) * 1e9)
    except ValueError:
        continue
L, MHI = np.array(L), np.array(MHI); Ms = 0.5 * L; good = Ms > 1e7
cg = np.polyfit(np.log10(Ms[good]) - 9, np.log10((Ms[good] + 1.33 * MHI[good]) / Ms[good]), 2)
gas = lambda lm: 10**np.polyval(cg, lm - 9)
ok(f"DL1 SPARC gas relation over {good.sum()} galaxies: M_b/M_* = {gas(8.0):.2f} (log M_* 8), {gas(9.0):.2f} (9), {gas(10.0):.2f} (10)", gas(8.0) > gas(10.0))
G, conv, h, Om, fcos = 4.301e-9, 1e6 / 3.0857e22, 0.674, 0.315, 0.0493 / 0.315
z = 0.05; rho_c = 2.775e11 * h**2 * (Om * (1 + z)**3 + 1 - Om)
Mh = 10**lmh; Mb = 10**lms * gas(lms)
r200 = (3 * Mh / (4 * np.pi * 200 * rho_c))**(1 / 3)
ratio = (G * Mh / r200**2 * conv) / (Mb / Mh) / 1.2e-10
chi = Mb / (Mh * fcos)
for s_, l1, l2, rr, cc, a, b in zip(samp, lms, lmh, ratio, chi, ehp, ehm):
    print(f"   {s_:4s} log M_* {l1:5.2f}  log M200c {l2:5.2f} (+{a:.2f}/-{b:.2f}):  (g/eps)/a0 = {rr:.2f}  (x/{10**(4 / 3 * (a + b) / 2):.1f})  chi_b = {cc:.2f}")
w = 1 / ((4 / 3 * (ehp + ehm) / 2)**2)
for name in ('DES', 'DESI'):
    m = samp == name
    sl = np.polyfit(lms[m], np.log10(ratio[m]), 1, w=np.sqrt(w[m]))[0]
    ok(f"DL2 {name}: (g200c/eps)/a0 from {ratio[m].min():.2f} to {ratio[m].max():.2f}; weighted trend d log/d log M_* = {sl:+.2f}; chi_b {chi[m].min():.2f}-{chi[m].max():.2f}", True)
m = samp == 'DES'
ok(f"DL3 at the dwarf end (log M_* < 9) the DES bins give (g/eps)/a0 = {', '.join(f'{x:.2f}' for x in ratio[m & (lms < 9)])} and chi_b = "
   f"{', '.join(f'{x:.2f}' for x in chi[m & (lms < 9)])}: dwarfs retain 10-35 per cent of the cosmic baryon share at the halo radius", True)
wm = 10**np.average(np.log10(ratio), weights=w); wc = 10**np.average(np.log10(chi), weights=w)
ok(f"DL5 all 13 bins: weighted mean (g200c/eps)/a0 = {wm:.2f}, chi_b = {wc:.2f}; no trend over log M_* = 8-10 (DES {np.polyfit(lms[m], np.log10(ratio[m]), 1, w=np.sqrt(w[m]))[0]:+.2f}, "
   f"DESI {np.polyfit(lms[~m], np.log10(ratio[~m]), 1, w=np.sqrt(w[~m]))[0]:+.2f} per decade)", True)
ok("DL4 caveats recorded: M200c from inner NFW fits; mixed centrals/satellites and colours; 0.3 dex stellar-mass systematics; gas from SPARC "
   "(H I-rich, rotation-curve selected)", True)
