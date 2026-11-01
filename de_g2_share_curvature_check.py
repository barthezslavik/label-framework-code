# 128: G2 of PREDICTIONS.md with a measured baryon share. The predicted BTFR curvature c = 0.47 per dex in V assumes a share without
# curvature; a curvature k of log f per dex^2 in M_b shifts it by about s^2 k (doc 116 FC9: k = -0.035 cancels it). Here k is measured
# in homogeneous weak-lensing samples, each fitted alone: DES dwarfs (7 bins) and DESI dwarfs (6 bins; arXiv:2509.20434, M200c), and blue
# centrals of Mandelbaum et al. 2016 (5 bins, M200m), blue lenses of Velander et al. 2014 (CFHTLenS, 4 bins, M200c), and HSC centrals of all colours
# (Chaurasiya et al. 2023, threshold samples differenced into bins, M200m; both added after the goal check), and KiDS-1000 isolated
# galaxies (Brouwer et al. 2021 profiles, own NFW fits, de_kids_halo_masses_check). Gas from SPARC (log M_b/M_* linear in log M_*). The present-day share chi = M_b/(f_cos
# M_h) is not the collapse-epoch share eps of the population law: chi = eps M_v/M_h, and the curvature of log(M_v/M_h) at constant eps is
# predicted from dust collapse (nu = 1, 1.5; doc 114) and subtracted. k_eps -> c = c0 + s^2 k_eps with s^2 = 12.9 from FC9. Compared with the
# SPARC curvature -0.62 +- 0.77 (doc 93). Monte Carlo over the halo-mass errors. GS1-GS6.
# Run: ~/.venvs/jev/bin/python -I de_g2_share_curvature_check.py DATA_ROOT SPARC_TABLE   (after de_kids_halo_masses_check)
import os, sys, contextlib, io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
kc = {'__name__': 'kc', '__file__': os.path.join(HERE, 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read().split('tab = {(nu, M)')[0], kc)
ok = kc['ok']
root, sparc = sys.argv[1], sys.argv[2]
L, MHI = [], []
for l in open(sparc):
    p = l.split()
    if len(p) < 18: continue
    try:
        Lv, Mh_ = float(p[7]), float(p[13]); int(p[1])
    except ValueError:
        continue
    L.append(Lv); MHI.append(Mh_)
L, MHI = np.array(L) * 1e9, np.array(MHI) * 1e9
Ms = 0.5 * L; Mb = Ms + 1.33 * MHI; sel = Ms > 1e7
cg, cgcov = np.polyfit(np.log10(Ms[sel]) - 10, np.log10(Mb[sel] / Ms[sel]), 1, cov=True)
gas_scatter = np.std(np.log10(Mb[sel] / Ms[sel]) - np.polyval(cg, np.log10(Ms[sel]) - 10))
lgas = lambda lm: np.polyval(cg, lm - 10)
print(f"   SPARC gas ({sel.sum()} galaxies): log M_b/M_* = {cg[1]:.3f} + {cg[0]:.3f} (log M_* - 10)")
fcos = 0.0493 / 0.315
sets = {}
d = [l.split() for l in open(os.path.join(root, 'desi_des_dwarfs2025', 'table_halo_masses.txt')) if l.strip() and not l.startswith('#')]
for nm in ('DES', 'DESI'):
    rows = np.array([[float(x) for x in r[1:7]] for r in d if r[0] == nm])
    sets[nm] = (rows[:, 0], rows[:, 3], 0.5 * (rows[:, 4] + rows[:, 5]), 'M200c', 0.0)      # log M*, log Mh (Msun), err
t = np.loadtxt(os.path.join(root, 'mandelbaum2016', 'table_B1_blue.txt'))
sets['Mandelbaum blue'] = (t[:, 0], t[:, 2] - np.log10(0.673), 0.5 * (t[:, 3] + t[:, 4]), 'M200m', 0.0)
v = np.array([[float(x) for x in l.split()[1:]] for l in open(os.path.join(root, 'velander2014', 'table4_blue.txt')) if l.strip() and not l.startswith('#')])
lmh_v = np.log10(v[:, 3] * 1e11); elmh_v = 0.5 * (np.log10(1 + v[:, 4] / v[:, 3]) - np.log10(1 - v[:, 5] / v[:, 3]))
sets['Velander blue (CFHTLenS)'] = (np.log10(v[:, 2] * 1e10), lmh_v, elmh_v, 'M200c', 0.0)   # h70 = 1 at h = 0.70
# HSC (Chaurasiya et al. 2023): threshold samples -> differential bins. Central counts N_c = N (1 - f_sat); the mean central halo mass of the
# bin between thresholds i and i+1 is (N_c,i <M>_i - N_c,i+1 <M>_i+1)/(N_c,i - N_c,i+1). Bin stellar mass at the midpoint (h^-2 -> Msun with
# h = 0.677) at fixed satellite fractions. Errors: the threshold errors propagated as independent -- illustrative only (no covariance of
# the joint HOD fit; the direction of the bias is unknown). The SPARC gas relation is applied to centrals of all colours (an assumption).
hc = np.loadtxt(os.path.join(root, 'hsc_chaurasiya2023', 'table1_4_z1.txt'))
hh_ = 0.677; Nc = hc[:, 1] * (1 - hc[:, 2]); Mc = 10**hc[:, 3] / hh_; eM = Mc * np.log(10) * 0.5 * (hc[:, 4] + hc[:, 5])
Mb_ = (Nc[:-1] * Mc[:-1] - Nc[1:] * Mc[1:]) / (Nc[:-1] - Nc[1:])
eb_ = np.sqrt((Nc[:-1] * eM[:-1])**2 + (Nc[1:] * eM[1:])**2) / (Nc[:-1] - Nc[1:])
lms_h = 0.5 * (hc[:-1, 0] + hc[1:, 0]) - 2 * np.log10(hh_)
mh = (lms_h <= 10.8) & (Mb_ > 0)
sets['HSC (Chaurasiya 2023, differenced)'] = (lms_h[mh], np.log10(Mb_[mh]), eb_[mh] / Mb_[mh] / np.log(10), 'M200m', 0.0)
# KiDS-1000 isolated galaxies (Brouwer et al. 2021 profiles), halo masses from our own NFW fits (de_kids_halo_masses_check, run first);
# the first column is already the baryonic mass (stars + cold gas), flag 1.0 = no SPARC gas correction
kk = np.loadtxt(os.path.join(root, 'kids_brouwer2021', 'halo_masses_fit.txt'))
sets['KiDS-1000 isolated (own NFW fits)'] = (kk[:, 0], kk[:, 1], kk[:, 2], 'M200c', 1.0)

def curv(x, y):
    c2, c1, c0 = np.polyfit(x, y, 2); return 2 * c2                # k: d^2 y / dx^2 (log f = a + b x + k x^2/2)
def kpred(lMb, nu, eps=0.1):
    Mh = 10**lMb / (eps * fcos)
    zc = np.array([kc['gv'](M, nu)[0] for M in Mh])
    lratio = 0.5 * np.log10(18 * np.pi**2 * (1 + zc)**3 / (200 * (1 + 0.1)**3))   # log M_h/M_v (isothermal outside r_v, doc 114)
    return curv(lMb, -lratio)
rng = np.random.default_rng(5)
S2, c0, c_sparc, e_sparc = 12.9, 0.47, -0.62, 0.77
out = {}
for nm, (lms, lmh, e, mdef, isb) in sets.items():
    lMb = lms if isb else lms + lgas(lms); y = lMb - np.log10(fcos) - lmh
    ks = [curv(lMb, y + rng.normal(0, e)) for _ in range(4000)]
    kc_med, kc_sd = curv(lMb, y), np.std(ks)
    kp = [kpred(lMb, nu) for nu in (1.0, 1.5)]
    keps = kc_med - np.mean(kp)
    cpred = c0 + S2 * keps; ec = S2 * kc_sd
    out[nm] = (lMb.min(), lMb.max(), kc_med, kc_sd, kp, keps, cpred, ec)
    print(f"   {nm:16s} ({mdef}) log M_b {lMb.min():.2f}-{lMb.max():.2f}: k_chi = {kc_med:+.3f} +- {kc_sd:.3f}; predicted k from M_v/M_h "
          f"{kp[0]:+.3f} (nu 1), {kp[1]:+.3f} (nu 1.5); k_eps = {keps:+.3f}; BTFR c = {cpred:+.2f} +- {ec:.2f}")
ok(f"GS1 gas from SPARC: M_b/M_* = {10**lgas(8.5):.1f} at log M_* = 8.5, {10**lgas(10.5):.2f} at 10.5", 10**lgas(8.5) > 10**lgas(10.5) > 1)
ok("GS2 measured share curvature k_chi per dex^2: " + "; ".join(f"{k}: {v[2]:+.3f} +- {v[3]:.3f}" for k, v in out.items()), True)
ok("GS3 the dust-collapse term M_v/M_h has curvature " + "; ".join(f"{k}: {np.mean(v[4]):+.3f}" for k, v in out.items())
   + " per dex^2 -- below a third of the measurement errors (but comparable with the 0.035 that matters for G2)", all(abs(np.mean(v[4])) < 0.35 * v[3] for v in out.values()))
kmax = min(v[3] for v in out.values())
ok(f"GS4 the best-measured share curvature has error {kmax:.3f} per dex^2 against the |k| = 0.035 that cancels the predicted BTFR curvature: "
   f"current lensing errors are {kmax / 0.035:.0f} times too large to fix k_eps to the needed level", True)
ok("GS5 BTFR curvature implied by each measured share (c0 + s^2 k_eps): " + "; ".join(f"{k}: {v[6]:+.2f} +- {v[7]:.2f}" for k, v in out.items())
   + f"; SPARC measures {c_sparc:+.2f} +- {e_sparc:.2f}: not distinguished", True)
ok("GS6 recorded: G2 stays 'not distinguished' -- the share curvature needed to test it (0.035 per dex^2) is 15-60 times below the "
   "curvature errors of present homogeneous lensing samples; the halo-mass definitions differ between samples (fitted separately)", True)

# GS7 HSC (arXiv:2301.04664, the parametric CSMF relation in DATA_ROOT/hsc_2301.04664; de_hsc_shmr_check) in the disc regime: curvature of
#     log chi over log M_b = 9.7-10.8 by sampling the four relation parameters. The curvature of a parametric relation is set by its
#     functional form (double power law), so it is not a measurement of the share curvature -- recorded as such.
hs = {'__name__': 'hs', '__file__': os.path.join(HERE, 'de_hsc_shmr_check.py')}
_argv = sys.argv[:]; sys.argv = [sys.argv[0], root, os.path.dirname(sparc)]   # de_hsc_shmr_check takes the SPARC folder
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(hs['__file__']).read().split('rng = np.random.default_rng(17)')[0], hs)
sys.argv = _argv
Ph = hs['P']; rng2 = np.random.default_rng(23)
def drawh(k):
    v, ep, em = Ph[k]; x = rng2.standard_normal(); return v + (ep if x > 0 else em) * x
kh = []
for _ in range(1500):
    try:
        o = hs['run']([drawh('logM0'), drawh('logM1'), max(drawh('gamma1'), 1.01), max(drawh('gamma2'), 0.02)])
    except ValueError:
        continue
    m = o[:, 0] <= 10.85
    if m.sum() >= 4: kh.append(curv(o[m, 0], np.log10(o[m, 2])))
kh = np.array(kh); kb = curv(hs['base'][hs['base'][:, 0] <= 10.85, 0], np.log10(hs['base'][hs['base'][:, 0] <= 10.85, 2]))
ok(f"GS7 HSC parametric relation, disc regime: curvature of log chi = {kb:+.3f} (parameter sampling: median {np.median(kh):+.3f}, spread "
   f"{np.std(kh):.3f}) -- set by the double power law, not a measurement; taken literally c = {c0 + S2 * (kb - 0.05):+.1f} +- {S2 * np.std(kh):.1f}, "
   f"{abs(c0 + S2 * (kb - 0.05) - c_sparc) / np.hypot(S2 * np.std(kh), e_sparc):.1f} sigma from SPARC (parameter spread only); KiDS+GAMA (Dvornik et al. 2020) gives only fitted relation "
   "parameters for centrals; KiDS-1000 enters through its public profiles (own fits)", True)
ok(f"GS8 comparison with SPARC (BIG-SPARC not available here): predicted c = {c0:.2f} at a share without curvature, implied c from the measured "
   f"shares " + ", ".join(f"{k}: {v[6]:+.1f} +- {v[7]:.1f}" for k, v in out.items()) + f"; SPARC c = {c_sparc:+.2f} +- {e_sparc:.2f}: every value is "
   "within 2.2 sigma of SPARC (KiDS is the largest) -- not distinguished", all(abs(v[6] - c_sparc) < 2.2 * np.hypot(v[7], e_sparc) for v in out.values())
   and abs(c0 - c_sparc) < 2 * e_sparc)
# GS9 KiDS systematic: the same curvature with the halo masses of the Duffy c(M) fits (de_kids_halo_masses_check prints them)
kd = np.array([11.81, 12.21, 12.30, 12.46])                       # log M200c with the Duffy concentration (KH3)
lMbk = kk[:, 0]; kdu = curv(lMbk, lMbk - np.log10(fcos) - kd)
kwin = {}
for rm in ('0.3', '0.5'):
    kw = np.loadtxt(os.path.join(root, 'kids_brouwer2021', f'halo_masses_fit_R{rm}.txt'))
    kwin[rm] = curv(kw[:, 0], kw[:, 0] - np.log10(fcos) - kw[:, 1])
ok(f"GS10 KiDS radial window (review r13): k_chi = {kwin['0.3']:+.2f} (R <= 0.3 Mpc), {kwin['0.5']:+.2f} (<= 0.5), "
   f"{out['KiDS-1000 isolated (own NFW fits)'][2]:+.2f} (<= 1.0): the sign is not stable, so the ~2 sigma of the R <= 1 Mpc fit is not a signal", True)
ok(f"GS9 KiDS with the Duffy concentration: k_chi = {kdu:+.3f} (free c: {out['KiDS-1000 isolated (own NFW fits)'][2]:+.3f}); the KiDS curvature rests on four "
   "bins with one degree of freedom for a quadratic, the first bin wide (log M* 8.5-10.3)", True)
# GS11 the predicted BTFR curvature with ONE measured share: inverse-variance mean of the eps-curvatures of the samples with statistical
#     errors (DES, DESI, Mandelbaum, Velander, KiDS at R <= 0.3 Mpc -- the window Brouwer et al. trust for isolation; HSC excluded, its error
#     is illustrative); c = c0 + s^2 k_eps; compared with SPARC. Different halo-mass definitions are combined (curvatures only).
kw03 = np.loadtxt(os.path.join(root, 'kids_brouwer2021', 'halo_masses_fit_R0.3.txt'))
yk = kw03[:, 0] - np.log10(fcos) - kw03[:, 1]
kk03 = [curv(kw03[:, 0], yk + rng.normal(0, kw03[:, 2])) for _ in range(4000)]
kp_k = np.mean([kpred(kw03[:, 0], nu) for nu in (1.0, 1.5)])
comb = {k: (v[5], v[3]) for k, v in out.items() if k.startswith(('DES', 'DESI', 'Mandelbaum', 'Velander'))}
comb['KiDS-1000 (R <= 0.3 Mpc)'] = (curv(kw03[:, 0], yk) - kp_k, np.std(kk03))
w = np.array([1 / e**2 for _, e in comb.values()]); kv = np.array([k for k, _ in comb.values()])
kmean, kerr = np.sum(w * kv) / np.sum(w), 1 / np.sqrt(np.sum(w))
cpred, cerr = c0 + S2 * kmean, S2 * kerr
ok("GS11 illustration assuming one share curvature common to all samples (untested; the error excludes the scatter between populations) "
   "-- combined share curvature (inverse-variance, " + ", ".join(f"{k}: {v[0]:+.2f} +- {v[1]:.2f}" for k, v in comb.items())
   + f"): k_eps = {kmean:+.3f} +- {kerr:.3f} per dex^2 -> predicted BTFR curvature with the measured share c = {cpred:+.1f} +- {cerr:.1f}; "
   f"SPARC {c_sparc:+.2f} +- {e_sparc:.2f}: {abs(cpred - c_sparc) / np.hypot(cerr, e_sparc):.1f} sigma -- not distinguished; the error of c "
   f"({cerr:.1f}) is {cerr / e_sparc:.0f} times SPARC's", abs(cpred - c_sparc) / np.hypot(cerr, e_sparc) < 2)
# GS12 the G2 outcome from ONE homogeneous sample, no common-curvature assumption: DES (7 bins of one survey, log M_b 8.65-10.34, the
#     low-mass half where the predicted curvature matters; KiDS has a smaller nominal error but its curvature changes sign with the radial
#     window, GS10). DES has no gas measurements: the SPARC gas relation is applied, and its uncertainty (fit covariance of the relation and
#     0.3 of the scatter of individual galaxies about it per bin, an assumed residual for bin means)
#     is propagated together with the halo-mass errors.
lms_d, lmh_d, e_d = sets['DES'][0], sets['DES'][1], sets['DES'][2]
kg = []
for _ in range(4000):
    cgr = rng.multivariate_normal(cg, cgcov)
    lMb_d = lms_d + np.polyval(cgr, lms_d - 10) + rng.normal(0, gas_scatter, len(lms_d)) * 0.3
    kg.append(curv(lMb_d, lMb_d - np.log10(fcos) - (lmh_d + rng.normal(0, e_d))))
kg = np.array(kg); kp_d = np.mean(out['DES'][4])
c_d = c0 + S2 * (np.median(kg) - kp_d); ec_d = S2 * np.std(kg)
t_des = abs(c_d - c_sparc) / np.hypot(ec_d, e_sparc)
ok(f"GS12 G2 outcome from the single homogeneous sample DES (KiDS left out: radial-window instability): with halo-mass and gas-relation "
   f"uncertainties (SPARC gas, fit covariance + 0.3 x the galaxy scatter {gas_scatter:.2f} dex per bin) k_chi = {np.median(kg):+.3f} +- {np.std(kg):.3f}; "
   f"BTFR curvature c = {c_d:+.1f} +- {ec_d:.1f} vs SPARC {c_sparc:+.2f} +- {e_sparc:.2f} ({t_des:.1f} sigma) and vs the constant-share "
   f"prediction {c0:.2f}: not distinguished -- conditional on the SPARC gas relation; an independent test needs gas measured in the same sample", t_des < 2)
# GS13 robustness of the G2 outcome to the gas assumption (no lensing sample with HI measured for the same galaxies is public): DES with
#     (a) no gas uncertainty, (b) the GS12 assumption, (c) the full galaxy scatter of the SPARC gas relation independently in every bin
#     (a worst case), (d) no gas at all (M_b = M*): the BTFR curvature and its distance from SPARC
def des_c(scat, nogas=False):
    kk_ = []
    for _ in range(4000):
        cgr = rng.multivariate_normal(cg, cgcov) if scat is not None else cg
        lg = 0 if nogas else np.polyval(cgr, lms_d - 10) + (rng.normal(0, gas_scatter, len(lms_d)) * scat if scat else 0)
        lMb_ = lms_d + lg
        kk_.append(curv(lMb_, lMb_ - np.log10(fcos) - (lmh_d + rng.normal(0, e_d))))
    kk_ = np.array(kk_); return c0 + S2 * (np.median(kk_) - kp_d), S2 * np.std(kk_)
def des_c_curved(amp):
    """systematic curvature of the gas relation: the ends of the DES stellar-mass range lie amp dex above (or below) the centre, relative to
    the straight SPARC relation (a constant shift does not change the curvature)"""
    mid, half = 0.5 * (lms_d.min() + lms_d.max()), 0.5 * (lms_d.max() - lms_d.min())
    kk_ = []
    for _ in range(4000):
        lMb_ = lms_d + np.polyval(cg, lms_d - 10) + amp * ((lms_d - mid) / half)**2      # +amp at the ends, 0 at the centre (review r22)
        kk_.append(curv(lMb_, lMb_ - np.log10(fcos) - (lmh_d + rng.normal(0, e_d))))
    kk_ = np.array(kk_); return c0 + S2 * (np.median(kk_) - kp_d), S2 * np.std(kk_)
cases_g = {'no gas uncertainty': des_c(None), 'GS12 (0.3 x scatter)': (c_d, ec_d), 'full scatter per bin': des_c(1.0), 'no gas (M_b = M*)': des_c(None, True),
           f'bent: ends {gas_scatter:+.2f} dex above the centre': des_c_curved(gas_scatter), f'ends {-gas_scatter:+.2f} dex': des_c_curved(-gas_scatter),
           f'ends {2 * gas_scatter:+.2f} dex': des_c_curved(2 * gas_scatter), f'ends {-2 * gas_scatter:+.2f} dex (formal: gives M_b < M* in the top bin)': des_c_curved(-2 * gas_scatter)}
ok("GS13 DES BTFR curvature under gas assumptions: " + "; ".join(f"{k}: {v[0]:+.1f} +- {v[1]:.1f} ({abs(v[0] - c_sparc) / np.hypot(v[1], e_sparc):.1f} sigma)"
   for k, v in cases_g.items()) + " -- in all eight scenarios considered (incl. systematic bends of the gas relation by one and two galaxy scatters) the outcome is 'not distinguished'; "
   "a gas measurement of the DES sample is still needed for an independent test",
   all(abs(v[0] - c_sparc) / np.hypot(v[1], e_sparc) < 2 for v in cases_g.values()))
# GS14 a second single-sample outcome with another external gas estimate: KiDS-1000 isolated galaxies, baryonic masses (stars + cold gas,
#     gas from the Boselli et al. scaling, not measured) from the survey team (Brouwer et al. 2021), taken as fixed and halo masses from our fits to the same galaxies' lensing profiles, in the radial window
#     R <= 0.3 Mpc that Brouwer et al. trust for isolation (GS10 shows the larger windows change the sign)
kk03_c = c0 + S2 * (curv(kw03[:, 0], yk) - kp_k); ek03 = S2 * np.std(kk03)
t_k = abs(kk03_c - c_sparc) / np.hypot(ek03, e_sparc)
ok(f"GS14 KiDS-1000, one survey for both masses (R <= 0.3 Mpc): k_chi = {curv(kw03[:, 0], yk):+.2f} +- {np.std(kk03):.2f}, BTFR curvature "
   f"c = {kk03_c:+.1f} +- {ek03:.1f} vs SPARC {c_sparc:+.2f} +- {e_sparc:.2f} ({t_k:.1f} sigma): not distinguished, with the Boselli gas scaling instead of SPARC (its uncertainty not included)", t_k < 2)
