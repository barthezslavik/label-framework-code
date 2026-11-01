# Test harness for the clock law a0 ~ 1/t against any set of a0(z) points (papers/restart/46-clock-combos.md, "Decisive test").
# Reads data/a0z_points.csv (or a path given as argv[1]); fits four laws with a free amplitude (optionally a factor per method), reports
# chi2, pulls, the constancy of a0 t/c, upper-limit violations and predictions at z = 2, 3. Rows of kind 'derived' are excluded unless
# argv[2] == 'with-derived'. W1-W3 check the harness on the current points; the verdict needs the rows still to be filled.
# Run: uv run --with sympy --with numpy --with scipy python inv_a0z_clock_test.py [csv] [with-derived]
import csv, os, sys
import numpy as np
from scipy.integrate import quad
from ga_core import ok

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'data', 'a0z_points.csv')
use_derived = len(sys.argv) > 2 and sys.argv[2] == 'with-derived'
rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith('#'))]
pts = [r for r in rows if r['kind'] == 'point' or (use_derived and r['kind'] == 'derived')]
lims = [r for r in rows if r['kind'] == 'upper']
z = np.array([float(r['z']) for r in pts]); a = np.array([float(r['a0']) for r in pts])
err = np.array([np.hypot(float(r['stat']), float(r['sys'])) for r in pts])

c = 2.998e8; H0 = 67.4e3 / 3.0857e22; Gyr = 3.156e16; Om, OL = 0.315, 0.685
E = lambda zz: np.sqrt(Om * (1 + zz)**3 + OL)
t_of_z = lambda zz: quad(lambda y: 1 / (y * np.sqrt(Om / y**3 + OL)), 1e-9, 1 / (1 + zz))[0] / H0
t0 = t_of_z(0)
laws = {'constant': lambda zz: np.ones_like(np.atleast_1d(zz), dtype=float), 'H(z)': lambda zz: E(np.atleast_1d(zz)),
        'clock t0/t': lambda zz: np.array([t0 / t_of_z(q) for q in np.atleast_1d(zz)]), '(1+z)': lambda zz: 1 + np.atleast_1d(zz)}

print(f"   points used: {len(pts)} ({'with' if use_derived else 'without'} derived), upper limits: {len(lims)}")
res = {}
for name, f in laws.items():
    fz = f(z)
    A = np.sum(a * fz / err**2) / np.sum(fz**2 / err**2)
    pulls = (a - A * fz) / err
    chi2 = float(np.sum(pulls**2)); dof = max(len(z) - 1, 1)
    viol = [(r['source'], float(r['a0']), A * f(float(r['z']))[0]) for r in lims if A * f(float(r['z']))[0] > float(r['a0'])]
    res[name] = (A, chi2, dof, pulls, viol)
    print(f"   {name:11s} A = {A:.2f}  chi2/dof = {chi2:.1f}/{dof}  max |pull| = {np.max(np.abs(pulls)):.1f}  "
          f"z=2: {A * f(2.0)[0]:.2f}, z=3: {A * f(3.0)[0]:.2f}  limits violated: {len(viol)}")
at = a * 1e-10 * np.array([t_of_z(q) for q in z]) / c
at_err = err * 1e-10 * np.array([t_of_z(q) for q in z]) / c
w = 1 / at_err**2; atm = np.sum(w * at) / np.sum(w)
chi_at = np.sum(((at - atm) / at_err)**2)
print(f"   a0 t/c per point: {np.round(at, 3)} +- {np.round(at_err, 3)}; weighted mean {atm:.3f}, chi2 about it {chi_at:.1f}/{len(at) - 1}")

ok(f"W1 harness reads {len(rows)} rows ({len(pts)} points, {len(lims)} limits) and fits {len(laws)} laws", len(pts) >= 3 and len(res) == 4)
ok(f"W2 on the current points the clock law has chi2 = {res['clock t0/t'][1]:.1f}, H(z) {res['H(z)'][1]:.1f}, (1+z) {res['(1+z)'][1]:.1f}, "
   f"constant {res['constant'][1]:.1f} (local SPARC vs MIGHTEE disagree; the verdict needs the high-z rows)", all(np.isfinite(v[1]) for v in res.values()))
ok(f"W3 the z = 2 limit (Milgrom 2017, < 4.8): clock predicts {res['clock t0/t'][0] * laws['clock t0/t'](2.0)[0]:.2f}, H(z) "
   f"{res['H(z)'][0] * laws['H(z)'](2.0)[0]:.2f}: violated by clock {len(res['clock t0/t'][4]) > 0}, by H(z) {len(res['H(z)'][4]) > 0}", True if lims else False)
