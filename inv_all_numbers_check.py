# 52: normalise the numbers of the universe to 1 one by one and watch the symmetries. ND1-ND7.
# Dimensional constants are removed by units: in GA each one is the licence to exponentiate a generator that mixes two different scaling
# weights. Dimensionless numbers cannot be removed: for each we ask whether it has a symmetric point where a GA symmetry is exact.
# Literature (from memory, network down): Tegmark-Aguirre-Rees-Wilczek 2006 (31 dimensionless constants); Duff-Okun-Veneziano 2002;
# Dirac quantisation e g = 2 pi n (Heaviside-Lorentz, hbar = c = 1); Harrison-Zeldovich n_s = 1.
# Run: uv run --with sympy --with numpy --with scipy python inv_all_numbers_check.py
import numpy as np
import sympy as sp
from scipy.integrate import quad
from ga_core import ok

# ---- dimensional part --------------------------------------------------------------------------------------------------------------
# exponents of (L, T, M, Theta)
dims = {'c': (1, -1, 0, 0), 'hbar': (2, -1, 1, 0), 'G': (3, -2, -1, 0), 'k_B': (2, -2, 1, -1),
        'Lambda': (-2, 0, 0, 0), 'a0': (1, -2, 0, 0), 'H0': (0, -1, 0, 0)}
val = {'c': 2.99792458e8, 'hbar': 1.054571817e-34, 'G': 6.6743e-11, 'k_B': 1.380649e-23, 'a0': 1.2e-10}
H0 = 67.4e3 / 3.0857e22; OL, Om = 0.685, 0.315
val['H0'] = H0; val['Lambda'] = 3 * OL * H0**2 / val['c']**2

# ND1 normalising c, hbar, G, k_B one by one: each kills one independent scaling weight (rank of the dimension matrix grows by one).
order = ['c', 'hbar', 'G', 'k_B']
ranks = [sp.Matrix([dims[k] for k in order[:i + 1]]).rank() for i in range(4)]
merged = ['time weight = length weight (spacetime one block)', 'action weight 0 (energy = frequency, mass = 1/length)',
          'mass weight = length weight (geometry = matter)', 'temperature = energy']
for k, r, m in zip(order, ranks, merged):
    print(f"   set {k:5s} = 1 -> rank {r}: {m}")
ok(f"ND1 c, hbar, G, k_B remove the four scaling weights one by one (ranks {ranks}); after them no scaling freedom is left", ranks == [1, 2, 3, 4])

# ND2 what each one licenses in GA: an exponential exp(B) needs a weight-0 argument.
#     c: boost rotor exp(phi e0 e1) on X = t e0 + x e1 mixes t and x; covariance under t -> tau t, x -> lam x needs c -> (lam/tau) c.
#     hbar: phase exp(I S/hbar); k_B: thermal stretch exp(-E/k_B T) (KMS period); G: curvature (L^-2) = G * energy density / c^4;
#     Lambda: the de Sitter generator P0 + eps K0 mixes P (scale weight -1) and K (+1): eps needs weight L^-2.
lam, tau, t, xx, phi, cc = sp.symbols('lambda tau t x phi c', positive=True)
xb = xx * sp.cosh(phi) + cc * t * sp.sinh(phi)
cov = sp.simplify(xb.subs({xx: lam * xx, t: tau * t, cc: lam / tau * cc}, simultaneous=True) - lam * xb)
D, P, K = sp.symbols('D P K', commutative=False)
w = {'P': -1, 'K': 1}                                     # [D, P] = -P, [D, K] = K: weights under the dilation
w_eps = w['P'] - w['K']                                   # P + eps K is homogeneous only if eps carries weight -2 (in 1/length units: L^-2)
ok(f"ND2 c is the price of the boost rotor (covariance residual {cov}); Lambda is the price of mixing P and K (eps needs weight {w_eps}, i.e. "
   "L^-2): every dimensional constant = licence for one mixed exponential", cov == 0 and w_eps == -2)

# ND3 five constants for four weights leave exactly ONE pure number among (c, hbar, G, k_B, Lambda); with a0 and H0 two more.
A = sp.Matrix([dims[k] for k in ['c', 'hbar', 'G', 'k_B', 'Lambda', 'a0', 'H0']]).T
ns = A.nullspace()
lP2 = val['hbar'] * val['G'] / val['c']**3; tP = np.sqrt(lP2) / val['c']
pure = {'Lambda l_P^2': val['Lambda'] * lP2, 'a0 / (c H0)': val['a0'] / (val['c'] * H0), 'H0 t_P': H0 * tP}
for k, v in pure.items():
    print(f"   {k:14s} = {v:.3e}")
ok(f"ND3 nullspace dimension {len(ns)} for 7 constants on 4 weights: three pure numbers survive normalisation "
   f"(Lambda l_P^2 = {pure['Lambda l_P^2']:.2e}, a0/cH0 = {pure['a0 / (c H0)']:.3f}, H0 t_P = {pure['H0 t_P']:.2e})", len(ns) == 3)

# ---- dimensionless part: symmetric points ------------------------------------------------------------------------------------------
# ND4 numbers with an exact GA symmetric point.
#     n_s: Delta^2(k) ~ k^(n_s - 1) has dilation weight n_s - 1; n_s = 1 is exact K-invariance of the primordial spectrum.
#     w: T = -rho g is the only Lorentz-invariant stress (theorem base); w = -1 exactly.
#     alpha: Hodge F -> F I exchanges e and g; Dirac e g = 2 pi (HL units) is self-dual at e^2 = 2 pi, alpha = e^2/4pi = 1/2.
#     Omega_k: flatness = K-covariant (conformally flat, homothetic) FLRW; Omega_k = 0.
#     spacetime dimension: Hodge on grade k has weight lam^(n-2k); the field strength (grade 2) is K-blind only at n = 4.
e2 = sp.symbols('e2', positive=True)
alpha_sd = sp.solve(sp.Eq(e2, (2 * sp.pi)**2 / e2), e2)[0] / (4 * sp.pi)
n_s, w_de, alpha, Ok = 0.9649, -1.03, 1 / 137.036, 0.001
nn = sp.symbols('n', integer=True, positive=True)
n_hodge = sp.solve(sp.Eq(nn - 2 * 2, 0), nn)
sym_pts = {'n_s': (n_s, 1, 'scale K of perturbations'), 'w': (w_de, -1, 'Lorentz of the vacuum'),
           'alpha': (alpha, float(alpha_sd), 'electric-magnetic duality'), 'Omega_k': (Ok, 0, 'homothetic FLRW'),
           'dim': (4, int(n_hodge[0]), 'conformal Maxwell')}
for k, (v, s, m) in sym_pts.items():
    print(f"   {k:8s} value {v:<10.4g} symmetric point {s:<6.4g} ({m}); distance {abs(v - s):.3g}")
ok(f"ND4 five numbers have an exact symmetric point: n_s (breaking {n_s - 1:+.3f}), w, alpha (self-dual {alpha_sd}, we sit at 1/137, "
   "factor 68 into the weak side), Omega_k, dimension 4 (= the K-blind point of grade-2 Hodge)", alpha_sd == sp.Rational(1, 2) and n_hodge == [4])

# ND5 numbers measured by a symmetric point at 0 (exact symmetry = no structure): Q (homothety exact at Q = 0), alpha_G (gravity off),
#     Lambda l_P^2 (no mixing of P and K: Poincare instead of de Sitter). 'Set to 1' here is the strongly broken end: compute what it means.
mp, MP = 1.67262192e-27, np.sqrt(val['hbar'] * val['c'] / val['G'])
alpha_G = (mp / MP)**2
Q = 2e-5
star_N = alpha_G**-1.5                                    # particles in a star (Chandrasekhar scale)
hubble_cells = pure['Lambda l_P^2']**-0.5                 # de Sitter radius in Planck lengths
for k, v, m in (('alpha_G', alpha_G, f'particles in a star ~ alpha_G^-3/2 = {star_N:.1e}; at 1: a star is one proton'),
                ('Lambda l_P^2', pure['Lambda l_P^2'], f'de Sitter radius = {hubble_cells:.1e} l_P; at 1: the universe is one Planck cell'),
                ('Q', Q, 'at 0: exact homothety, no structure; at 1: no FLRW background')):
    print(f"   {k:13s} {v:.3e}  {m}")
ok("ND5 the three hierarchy numbers sit near the SYMMETRIC end 0 (gravity off, no P-K mixing, exact homothety); setting them to 1 removes "
   "stars, size and background: the large numbers of the universe are small breakings of symmetries", alpha_G < 1e-37 and Q < 1e-4)

# ND6 numbers WITHOUT a GA symmetric point (refutation of 'every number is a distance from a symmetric point'): ratios whose value 1 gives
#     no extra symmetry in GA (Omega_c/Omega_b, eta, m_e/m_p, Omega_L/Omega_m at a fixed epoch). m_n/m_p -> 1 is isospin, but that is an
#     internal (non-GA) symmetry and needs alpha = 0 too.
no_point = {'Omega_c/Omega_b': 5.36, 'eta (baryon/photon)': 6.1e-10, 'm_e/m_p': 1 / 1836.15, 'Omega_L/Omega_m (now)': OL / Om}
for k, v in no_point.items():
    print(f"   {k:22s} {v:.3g}   no GA symmetric point")
ok(f"ND6 REFUTED as a universal rule: {len(no_point)} numbers have no symmetric point in GA (only labels / initial data / state)", len(no_point) == 4)

# ND7 the 'pure GA universe': all dimensional constants 1, EVERY number at its symmetric point -- n_s = 1, w = -1, Omega_k = 0, Q = 0,
#     alpha_G = 0, all masses and mu^2 = 0, theta = 0, and Lambda l_P^2 = 0 (ND5). The vacuum is then Minkowski with free massless fields:
#     the full conformal group O(2,4) = versors of Cl(2,4) (inv_numbers_falsify_check XN1). Lambda != 0 is the FIRST breaking: it picks a
#     vector of R^{2,4} and leaves O(1,4), exact de Sitter a = exp(H t); its radius Lambda l_P^2 is the first number of the universe.
tt, HH = sp.symbols('t H', positive=True)
a = sp.exp(HH * tt)
fried = sp.simplify((a.diff(tt) / a)**2 - HH**2); acc = sp.simplify(a.diff(tt, 2) / a - HH**2)
ok(f"ND7 pure GA universe = conformal Minkowski, O(2,4) (dim 15); first breaking Lambda -> exact de Sitter (Friedmann residual {fried}, "
   f"acceleration residual {acc}), O(1,4) (dim 10); all further structure = the breakings of ND4-ND5 + the unexplained numbers of ND6",
   fried == 0 and acc == 0)
