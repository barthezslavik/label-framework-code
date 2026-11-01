# 92: compactness -> integers. Which numbers of the framework can become integers? Two sources: the charge of a compact generator (a
# bivector with B^2 < 0, whose rotor exp(B t) is periodic) together with single-valuedness at level 3, and the winding number of a map
# (CI6, pi_2(S^2) = Z), which needs no generator. The statement is about charges, not energy spectra.
# CI1-CI8. Run: ~/.venvs/jev/bin/python ga_compact_integers_check.py
import itertools
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# so(2,4) = bivectors of Cl(2,4): basis e0 (+), e1, e2, e3 (-), e5 (+), e6 (-); M_ab acts on vectors as v -> (v.e_b) e_a - (v.e_a) e_b.
eta = np.diag([1, -1, -1, -1, 1, -1]).astype(float)
def M(a, b):
    m = np.zeros((6, 6)); m[a, :] += eta[b]; m[b, :] -= eta[a]
    return m.T.copy()
def kind(X):
    if np.abs(np.linalg.matrix_power(X, 6)).max() < 1e-12: return 'null'           # nilpotent (eigenvalues of Jordan blocks are inexact)
    ev = np.linalg.eigvals(X); re, im = np.abs(ev.real).max(), np.abs(ev.imag).max()
    return 'compact' if re < 1e-12 else ('noncompact' if im < 1e-12 else 'mixed')

# CI1 the 15 basis bivectors e_a e_b: compact iff e_a^2 and e_b^2 have the same sign ((e_a e_b)^2 = -e_a^2 e_b^2 < 0). Count 7 = dim so(2)+so(4)
kinds = {(a, b): kind(M(a, b)) for a, b in itertools.combinations(range(6), 2)}
nc = sum(k == 'compact' for k in kinds.values()); nn = sum(k == 'noncompact' for k in kinds.values())
ok(f"CI1 basis bivectors of Cl(2,4): {nc} compact, {nn} non-compact (maximal compact so(2)+so(4), dim 1 + 6 = 7)", nc == 7 and nn == 8)

# CI2 the physical generators (index 4 = e5, 5 = e6): rotations J = M_ij compact; boosts M_0i non-compact; dilation D = M_56 non-compact;
#     P = M_m5 + M_m6 and K = M_m5 - M_m6 nilpotent (null); the conformal Hamiltonian (P0 + K0)/2 = M_05 compact.
J = M(1, 2); B = M(0, 1); D = M(4, 5); P0 = M(0, 4) + M(0, 5); K0 = M(0, 4) - M(0, 5)
res = {'J': kind(J), 'boost': kind(B), 'D': kind(D), 'P0': kind(P0), 'K0': kind(K0), '(P0+K0)/2': kind((P0 + K0) / 2)}
ok(f"CI2 generator types: {res}", res == {'J': 'compact', 'boost': 'noncompact', 'D': 'noncompact', 'P0': 'null', 'K0': 'null',
                                         '(P0+K0)/2': 'compact'})

# CI3 compact => periodic rotor; on spinors (rotor R = exp(-B t/2)) the period doubles: R(2 pi) = -1, so angular momentum takes values in
#     (hbar/2) Z (integer for vectors, half-integer allowed for spinors). Check with the Pauli rotor of e1 e2.
s1 = sp.Matrix([[0, 1], [1, 0]]); s2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
e12 = s1 * s2
R = lambda th: sp.simplify((-e12 * th / 2).exp())
ok(f"CI3 spinor rotor of e1e2 at 2 pi = {R(2 * sp.pi).tolist()}, at 4 pi = {R(4 * sp.pi).tolist()}: j in Z/2", R(2 * sp.pi) == -sp.eye(2) and R(4 * sp.pi) == sp.eye(2))

# CI4 the duality circle (base B1): on bivectors I^2 = -1, so exp(I theta) is compact (period 2 pi). With magnetic sources (Hodge with sources:
#     grade-3 currents, base) single-valuedness of exp(i e int A / hbar) around the Dirac string gives e g = n hbar / 2 (Gaussian units):
#     the string carries flux 4 pi g; phase e 4 pi g / (hbar c) in 2 pi Z.
e_, g_, hb, n_ = sp.symbols('e g hbar n', positive=True)
phase = e_ * 4 * sp.pi * g_ / hb
ok(f"CI4 Dirac: phase {phase} = 2 pi n  =>  e g = {sp.solve(sp.Eq(phase, 2 * sp.pi * n_), g_)[0] * e_}: the PRODUCT of charges is quantised, not e itself",
   sp.simplify(sp.solve(sp.Eq(phase, 2 * sp.pi * n_), g_)[0] * e_ - n_ * hb / 2) == 0)

# CI5 Madelung (ladder level 3): psi = sqrt(rho) exp(i S/hbar), v = grad S / m. The Kelvin circulation -- the Noether charge of relabelling
#     (SDiff) -- is quantised: for psi ~ r^|n| exp(i n phi), oint v.dl = 2 pi hbar n / m. Label fluids become quantised-vortex fluids.
r, ph, m = sp.symbols('r phi m', positive=True); nint = sp.symbols('n', integer=True)
S = hb * nint * ph
vphi = sp.diff(S, ph) / (m * r)
circ = sp.integrate(vphi * r, (ph, 0, 2 * sp.pi))
ok(f"CI5 circulation of a level-3 label fluid = {circ}: the relabelling charge becomes an integer in units 2 pi hbar/m", sp.simplify(circ - 2 * sp.pi * hb * nint / m) == 0)

# CI6 the angular label blade of 3 = 1 + 2 (paper3): the angular labels map a sphere around the centre onto the label sphere S^2 (compact).
#     The flux of N_2 through a closed surface = (label-sphere area) x degree, an integer (pi_2(S^2) = Z). Check the degree of a deformed map
#     numerically: integral of the pulled-back area form / 4 pi.
def degree(f, N=240):
    th = (np.arange(N) + 0.5) * np.pi / N; phv = (np.arange(2 * N) + 0.5) * np.pi / N
    T, P = np.meshgrid(th, phv, indexing='ij'); h = np.pi / N
    X = f(T, P); Xt = (f(T + 1e-5, P) - f(T - 1e-5, P)) / 2e-5; Xp = (f(T, P + 1e-5) - f(T, P - 1e-5)) / 2e-5
    dens = np.einsum('i...,i...->...', X, np.cross(Xt, Xp, axis=0))
    return dens.sum() * h * h / (4 * np.pi)
def sph(T, P, k=1, eps=0.3):
    t2 = T + eps * np.sin(T) * np.cos(2 * P); p2 = k * P + eps * np.sin(3 * T)
    v = np.array([np.sin(t2) * np.cos(p2), np.sin(t2) * np.sin(p2), np.cos(t2)])
    return v / np.linalg.norm(v, axis=0)
degs = [round(degree(lambda T, P: sph(T, P, k)), 3) for k in (1, 2, 3)]
ok(f"CI6 degrees of deformed angular label maps (winding k = 1, 2, 3): {degs}: the N_2 flux counts integers; its unit (label area) is a label",
   all(abs(d - k) < 1e-2 for d, k in zip(degs, (1, 2, 3))))

# CI7 the dilation K (D = M_56) is non-compact: exp(D t) has real eigenvalues e^{+-t}, no period, so its charge -- the scaling weight of a
#     dimensionful number -- has a continuous spectrum. Every label with K weight != 0 (masses, Lambda, G, a0, rho_vac) cannot be
#     quantised by compactness; dimensionless ratios of compact charges can.
evD = np.linalg.eigvals(D)
ok(f"CI7 eigenvalues of the dilation generator: {sorted(np.round(evD.real, 6))} (real): no period, continuous K charge", np.abs(evD.imag).max() < 1e-12)

# CI8 Lambda < 0 (anti-de Sitter, stabiliser O(2,3)): the time generator is elliptic in SO(2,3); Lambda > 0 (de Sitter, O(1,4)) has
#     compact part so(4) = rotations only. Only the type of the generator is checked here: on the universal cover used for physical AdS time
#     is not periodic, and the discrete mode spectrum needs a separate calculation with regularity and normalisability conditions. V = e5 (V^2 > 0) or e6 (V^2 < 0).
def compact_time(vidx):
    rest = [i for i in range(6) if i != vidx]
    return any(kind(M(a, b)) == 'compact' and 0 in (a, b) for a, b in itertools.combinations(rest, 2))
ok(f"CI8 compact time generator: anti-de Sitter (V = e6, keeps e0, e5): {compact_time(5)}; de Sitter (V = e5): {compact_time(4)}",
   compact_time(5) and not compact_time(4))
