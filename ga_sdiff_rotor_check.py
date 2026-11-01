# Rotor formalisation of SDiff (papers/restart/47-sdiff-rotors.md). SR1-SR7.
# Claim: (i) the Lie algebra of SDiff is {v = nabla . B : B a bivector field} (exact divergence-free fields; B defined up to B + nabla . T);
# (ii) Hamiltonian (symplectic) SDiff is the case B = H J with J the symplectic bivector: v = (nabla H) . J; (iii) the 1-jet of every SDiff
# flow is a field of SL(n) elements, i.e. of rotors: the rotation part is an ordinary rotor e^{-Omega t/2}, the shear part a rotor of the
# doubled algebra Cl(n,n) (Doran-Hestenes); det = 1 is the vanishing of the scale (trace) component.
# Literature: exact divergence-free fields have an (n-2)-form potential (Arnold-Khesin); GL(n) as a spin group in Cl(n,n) (Doran, Hestenes,
# Sommen, Van Acker 1993); potentials themselves do not form a Lie algebra (hep-th/9407066) -- only the fields v do.
# Run: uv run --with sympy --with numpy --with scipy python ga_sdiff_rotor_check.py
import itertools
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from ga_core import *

def lcontract_grad(F, coords, sig):
    """nabla . F = sum_mu grade_{r-1}(e^mu d_mu F) for an r-vector field F (reciprocal frame = frame here)."""
    R = {}
    r = max(grades(F)) if F else 0
    for m, x in enumerate(coords):
        dF = {k: sp.diff(v, x) for k, v in F.items()}
        R = add(R, grade(gp({(m,): sig[m]}, dF, sig), r - 1))
    return clean({k: sp.simplify(v) for k, v in R.items()})

# SR1 nabla . (nabla . B) = 0 for a generic bivector field, in Euclidean 3D and Lorentzian 4D: every v = nabla . B is divergence-free.
for dim, sig in ((3, {0: 1, 1: 1, 2: 1}), (4, {0: 1, 1: -1, 2: -1, 3: -1})):
    X = sp.symbols(f'x0:{dim}', real=True)
    B = {pair: sp.Function(f'b{pair[0]}{pair[1]}')(*X) for pair in itertools.combinations(range(dim), 2)}
    v = lcontract_grad(B, X, sig)
    divv = lcontract_grad(v, X, sig)
    ok(f"SR1 dim {dim}: v = nabla.B is a vector field ({sorted(set(len(k) for k in v))}), nabla.v = {divv}", divv == {} and set(len(k) for k in v) == {1})

# SR2 converse on a known solution: the ABC flow (Beltrami, curl w = w) is v = nabla . B with B = w I up to sign (3D), i.e. its own potential.
E3 = {0: 1, 1: 1, 2: 1}
x, y, z = sp.symbols('x y z', real=True); X3 = (x, y, z)
A_, B_, C_ = 1, sp.Rational(2, 3), sp.Rational(1, 2)
w = vec([A_ * sp.sin(z) + C_ * sp.cos(y), B_ * sp.sin(x) + A_ * sp.cos(z), C_ * sp.sin(y) + B_ * sp.cos(x)])
I3 = {(0, 1, 2): 1}
vI = lcontract_grad(grade(gp(w, I3, E3), 2), X3, E3)
match = [sp.simplify(vI.get((k,), 0) - s * w.get((k,), 0)) for s in (1, -1) for k in range(3)]
sgn = 1 if all(m == 0 for m in match[:3]) else (-1 if all(m == 0 for m in match[3:]) else 0)
ok(f"SR2 ABC flow: nabla.(w I) = {sgn} * w: an explicit bivector potential B = {sgn} w I", sgn != 0)

# SR3 gauge: B -> B + nabla . T (T a trivector field) leaves v unchanged.
T = {(0, 1, 2): sp.Function('tau')(*X3)}
dB = lcontract_grad(T, X3, E3)
ok(f"SR3 nabla.(nabla.T) = {lcontract_grad(dB, X3, E3)}: B is fixed only up to nabla.T", lcontract_grad(dB, X3, E3) == {})

# SR4 Hamiltonian SDiff: phase plane (q, p), J = e_q ^ e_p; v = nabla.(H J) = (nabla H).J = Hamilton's vector field, divergence-free.
q, p = sp.symbols('q p', real=True)
E2 = {0: 1, 1: 1}
Hf = sp.Function('H')(q, p)
J = {(0, 1): 1}
vH = lcontract_grad(scal(Hf, J), (q, p), E2)
ok(f"SR4 nabla.(H J) = ({vH.get((0,), 0)}, {vH.get((1,), 0)}) = +-(dH/dp, -dH/dq): Hamilton's equations; divergence {lcontract_grad(vH, (q, p), E2)}",
   sp.simplify(vH.get((0,), 0) ** 2 - sp.diff(Hf, p) ** 2) == 0 and sp.simplify(vH.get((1,), 0) ** 2 - sp.diff(Hf, q) ** 2) == 0
   and lcontract_grad(vH, (q, p), E2) == {})

# SR5 1-jet: a linear divergence-free field v = L x has tr L = 0 (sl(n)); its antisymmetric part is a bivector Omega (v = x.Omega) whose
#    finite action is the rotor R = e^{-Omega t/2}; exp(t L) has det 1.
rng = np.random.default_rng(5)
L = rng.normal(size=(3, 3)); L -= np.trace(L) / 3 * np.eye(3)
Aanti = (L - L.T) / 2
tt = 0.7
Om = {(0, 1): Aanti[1, 0], (0, 2): Aanti[2, 0], (1, 2): Aanti[2, 1]}
nrm = np.sqrt(sum(c**2 for c in Om.values()))
Bunit = {k: v / nrm for k, v in Om.items()}
R = add({(): np.cos(nrm * tt / 2)}, scal(-np.sin(nrm * tt / 2), Bunit))
Rrev = add({(): np.cos(nrm * tt / 2)}, scal(np.sin(nrm * tt / 2), Bunit))
x0 = np.array([0.3, -1.1, 0.8])
rot = gp(gp(R, vec(list(x0)), E3), Rrev, E3)
rot_v = np.array([float(rot.get((k,), 0)) for k in range(3)])
ok(f"SR5 det exp(tL) = {np.linalg.det(expm(tt * L)):.12f}; rotor sandwich of the rotation part = expm(t Aanti) x: max diff "
   f"{np.max(np.abs(rot_v - expm(tt * Aanti) @ x0)):.1e} (sign convention up to Omega -> -Omega)",
   abs(np.linalg.det(expm(tt * L)) - 1) < 1e-12 and min(np.max(np.abs(rot_v - expm(tt * Aanti) @ x0)), np.max(np.abs(rot_v - expm(-tt * Aanti) @ x0))) < 1e-12)

# SR6 the whole of SL(2) (rotation + shear) as rotors of Cl(2,2): null vectors n_i = e_i + f_i, m_i = e_i - f_i; for traceless L the bivector
#    B_L = (1/2) sum L_ij n_i ^ m_j acts on the n's as L (commutator), and the rotor exp(B_L) sandwich gives expm(L) on span(n). The trace part
#    sum n_i ^ m_i is the scale K: det = 1 <=> no K component.
C22 = {0: 1, 1: 1, 2: -1, 3: -1}
nvec = [add({(0,): 1}, {(2,): 1}), add({(1,): 1}, {(3,): 1})]
mvec = [add({(0,): 1}, {(2,): -1}), add({(1,): 1}, {(3,): -1})]
l11, l12, l21, l22 = sp.symbols('l11 l12 l21 l22', real=True)
Ls = sp.Matrix([[l11, l12], [l21, l22]])                       # general L; the trace is tested below
BL = {}
for i in range(2):
    for j in range(2):
        BL = add(BL, scal(Ls[i, j] / 2, wedge(nvec[i], mvec[j], C22)))
def comm(Bv, Y):
    return clean({k: sp.simplify(v / 2) for k, v in add(gp(Bv, Y, C22), scal(-1, gp(Y, Bv, C22))).items()})
acts = []
for k in range(2):
    cm = comm(BL, nvec[k])
    target = add(scal(Ls[0, k], nvec[0]), scal(Ls[1, k], nvec[1]))
    acts.append(clean({kk: sp.simplify(cm.get(kk, 0) - target.get(kk, 0)) for kk in set(cm) | set(target)}) == {}
                or clean({kk: sp.simplify(cm.get(kk, 0) + target.get(kk, 0)) for kk in set(cm) | set(target)}) == {})
Kb = add(wedge(nvec[0], mvec[0], C22), wedge(nvec[1], mvec[1], C22))
trace_comp = sp.simplify(gp(BL, Kb, C22).get((), 0))
ratio_tr = sp.simplify(trace_comp / (l11 + l22))
ok(f"SR6 [B_L, n_k]/2 = (+-) L n_k for both k: {acts}; <B_L K>_0 = {sp.factor(trace_comp)} = {ratio_tr} tr L: the scale K is exactly the trace, "
   "so SL(2) = rotors of Cl(2,2) with no K component (det = 1)", all(acts) and ratio_tr.free_symbols == set() and ratio_tr != 0)

# SR6b the finite rotor: R = exp(B_L/2) (power series) gives R n_k R^-1 = sum_i expm(L)_ik n_i on span(n), for a numeric traceless L.
Lnum = np.array([[0.4, -0.7], [0.9, -0.4]])
BLn = {}
for i in range(2):
    for j in range(2):
        BLn = add(BLn, scal(Lnum[i, j] / 2, wedge(nvec[i], mvec[j], C22)))
def gexp(Bv, sgn):
    term, tot = {(): 1.0}, {(): 1.0}
    for kk in range(1, 30):
        term = scal(sgn / (2 * kk), gp(term, Bv, C22)); tot = add(tot, term)
    return {k: float(v) for k, v in tot.items()}
Rp, Rm = gexp(BLn, 1), gexp(BLn, -1)
Ex = expm(Lnum); errs = []
for k in range(2):
    img = gp(gp(Rp, nvec[k], C22), Rm, C22)
    target = add(scal(Ex[0, k], nvec[0]), scal(Ex[1, k], nvec[1]))
    errs.append(max(abs(float(img.get(kk, 0)) - float(target.get(kk, 0))) for kk in set(img) | set(target)))
ok(f"SR6b finite rotor exp(B_L/2) sandwich reproduces expm(L) on the null subspace: max error {max(errs):.1e}; det expm(L) = {np.linalg.det(Ex):.12f}",
   max(errs) < 1e-10 and abs(np.linalg.det(Ex) - 1) < 1e-12)

# SR7 a nonlinear SDiff flow (ABC, t = 1): its Jacobian J satisfies det J = 1 at every point, and splits (polar) into a rotation rotor times a
#    symmetric unimodular stretch -- a field of SL(3) rotors.
Af, Bf, Cf = 1.0, 2 / 3, 0.5
def vel(s, u):
    x_, y_, z_ = u[:3]; Jm = u[3:].reshape(3, 3)
    v_ = [Af * np.sin(z_) + Cf * np.cos(y_), Bf * np.sin(x_) + Af * np.cos(z_), Cf * np.sin(y_) + Bf * np.cos(x_)]
    G = np.array([[0, -Cf * np.sin(y_), Af * np.cos(z_)], [Bf * np.cos(x_), 0, -Af * np.sin(z_)], [-Bf * np.sin(x_), Cf * np.cos(y_), 0]])
    return np.concatenate([v_, (G @ Jm).ravel()])
dets, polar_ok = [], []
for start in rng.uniform(0, 2 * np.pi, size=(4, 3)):
    sol = solve_ivp(vel, [0, 1], np.concatenate([start, np.eye(3).ravel()]), rtol=1e-11, atol=1e-12)
    Jm = sol.y[3:, -1].reshape(3, 3)
    dets.append(np.linalg.det(Jm))
    U, s, Vt = np.linalg.svd(Jm); Rp = U @ Vt; Sp = Vt.T @ np.diag(s) @ Vt
    polar_ok.append(abs(np.linalg.det(Rp) - 1) < 1e-9 and abs(np.linalg.det(Sp) - 1) < 1e-8 and np.allclose(Rp @ Sp, Jm))
ok(f"SR7 ABC flow Jacobians: det = {np.round(dets, 10)}; polar split rotation x unimodular stretch at every point: {polar_ok}",
   all(abs(d - 1) < 1e-8 for d in dets) and all(polar_ok))
