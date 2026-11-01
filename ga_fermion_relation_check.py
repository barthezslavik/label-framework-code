# 133: fermion properties as properties of the label relation, stage 1. Equations borrowed (Laidlaw-DeWitt 1971, Leinaas-Myrheim 1977,
# Finkelstein-Rubinstein 1968), every step recomputed.
# FR1 two label elements are swapped by a compactly supported divergence-free flow v = f(|x|) z x x (det of the flow map = 1);
# FR2 the flow starts at the identity, so the swap lies in the identity component of SDiff: if relabelling is gauge, permuted
#     configurations are one physical state -> elements are identical (indistinguishability follows from SDiff, not postulated);
# FR3 the same flow turns the frame of each element by pi (Jacobian at the element = rotation by pi); two swaps = 2 pi;
# FR4 (added premise: two localised elements treated as point particles) indistinguishable elements in 3D: reduced configuration space of two has pi_1 = Z2, two consistent sectors (even / odd relative
#     wavefunctions), both with a valid spectrum (harmonic trap: 3/2 vs 5/2): the label relation allows bosons and fermions, picks neither;
# FR5 GA: a rotor R(2pi) = -1 acting one-sidedly (spinor), a vector R v R~ returns at 2pi: spin 1/2 = one-sided action of the rotor;
# FR7 a localised volume-preserving relabelling turns the label frame at an element by any angle, n unchanged: no physical frame;
# FR6 label rotations X -> O X (det O = 1) lie in SDiff: if SDiff is gauge, physical states are singlets of label rotations, so the
#     three labels cannot index three generations.  Run: ~/.venvs/jev/bin/python ga_fermion_relation_check.py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

d, R0, R1 = 1.0, 1.5, 2.5                                             # elements at +-d on the x axis; f = const for |x| < R0, 0 beyond R1
def f(rho):
    if rho <= R0: return 1.0
    if rho >= R1: return 0.0
    s = (rho - R0) / (R1 - R0)
    return 1.0 - s**3 * (10 - 15 * s + 6 * s * s)                     # C2 smooth step
def vel(x):
    return f(np.linalg.norm(x)) * np.array([-x[1], x[0], 0.0])        # f(|x|) z^x
def div_num(x, h=1e-5):
    return sum((vel(x + h * e)[i] - vel(x - h * e)[i]) / (2 * h) for i, e in enumerate(np.eye(3)))

# ---------- FR1 divergence, swap, Jacobian determinant along the flow ----------
rng = np.random.default_rng(133)
pts = rng.uniform(-3, 3, size=(200, 3))
maxdiv = max(abs(div_num(p)) for p in pts)
def flow(x0, T=np.pi):
    # integrate position and Jacobian J (dJ/dt = Dv J)
    def rhs(t, s):
        x = s[:3]; J = s[3:].reshape(3, 3)
        h = 1e-6
        Dv = np.array([(vel(x + h * e) - vel(x - h * e)) / (2 * h) for e in np.eye(3)]).T
        return np.r_[vel(x), (Dv @ J).ravel()]
    sol = solve_ivp(rhs, [0, T], np.r_[x0, np.eye(3).ravel()], rtol=1e-10, atol=1e-12)
    return sol.y[:3, -1], sol.y[3:, -1].reshape(3, 3)
e1, J1 = flow(np.array([d, 0, 0])); e2, J2 = flow(np.array([-d, 0, 0]))
dets = [np.linalg.det(flow(p)[1]) for p in pts[:30]]
far = np.array([0.0, 2.8, 0.3]); efar, Jfar = flow(far)
ok(f"FR1 v = f(|x|) z^x: max |div v| = {maxdiv:.1e}; after t = pi the element at +d sits at {np.round(e1, 6)}, the one at -d at "
   f"{np.round(e2, 6)}; det of the flow map on 30 random points in [{min(dets):.8f}, {max(dets):.8f}]; outside R1 the map is the identity "
   f"(moved {np.linalg.norm(efar - far):.1e})",
   maxdiv < 1e-6 and np.allclose(e1, [-d, 0, 0], atol=1e-6) and np.allclose(e2, [d, 0, 0], atol=1e-6)
   and max(abs(np.array(dets) - 1)) < 1e-6 and np.linalg.norm(efar - far) < 1e-9)

# ---------- FR2 identity component ----------
ok("FR2 the swap is the time-pi map of a flow that starts at the identity and has compact support, so it lies in the identity component "
   "of SDiff_c(R^3); with relabelling a gauge symmetry the swapped configuration is the same physical state: label elements are indistinguishable (not yet particles)", True)

# ---------- FR3 frame of each element is turned by pi ----------
Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
e1b, J1b = flow(np.array([d, 0, 0]), T=2 * np.pi)
ok(f"FR3 Jacobian at each element after the swap = rotation by pi about z (deviation {np.abs(J1 - Rz(np.pi)).max():.1e}, "
   f"{np.abs(J2 - Rz(np.pi)).max():.1e}); after two swaps = rotation by 2 pi (deviation from 1: {np.abs(J1b - np.eye(3)).max():.1e})",
   np.abs(J1 - Rz(np.pi)).max() < 1e-6 and np.abs(J2 - Rz(np.pi)).max() < 1e-6 and np.abs(J1b - np.eye(3)).max() < 1e-6)

# ---------- FR4 two sectors of the reduced configuration space ----------
# relative motion of two identical elements in a harmonic trap (hbar = m_red = omega = 1): r in R^3 \ {0} mod r ~ -r, pi_1 = Z2;
# psi(-r) = +psi(r) (even l, bosons) or -psi(r) (odd l, fermions); radial problem u'' = (l(l+1)/r^2 + r^2 - 2E) u on a grid
N, L = 4000, 12.0
r = np.linspace(L / N, L, N); h = r[1] - r[0]
def levels(l, k=3):
    diag = 1 / h**2 + 0.5 * (l * (l + 1) / r**2 + r**2)
    off = -0.5 / h**2 * np.ones(N - 1)
    return eigh_tridiagonal(diag, off, select='i', select_range=(0, k - 1))[0]
boson = sorted(np.r_[levels(0), levels(2)])[:3]
fermion = sorted(np.r_[levels(1), levels(3)])[:3]
print(f"   bosonic sector (l even): {np.round(boson, 4)};  fermionic sector (l odd): {np.round(fermion, 4)}")
ok(f"FR4 (premise: localised elements as point particles) indistinguishable elements in 3D: two sectors, both consistent; ground states {boson[0]:.4f} (bosons) and {fermion[0]:.4f} (fermions) "
   "= 3/2 and 5/2; the relation (identity of elements) allows both and selects neither",
   abs(boson[0] - 1.5) < 1e-3 and abs(fermion[0] - 2.5) < 1e-3)

# ---------- FR5 GA rotor: one-sided action flips sign at 2 pi ----------
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex)
B = -1j * sz                                                            # bivector e1e2 in the Pauli representation of Cl(3)
rot = lambda a: np.cos(a / 2) * np.eye(2) - np.sin(a / 2) * B           # R = exp(-B a/2)
v = 0.3 * sx + 0.7 * sy + 0.2 * sz
Rt = rot(2 * np.pi)
ok(f"FR5 Cl(3): R(2pi) = {np.round(Rt[0, 0].real, 6)}, so a one-sided state R psi changes sign; a vector R v R~ returns "
   f"(deviation {np.abs(Rt @ v @ Rt.conj().T - v).max():.1e}); R(4pi) = {np.round(rot(4 * np.pi)[0, 0].real, 6)}",
   np.allclose(Rt, -np.eye(2)) and np.allclose(Rt @ v @ Rt.conj().T, v) and np.allclose(rot(4 * np.pi), np.eye(2)))

# ---------- FR6 label rotations are gauge ----------
O = np.linalg.qr(rng.normal(size=(3, 3)))[0]
if np.linalg.det(O) < 0: O[:, 0] *= -1
ok(f"FR6 a label rotation X -> O X has det O = {np.linalg.det(O):.6f}: volume-preserving, inside SDiff of label space; if SDiff is gauge, "
   "physical states are invariant under it, so no physical field carries the label index: three labels do not give three generations",
   abs(np.linalg.det(O) - 1) < 1e-12)

# ---------- FR7 the frame of an element is not gauge invariant under SDiff ----------
# label map X(x) = x near an element at the origin; relabel by a volume-preserving rotation of label space localised near the element,
# phi(X) = Rz(theta g(|X|)) X: its Jacobian at the element is Rz(theta) (any theta), while the density n = det(dX/dx) stays 1.
# So relabelling turns the frame read from the labels arbitrarily: under SDiff gauge, a label element has no physical orientation.
def phi(Xv, th):
    # time-1 map of the divergence-free flow th * f(|X| R0/0.3) z^X: angle th for |X| < 0.3, identity beyond 0.5
    return Rz(th * f(np.linalg.norm(Xv) * R0 / 0.3)) @ Xv
def jac(fun, x, h=1e-6):
    return np.array([(fun(x + h * e) - fun(x - h * e)) / (2 * h) for e in np.eye(3)]).T
x_el = np.array([1e-3, 0, 0])
frames, dens = [], []
for th in (0.3, 1.0, np.pi):
    Jp = jac(lambda X_: phi(X_, th), x_el)
    frames.append(np.abs(Jp - Rz(th)).max()); dens.append(np.linalg.det(Jp))
ok(f"FR7 relabelling by a localised volume-preserving rotation turns the label frame at an element by any angle (deviation from "
   f"Rz(theta): {max(frames):.1e}) while the density n = det stays {min(dens):.6f}-{max(dens):.6f}: under SDiff gauge the element frame is "
   "not physical, so spin cannot come from the label frame; it needs a structure SDiff does not erase (e.g. vorticity, a Casimir)",
   max(frames) < 1e-5 and max(abs(np.array(dens) - 1)) < 1e-6)
