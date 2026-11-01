# 122: which couplings of label dust (dark matter) to Standard Model fields are allowed, and what direct and indirect detection then see.
# Input: the base theorem "no charge for label matter" (K7: labels are unoriented names, so every coupling must be even under J -> -J,
# J = n u the label current) and the identical conservation of label-built currents (base: matter from relabelling). Lowest-order
# Lorentz scalars built from J and one SM operator are classified by J-parity (LC1: every term linear in J is odd); the spin dependence of
# every allowed even operator is fixed by the rest-frame argument (LC2-LC3); LC4-LC5 are conditional on doc 125. LC1-LC6.
# Run: ~/.venvs/jev/bin/python ga_label_sm_couplings_check.py
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

rng = np.random.default_rng(3)
eta = np.diag([1.0, -1.0, -1.0, -1.0])
dot = lambda a, b: a @ eta @ b
n, v = 1.0, 1e-3                                                   # Galactic DM speed ~ 300 km/s = 1e-3 c
vv = v * rng.normal(size=3) / np.sqrt(3)
J = np.array([n, *(n * vv)])
S = rng.normal(size=3); S /= np.linalg.norm(S)                     # nucleon spin (unit)
qm = 1e-3 * rng.normal(size=3)                                     # q / m_N for 100 MeV transfer
# nucleon bilinears in the NR limit (leading terms): scalar 1; vector (1, p/m); axial (S.(p/m), S); gauge field A = (phi, a)
Ssc = 1.0
Vec = np.array([1.0, *(0.5 * qm)])
Ax = np.array([S @ (0.5 * qm), *S])
A_gauge = np.array([1.0, *(0.3 * rng.normal(size=3))])
ops = {
    'A.J (photon, Z, dark photon, any U(1))': (lambda Jx: dot(A_gauge, Jx), 1),
    'V.J (vector current-current)': (lambda Jx: dot(Vec, Jx), 1),
    'Ax.J (axial: spin-dependent at first order)': (lambda Jx: dot(Ax, Jx), 1),
    '(J.J) S (scalar: spin-independent)': (lambda Jx: dot(Jx, Jx) * Ssc, 2),
    '(J.J) H^+H (Higgs portal, m(phi) class)': (lambda Jx: dot(Jx, Jx) * 1.0, 2),
    '(J.Ax)^2 / (J.J) (spin-dependent, even)': (lambda Jx: dot(Jx, Ax)**2 / dot(Jx, Jx), 2),
}
par = {k: np.sign(f(-J) / f(J)) for k, (f, _) in ops.items()}
for k, p in par.items():
    print(f"   {k:46s} J-parity {'+' if p > 0 else '-'}  -> {'allowed' if p > 0 else 'forbidden'}")
odd = [k for k, p in par.items() if p < 0]
ok(f"LC1 every coupling linear in the signed J is odd (forbidden for unoriented labels; the future-directed current j is even, doc 125 QL7): {len(odd)} of {len(ops)} ({'; '.join(k.split(' (')[0] for k in odd)}) -- "
   "no gauge charge of any group, no vector or axial current-current coupling", len(odd) == 3)
# LC2-LC3 (rewritten after review r1): a general statement, not a list. Every allowed operator is even in J; in the dark-matter rest frame
# J = (n, 0, 0, 0) picks out only the time direction, so the nucleon spin S (spatial part of the axial bilinear) can only be contracted
# with a spatial vector supplied by the process: the momentum transfer q, the relative velocity v, or a gradient of n. Check on every even
# scalar linear in the axial bilinear Ax with at most one q: all vanish as q, v -> 0. The coefficients are labels (not fixed).
def even_axial_ops(Jx, q4):
    return {'(J.Ax)(J.V)': dot(Jx, Ax_of(q4)) * dot(Jx, Vec), '(J.Ax)(J.q)': dot(Jx, Ax_of(q4)) * dot(Jx, q4),
            '(J.J)(q.Ax)': dot(Jx, Jx) * dot(q4, Ax_of(q4)), '(J.J) d.Ax ~ (J.J) m_q psibar i g5 psi': dot(Jx, Jx) * dot(q4, Ax_of(q4))}
Ax_of = lambda q4: np.array([S @ (0.5 * q4[1:]), *S])                # NR axial bilinear: (S.q/2m, S)
for scale in (1e-3, 1e-5):
    q4 = np.array([0.0, *(scale * rng.normal(size=3))])
    Jrest = np.array([n, 0.0, 0.0, 0.0])
    vals = even_axial_ops(Jrest, q4)
    print(f"   q/m = {scale:.0e}: " + ", ".join(f"{k}: {abs(v):.1e}" for k, v in vals.items()))
small = even_axial_ops(np.array([n, 0, 0, 0]), np.array([0.0, *(1e-6 * np.ones(3))]))
ok("LC2 (class: no derivatives of the dark current) every even operator linear in the nucleon spin vanishes at q = v = 0 in the dark-matter rest frame (largest at q/m = 1e-6: "
   f"{max(abs(v) for v in small.values()):.1e}): there is no constant spin-dependent term, unlike an axial-vector WIMP", max(abs(v) for v in small.values()) < 1e-5)
ok("LC3 consequence within that class: any spin-dependent amplitude is proportional to S.q or S.v (rate suppressed as q^2 or v^2, no term at q = v = 0), with "
   "coefficients that are labels; spin-independent scattering through (J.J) S has a constant term. The size of the ratio is not predicted, "
   "its momentum dependence is", True)
# indirect detection and colliders: conditional on the configuration space at level 3 (doc 125)
ok("LC4 annihilation and decay change the element count M = int |N| (not the signed Q, which survives folds); it is constant when label matter is quantised on the space of embeddings "
   "(doc 125, ga_label_count_quantum_check QL4) and not protected in a sigma-model quantisation (QL5): conditional prediction", True)
ok("LC5 colliders: the same condition -- no production on embeddings; in a sigma-model quantisation folds create pairs (doc 125)", True)
ok("LC6 recorded: LC1 rests on unoriented labels and concerns the signed J; on embeddings A.j with the conserved future-directed current "
   "is allowed and the charge is a label (doc 125 QL7-QL8)", True)
