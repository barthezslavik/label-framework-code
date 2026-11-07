# 146/147: step 2 of paper12 (internal quantum numbers). The internal sector as relabellings of a finite internal space with the unimodularity condition ("S").
# Chosen (premises): the algebra A_F = C + H + M_N(C) (N = 3 for the Standard Model), its representation on one generation of
# particle states (left action: lambda on nu_R, u_R; conj(lambda) on e_R, d_R; the quaternion q on left doublets; right action:
# conj(lambda) on leptons, conj(m) on the colour index of quarks; antiparticles carry the conjugate action), the chirality, the number of
# generations. Derived here by computation:
# IN1 relabellings Ad(u), u in U(A_F), span a 13-dimensional Lie algebra u(1)+u(1)+su(2)+su(3); the condition det pi(u) = 1 on H_F
#     (particles and antiparticles) removes exactly one abelian direction and leaves dimension 12, the Standard Model gauge algebra;
#     the condition det Ad(u) = 1 on particles alone removes the same direction;
# IN2 the surviving abelian generator, normalised to Y(lepton doublet) = -1, gives the hypercharges 0, -2, -1, 4/3, -2/3, 1/3 of
#     nu_R, e_R, L, u_R, d_R, Q;
# IN3 with these hypercharges all gauge and gravitational anomalies of one generation vanish; on the two-dimensional abelian
#     algebra the gravitational and SU(3)^2 U(1) anomalies vanish identically; SU(2)^2 U(1) alone selects the line of S; U(1)^3 is
#     proportional to alpha^2 (alpha + N beta) and vanishes on that line and on alpha = 0, the vector-like baryon number of the right
#     action, which violates only SU(2)^2 U(1);
# IN4 for A_F = C + H + M_N(C), N = 1..7, S gives Y(Q) = 1/N and every anomaly still vanishes: S does not fix the colour number;
# IN5 the global SU(2) anomaly (Witten 1982) needs an even total number of Weyl doublets; the representation has 1 + N per generation,
#     so with an odd number of generations (three) N must be odd; with two generations every N is allowed; S does not enter.
# Step 3 (three layers of the internal numbers):
# IN6 if the gauge kinetic terms are fixed by a trace over H_F (premise of the spectral action), the traces of T3^2, T_a^2 (colour)
#     and (Y/2)^2 over one generation are 2, 2, 10/3: g3 = g2 and g1 with the normalisation 5/3, a relation, not a value;
# IN7 one-loop running in the minimal Standard Model from M_Z: alpha_2 = alpha_3 and alpha_1 = alpha_2 (normalisation 5/3) meet at
#     scales four orders of magnitude apart (10^13 and 10^17 GeV), so the relation of IN6 holds at no single scale; the scale is not fixed.
# Run: ~/.venvs/jev/bin/python ga_internal_check.py
import numpy as np
from fractions import Fraction

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

FL = ['nuR', 'eR', 'nuL', 'eL', 'uR', 'dR', 'uL', 'dL']
RIGHT = {'nuR', 'eR', 'uR', 'dR'}

def build(N):
    """Particle states of one generation: leptons x1, quarks x N colours. Returns index list and generator matrices of Ad(u)."""
    states = [(f, 0) for f in FL[:4]] + [(f, c) for f in FL[4:] for c in range(N)]
    d = len(states); idx = {s: i for i, s in enumerate(states)}
    lep = lambda f: f in FL[:4]
    # abelian generators of Ad(u) on particles: lambda = e^{i alpha}, m = e^{i beta} 1_N
    A = np.zeros(d); B = np.zeros(d)
    for i, (f, c) in enumerate(states):
        left = {'nuR': 1, 'uR': 1, 'eR': -1, 'dR': -1}.get(f, 0)        # q_lambda = diag(lambda, conj lambda) on right-handed states
        right = -1 if lep(f) else 0                                    # right action conj(lambda) on leptons
        A[i] = left + right
        B[i] = 0 if lep(f) else -1                                     # right action conj(m) on quarks
    A = np.diag(A).astype(complex); B = np.diag(B).astype(complex)
    # su(2): left action of q on doublets (nuL, eL) and (uL, dL) for each colour
    sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    T2 = []
    for s in sig:
        M = np.zeros((d, d), complex)
        pairs = [(idx[('nuL', 0)], idx[('eL', 0)])] + [(idx[('uL', c)], idx[('dL', c)]) for c in range(N)]
        for a, b in pairs:
            M[np.ix_([a, b], [a, b])] = s / 2
        T2.append(M)
    # su(N): right action conj(m) on the colour index of quarks, generator -conj(t)
    gens = []
    for a in range(N):
        for b in range(a + 1, N):
            E = np.zeros((N, N), complex); E[a, b] = E[b, a] = 0.5; gens.append(E)
            E = np.zeros((N, N), complex); E[a, b] = -0.5j; E[b, a] = 0.5j; gens.append(E)
    for k in range(1, N):
        E = np.zeros((N, N), complex)
        for j in range(k): E[j, j] = 1
        E[k, k] = -k; gens.append(E / np.sqrt(2 * k * (k + 1)))
    TN = []
    for t in gens:
        M = np.zeros((d, d), complex)
        for f in FL[4:]:
            ii = [idx[(f, c)] for c in range(N)]
            M[np.ix_(ii, ii)] = -t.conj()
        TN.append(M)
    gam = np.diag([-1.0 if f in RIGHT else 1.0 for f, c in states])
    return states, A, B, T2, TN, gam

def left_traces(N):
    """Trace of the left representation pi(u) on particles + antiparticles for the generators alpha, beta (unimodularity)."""
    # particles: q_lambda on right-handed states gives +1, -1 pairs (trace 0 per colour); doublets carry q (traceless)
    tp_a = (1 - 1) + N * (1 - 1); tp_b = 0
    # antiparticles: lambda on the 4 lepton antistates, m on the N colours of the 4 quark antiflavours
    ta_a = 4; ta_b = 4 * N
    return tp_a + ta_a, tp_b + ta_b

def hyper(N):
    states, A, B, T2, TN, gam = build(N)
    ta, tb = left_traces(N)
    y = -ta / tb                                                       # det pi(u) = 1  <=>  ta*alpha + tb*beta = 0, beta = y*alpha
    Y = A + y * B
    return states, Y, A, B, T2, TN, gam, y

def anomalies(Y, T2, TN, gam):
    g = np.trace(gam @ Y).real
    c3 = np.trace(gam @ Y @ Y @ Y).real
    s2 = np.trace(gam @ T2[0] @ T2[0] @ Y).real
    s3 = np.trace(gam @ TN[0] @ TN[0] @ Y).real if TN else 0.0
    nn = 0.0                                                           # SU(N)^3: d-symbol traces
    for a in TN:
        for b in TN:
            for c in TN[:3]:
                nn += abs(np.trace(gam @ a @ (b @ c + c @ b)))
    return g, c3, s2, s3, nn

# ---------- IN1 dimension of the relabelling algebra with and without S ----------
states, A, B, T2, TN, gam = build(3)
basis = [A, B] + T2 + TN
V = np.array([M.flatten() for M in basis])
dim_all = np.linalg.matrix_rank(np.vstack([V.real, V.imag]).T)
ta, tb = left_traces(3)
# with S: the abelian part is restricted to ta*alpha + tb*beta = 0
basisS = [A - (ta / tb) * B] + T2 + TN
VS = np.array([M.flatten() for M in basisS])
dim_S = np.linalg.matrix_rank(np.vstack([VS.real, VS.imag]).T)
# det Ad(u) = 1 on particles alone: traces of A and B on particles
pa, pb = np.trace(A).real, np.trace(B).real
ok(f"IN1 relabellings Ad(u) span dim {dim_all} = 13; with det pi(u) = 1 dim {dim_S} = 12 (SM); particles-only det gives the same "
   f"line beta/alpha = {-pa/pb:.4f} = {-ta/tb:.4f}",
   dim_all == 13 and dim_S == 12 and abs(-pa / pb - (-ta / tb)) < 1e-12)

# ---------- IN2 hypercharges ----------
states, Y, A, B, T2, TN, gam, y = hyper(3)
val = {}
for i, (f, c) in enumerate(states):
    val.setdefault(f, Y[i, i].real)
want = {'nuR': 0, 'eR': -2, 'nuL': -1, 'eL': -1, 'uR': 4 / 3, 'dR': -2 / 3, 'uL': 1 / 3, 'dL': 1 / 3}
ok("IN2 S gives hypercharges " + ", ".join(f"{f} {Fraction(val[f]).limit_denominator(12)}" for f in FL)
   + " (Standard Model, Y(L) = -1)", all(abs(val[f] - want[f]) < 1e-12 for f in FL))

# ---------- IN3 anomalies vanish; linear anomaly conditions select the S line; the other direction is anomalous ----------
an = anomalies(Y, T2, TN, gam)
lin = np.array([[np.trace(gam @ G).real, np.trace(gam @ TN[0] @ TN[0] @ G).real] for G in (A, B)])
auto = np.abs(lin).max() < 1e-12                                       # grav and SU(3)^2 U(1) vanish on the whole abelian plane
s2 = [np.trace(gam @ T2[0] @ T2[0] @ G).real for G in (A, B)]
line_s2 = -s2[0] / s2[1]                                               # beta/alpha where SU(2)^2 U(1) vanishes
# U(1)^3 as a homogeneous cubic in (alpha, beta): c(alpha, beta) = Tr gam (alpha A + beta B)^3; fit the four coefficients exactly
pts = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (2.0, 1.0)]
vals = [np.trace(gam @ np.linalg.matrix_power(al * A + be * B, 3)).real for al, be in pts]
M3 = np.array([[al**3, al**2 * be, al * be**2, be**3] for al, be in pts])
coef = np.linalg.lstsq(M3, np.array(vals), rcond=None)[0]               # c = c30 a^3 + c21 a^2 b + c12 a b^2 + c03 b^3
assert np.allclose(M3 @ coef, vals)
c30, c21, c12, c03 = coef
cub_ok = abs(c12) < 1e-9 and abs(c03) < 1e-9 and abs(c21 / c30 + 1 / y) < 1e-9   # c = c30 a^2 (a - b/y): roots a = 0 (double), b = y a
anB = anomalies(B, T2, TN, gam)
ok(f"IN3 all anomalies of one generation vanish with S (|max| = {max(abs(x) for x in an):.1e}); on the abelian plane the "
   f"gravitational and SU(3)^2 U(1) anomalies vanish identically; SU(2)^2 U(1) selects beta/alpha = {line_s2:.4f} = S alone; "
   f"U(1)^3 = {c30:.0f} alpha^2 (alpha + {c21/c30:.0f} beta) vanishes on the S line and on alpha = 0 (baryon number of the right action, "
   f"vector-like); that direction violates only SU(2)^2 U(1) ({anB[2]:.3f})",
   max(abs(x) for x in an) < 1e-10 and auto and abs(line_s2 - y) < 1e-12 and cub_ok and abs(anB[2]) > 0.1 and abs(anB[1]) < 1e-12)

# ---------- IN4 S does not fix the colour number ----------
rows = []
good = True
for N in range(1, 8):
    st, YN, *_rest = hyper(N)
    T2N, TNN, gN = _rest[2], _rest[3], _rest[4]
    yQ = YN[[i for i, s in enumerate(st) if s[0] == 'uL'][0]].real.max()
    a = anomalies(YN, T2N, TNN, gN)
    rows.append((N, yQ, max(abs(x) for x in a)))
    good &= abs(yQ - 1 / N) < 1e-12 and max(abs(x) for x in a) < 1e-9
ok("IN4 for N = 1..7 S gives Y(Q) = 1/N and all anomalies vanish: " + "; ".join(f"N={n} Y(Q)={Fraction(q).limit_denominator(20)}"
   for n, q, _ in rows) + "; S leaves N to the choice of algebra", good)

# ---------- IN5 Witten SU(2) anomaly: the total number of Weyl doublets must be even ----------
cnt = {N: 1 + N for N in range(1, 8)}
for N in range(1, 8):
    st, *_ = build(N)
    nd = sum(1 for f, c in st if f in ('nuL', 'uL'))
    assert nd == cnt[N]
ok3 = [N for N in range(1, 8) if (3 * cnt[N]) % 2 == 0]                 # three generations
ok2 = [N for N in range(1, 8) if (2 * cnt[N]) % 2 == 0]                 # two generations, for contrast
ok(f"IN5 Weyl doublets per generation 1 + N; the global SU(2) anomaly needs an even total; with three generations it is absent only for "
   f"N in {ok3} (odd N), with two generations for every N {ok2}: odd N follows for an odd number of generations; S plays no part",
   ok3 == [1, 3, 5, 7] and ok2 == list(range(1, 8)))

# ---------- IN6 trace normalisation of the gauge kinetic terms ----------
states, Y, A, B, T2, TN, gam, y = hyper(3)
t2 = np.trace(T2[2] @ T2[2]).real
t3 = np.trace(TN[-1] @ TN[-1]).real
ty = np.trace((Y / 2) @ (Y / 2)).real
ok(f"IN6 traces over one generation: T3^2 {t2:.4f}, colour T8^2 {t3:.4f}, (Y/2)^2 {Fraction(ty).limit_denominator(12)}; "
   f"ratio (Y/2)^2 / T3^2 = {Fraction(ty / t2).limit_denominator(12)}: g3 = g2, g1^2 normalisation 5/3 (relation under the trace premise)",
   abs(t2 - 2) < 1e-12 and abs(t3 - 2) < 1e-12 and abs(ty / t2 - 5 / 3) < 1e-12)

# ---------- IN7 one-loop running: the relation holds at no single scale ----------
MZ, aem_inv, s2w, a3 = 91.1876, 127.95, 0.2312, 0.1180               # PDG values at M_Z (MS-bar)
inv = np.array([aem_inv * (1 - s2w) * 3 / 5, aem_inv * s2w, 1 / a3])  # alpha_1 (normalisation 5/3), alpha_2, alpha_3
b = np.array([41 / 10, -19 / 6, -7.0])                                # one-loop coefficients of the minimal SM
def meet(i, j):                                                       # ln(mu/M_Z) where alpha_i = alpha_j
    return 2 * np.pi * (inv[i] - inv[j]) / (b[i] - b[j])
m23 = MZ * np.exp(meet(1, 2)); m12 = MZ * np.exp(meet(0, 1)); m13 = MZ * np.exp(meet(0, 2))
ok(f"IN7 one-loop SM running: alpha_2 = alpha_3 at {m23:.1e} GeV, alpha_1 = alpha_2 at {m12:.1e} GeV, alpha_1 = alpha_3 at "
   f"{m13:.1e} GeV; spread {np.log10(m23 / m12):.1f} decades: the relation of IN6 holds at no single scale",
   np.log10(m23 / m12) > 2)

print("ALL PASS")
