# 138: ordinary matter in the framework: electron, charge, gauge groups, generations, mass.
# Objects: Cl(1,3) (signature +---), the even subalgebra Cl+ (8 real dimensions, the Hestenes spinor psi), the label medium and the lumps of 134.
# OM1 the right multiplications psi -> psi B (B in Cl+) commute with every Lorentz rotor acting from the left; the commutant of the left
#     Spin(1,3) action in gl(Cl+) = gl(8, R) has dimension 8: it is exactly Cl+ acting from the right (no other internal operators);
# OM2 the right multiplications that keep the Dirac current J = psi g0 psi~ for every psi form the Lie algebra span{I, I s1, I s2, I s3}
#     = u(2) = su(2) + u(1): the largest internal symmetry of one rotor field is U(2), the electroweak form (Hestenes 1982, known);
# OM3 the Dirac equation  grad psi I s3 = m psi g0  keeps only right multiplications commuting with I s3 and g0: span{I s3} (U(1), the
#     electric phase) for m != 0; I (chiral) is lost by the mass; the su(2) part is not a symmetry of the free massive equation;
# OM4 su(3) (dimension 8, compact) does not act on ONE rotor field: its compact internal algebra is 4-dimensional; three real labels carry
#     no complex structure (det(-1_3) < 0), and sl(3,R) (linear relabellings) and su(3) are two real forms of sl(3,C) with Killing
#     signatures (5,3) and (0,8): colour is not the label group. With three copies psi^a and the complex structure of right multiplication
#     by I s3, U(3) acts on the copy index commuting with Lorentz and keeping sum_a J0(psi^a): SU(3) is admissible with an added index,
#     which is not derived;
# OM5 Fierz: for every spinor J.J = (psi-bar psi)^2 + (psi-bar i g5 psi)^2 >= 0: the Dirac current is causal; on solutions of the Dirac
#     equation it is conserved (known, not recomputed), and a conserved timelike current is locally a label current (doc 22 E12, straightening,
#     not recomputed). Reading (hypothesis): the electron = label current plus a physical rotor frame R;
# OM6 lumps of 134 have J = I half-integer: their label-isospin multiplets have dimension 2J+1 = 2, 4, 6, 8: even, so no fermion triplet:
#     three generations are not a label-isospin multiplet;
# OM7 charges on one compact U(1): all charges are integer multiples of a smallest unit q0 (commensurate); q0 itself is not fixed (a convention;
#     the observable is the coupling times the charge weights: one label, e): 1/3 is allowed with q0 = 1/3, an irrational ratio is not; masses: the mass term has dilation
#     weight 1 (doc 92: non-compact, never quantised): mass values are labels.
# OM8a illustration: composites Psi^a = psi B^a with constant B^a; su(3) never keeps their image (the general statement is OM9);
# OM8 copies built from the framework's own objects, psi^a = psi v^a with one rotor field and a real label vector v (e.g. components of
#     dX^a), form an 8x3 matrix of rank 1; su(3) acting through the complex structure I s3 maps them to rank 2: SU(3) is not a symmetry of
#     label-built copies, and such copies are not independent fields (one psi, no independent masses): colour and generations need three
#     independent rotor fields, which are not objects of the framework.
# OM9 general: the symmetries of a theory of the framework's fields (one rotor field, labels, the bivector force field) that keep the type of
#     each field and commute with Lorentz form a subgroup of U(1)_psi (OM3) x U(1)_duality x smooth actions on the 3-dim label space;
#     su(3) is perfect ([su(3), su(3)] = su(3), computed), so its homomorphisms to the abelian factors are trivial, and its smooth actions on a
#     3-manifold are trivial (OM4e): within the class of local point symmetries keeping the type of each field and linear on psi, SU(3) acts
#     trivially; nonlinear point maps of psi are not classified.
# Run: ~/.venvs/jev/bin/python ga_ordinary_matter_check.py
import itertools
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# ---------- a small Cl(1,3): basis blades as bitmasks, metric (+,-,-,-) ----------
SIG = [1, -1, -1, -1]
def blade_mul(a, b):
    s = 1
    x = a >> 1
    while x:
        s *= (-1)**bin(x & b).count('1')
        x >>= 1
    common = a & b
    for i in range(4):
        if common >> i & 1:
            s *= SIG[i]
    return s, a ^ b
class MV:
    def __init__(self, c=None):
        self.c = np.zeros(16) if c is None else np.array(c, float)
    def __mul__(self, o):
        if not isinstance(o, MV):
            return MV(self.c * o)
        r = np.zeros(16)
        for i in np.nonzero(self.c)[0]:
            for j in np.nonzero(o.c)[0]:
                s, k = blade_mul(i, j)
                r[k] += s * self.c[i] * o.c[j]
        return MV(r)
    __rmul__ = lambda self, o: MV(self.c * o)
    def __add__(self, o): return MV(self.c + o.c)
    def __sub__(self, o): return MV(self.c - o.c)
    def __neg__(self): return MV(-self.c)
    def rev(self):
        r = self.c.copy()
        for k in range(16):
            g = bin(k).count('1')
            if (g * (g - 1) // 2) % 2: r[k] = -r[k]
        return MV(r)
def e(*idx):
    m = MV(); k = 0; s = 1
    v = MV(); v.c[0] = 1
    for i in idx:
        b = MV(); b.c[1 << i] = 1
        v = v * b
    return v
one = e(); g = [e(i) for i in range(4)]; I = g[0] * g[1] * g[2] * g[3]
s_ = [g[k] * g[0] for k in (1, 2, 3)]                                        # sigma_k = gamma_k gamma_0
even = [k for k in range(16) if bin(k).count('1') % 2 == 0]                  # 8 even blades
def vec_even(M): return M.c[even]
def mv_even(v):
    m = MV(); m.c[even] = v; return m
basis = [mv_even(np.eye(8)[i]) for i in range(8)]
def op_matrix(f):
    return np.array([vec_even(f(b)) for b in basis]).T                       # 8 x 8 real matrix of a linear map on Cl+

# ---------- OM1 commutant of the left Lorentz action ----------
biv = [g[i] * g[j] for i, j in itertools.combinations(range(4), 2)]
Ls = [op_matrix(lambda p, B=B: B * p) for B in biv]
rows = []
for L in Ls:                                                                 # X L - L X = 0, X unknown 8x8
    rows.append(np.kron(L.T, np.eye(8)) - np.kron(np.eye(8), L))
A = np.vstack(rows)
sv = np.linalg.svd(A, compute_uv=False)
dim_comm = int(np.sum(sv < 1e-9))
Rs = [op_matrix(lambda p, B=B: p * B) for B in basis]
in_span = np.linalg.matrix_rank(np.array([R.ravel() for R in Rs]))
comm_ok = all(np.abs(R @ L - L @ R).max() < 1e-12 for R in Rs for L in Ls)
ok(f"OM1 operators on Cl+ commuting with every left Lorentz bivector: dimension {dim_comm}; right multiplications by Cl+ span {in_span} and "
   f"commute ({comm_ok}): the internal operators of a rotor field are exactly the right multiplications", dim_comm == 8 and in_span == 8 and comm_ok)

# ---------- OM2 right multiplications keeping the Dirac current ----------
# infinitesimal: J' = psi (B g0 + g0 B~) psi~ ; it vanishes for every psi iff B g0 + g0 B~ = 0
M = np.array([(B * g[0] + g[0] * B.rev()).c for B in basis]).T                # 16 x 8
null = np.linalg.svd(M)[2][np.sum(np.linalg.svd(M, compute_uv=False) > 1e-9):]
dim_u = null.shape[0]
cand = [I, I * s_[0], I * s_[1], I * s_[2]]
cand_in = np.linalg.matrix_rank(np.vstack([null, np.array([vec_even(c) for c in cand])])) == dim_u
sq = [float((c * c).c[0]) for c in cand]
comm12 = I * s_[0] * (I * s_[1]) - I * s_[1] * (I * s_[0])
su2_ok = np.allclose(comm12.c, (2 * (I * s_[2])).c) or np.allclose(comm12.c, (-2 * (I * s_[2])).c)
I_central = all(np.allclose((I * c - c * I).c, 0) for c in cand)
ok(f"OM2 right multiplications keeping J = psi g0 psi~: a {dim_u}-dimensional algebra spanned by I, I s1, I s2, I s3 ({cand_in}); squares "
   f"{sq}; [I s1, I s2] = +-2 I s3 ({su2_ok}); I central ({I_central}): u(2) = su(2) + u(1), the electroweak form (Hestenes 1982)",
   dim_u == 4 and cand_in and all(abs(q + 1) < 1e-12 for q in sq) and su2_ok and I_central)

# ---------- OM3 the Dirac equation keeps U(1) ----------
def commutes(a, b): return np.allclose((a * b - b * a).c, 0)
keep_m = [n for n, c in zip(['I', 'I s1', 'I s2', 'I s3'], cand) if commutes(c, I * s_[2]) and commutes(c, g[0])]
keep_0 = [n for n, c in zip(['I', 'I s1', 'I s2', 'I s3'], cand) if commutes(c, I * s_[2])]
ok(f"OM3 grad psi I s3 = m psi g0 keeps the right multiplications commuting with I s3 and g0: {keep_m} for m != 0 (U(1), the electric phase); "
   f"for m = 0 also the chiral one: {keep_0}; the su(2) part needs left/right projections (Hestenes), it is not a symmetry of the free massive "
   "equation", keep_m == ['I s3'] and keep_0 == ['I', 'I s3'])

# ---------- OM4 no SU(3) ----------
# (a) dimension: compact internal algebra is 4 < 8
# (b) no complex structure on three real labels
Jtry = np.linalg.det(-np.eye(3))
# (c) Killing forms of sl(3,R) and su(3)
def gl_basis_sl3R():
    B = []
    for i in range(3):
        for j in range(3):
            if i != j:
                E = np.zeros((3, 3)); E[i, j] = 1; B.append(E)
    B.append(np.diag([1, -1, 0.0])); B.append(np.diag([0, 1, -1.0]))
    return B
def su3_basis():
    l = []
    for i in range(3):
        for j in range(i + 1, 3):
            E = np.zeros((3, 3), complex); E[i, j] = 1; E[j, i] = -1; l.append(E)
            E = np.zeros((3, 3), complex); E[i, j] = 1j; E[j, i] = 1j; l.append(E)
    l.append(np.diag([1j, -1j, 0])); l.append(np.diag([0, 1j, -1j]))
    return l
def killing_signature(Bs):
    n = len(Bs); flat = np.array([b.ravel() for b in Bs]).T
    def coords(X): return np.linalg.lstsq(flat, X.ravel(), rcond=None)[0]
    ad = [np.array([coords(X @ Y - Y @ X) for Y in Bs]).T for X in Bs]
    K = np.array([[np.trace(a @ b).real for b in ad] for a in ad])
    ev = np.linalg.eigvalsh((K + K.T) / 2)
    return int(np.sum(ev > 1e-9)), int(np.sum(ev < -1e-9))
sigR = killing_signature(gl_basis_sl3R()); sigU = killing_signature([b.real + 0 * 1j if False else b for b in su3_basis()])
# (e) nonlinear actions: a compact group acting smoothly on a 3-manifold keeps an averaged Riemannian metric, so it acts by isometries and its
#     effective quotient has dimension <= n(n+1)/2 = 6 < 8 = dim SU(3); SU(3) is simple (kernel = finite centre or everything), so every smooth
#     action of SU(3) on a 3-dimensional label space is trivial (known: Myers-Steenrod bound)
n_lab = 3; iso_bound = n_lab * (n_lab + 1) // 2; dim_su3 = 8
# (d) three copies psi^a with J = right multiplication by I s3: u(3) generators act on the copy index as Re(l) x 1 + Im(l) x J
Jm = op_matrix(lambda p: p * (I * s_[2]))
def u3_op(lam):
    return np.kron(lam.real, np.eye(8)) + np.kron(lam.imag, Jm)
gell = su3_basis()
Ls3 = [np.kron(np.eye(3), L) for L in Ls]
# quadratic form sum_a <psi^a g0 psi^a~ g0>_0 (the density J0) as a matrix Q on R^24
def J0_of(v):
    tot = 0.0
    for a_ in range(3):
        ps = mv_even(v[8 * a_:8 * a_ + 8]); tot += (ps * g[0] * ps.rev() * g[0]).c[0]
    return tot
Q = np.zeros((24, 24))
for i in range(24):
    for j in range(24):
        ei = np.eye(24)[i]; ej = np.eye(24)[j]
        Q[i, j] = (J0_of(ei + ej) - J0_of(ei - ej)) / 4
u3_comm = max(np.abs(u3_op(l) @ L - L @ u3_op(l)).max() for l in gell for L in Ls3)
u3_keep = max(np.abs(u3_op(l).T @ Q + Q @ u3_op(l)).max() for l in gell)
ok(f"OM4 su(3) has dimension 8 > 4 (the compact internal algebra of one rotor field); three real labels admit no complex structure "
   f"(det(-1_3) = {Jtry:.0f} < 0, so no real J with J^2 = -1); linear relabellings sl(3,R) have Killing signature (+,-) = {sigR}, su(3) {sigU}: "
   f"two real forms of sl(3,C); nonlinearly, isometry groups of a 3-manifold have dimension <= {iso_bound} < {dim_su3}, so every smooth SU(3) "
   "action on the label space is trivial: colour is not the label group. With three copies psi^a and the complex structure I s3, su(3) acts on the copy "
   f"index commuting with Lorentz (max {u3_comm:.1e}) and keeping the density J0 (max {u3_keep:.1e}): SU(3) is admissible with an added index, "
   "not derived", Jtry < 0 and sigR == (5, 3) and sigU == (0, 8) and u3_comm < 1e-12 and u3_keep < 1e-12 and iso_bound < dim_su3)

# ---------- OM5 Fierz: the Dirac current is a label current ----------
gm = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], complex)]
sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
for sk in sig:
    gm.append(np.block([[np.zeros((2, 2)), sk], [-sk, np.zeros((2, 2))]]).astype(complex))
g5 = 1j * gm[0] @ gm[1] @ gm[2] @ gm[3]
eta = np.diag([1, -1, -1, -1.0])
rng = np.random.default_rng(138)
worst, minJ2 = 0.0, np.inf
for _ in range(2000):
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    bar = psi.conj() @ gm[0]
    J = np.array([(bar @ gm[m] @ psi).real for m in range(4)])
    sc = (bar @ psi).real; ps = (bar @ (1j * g5) @ psi).real
    J2 = J @ eta @ J
    worst = max(worst, abs(J2 - (sc**2 + ps**2)) / (J[0]**2)); minJ2 = min(minJ2, J2 / J[0]**2)
ok(f"OM5 Fierz identity J.J = (psi-bar psi)^2 + (psi-bar i g5 psi)^2 on 2000 random spinors (deviation {worst:.1e}); min J.J/J0^2 = {minJ2:.3f} >= 0: "
   "the Dirac current is causal; conserved on Dirac solutions (known) it is locally a label current (22 E12); 'electron = label current + "
   "rotor frame' is a reading, not derived",
   worst < 1e-10 and minJ2 >= -1e-12)

# ---------- OM6 no fermion triplet ----------
dims = [int(2 * j + 1) for j in (0.5, 1.5, 2.5, 3.5)]
ok(f"OM6 lumps with J = I half-integer (134 LS5) form label-isospin multiplets of dimension {dims}: always even, never 3: three generations "
   "are not a label-isospin multiplet of fermionic lumps", all(d % 2 == 0 for d in dims))

# ---------- OM7 charges and masses ----------
def commensurate(qs, Nmax=60):
    # one compact U(1): exists a period 2 pi/q0 with every q/q0 integer, q0 = 1/N for some N <= Nmax (rational charges)
    return any(all(abs(q * N - round(q * N)) < 1e-9 for q in qs) for N in range(1, Nmax + 1))
sets = {'(1, -1, 2)': [1, -1, 2], '(-1, 2/3, -1/3)': [-1, 2 / 3, -1 / 3], '(1, sqrt 2)': [1, np.sqrt(2)]}
res7 = {k: commensurate(v) for k, v in sets.items()}
ok(f"OM7 charges on one compact U(1) must be integer multiples of one unit q0: {res7}; 1/3 is allowed (q0 = 1/3), an irrational ratio is not; "
   "the unit q0 is a convention, the observable is the coupling times the charge weights (one label, e); the mass term m psi g0 has dilation weight 1, a non-compact charge (doc 92): mass "
   "values are labels", res7 == {'(1, -1, 2)': True, '(-1, 2/3, -1/3)': True, '(1, sqrt 2)': False})

# ---------- OM8 label-built copies ----------
# general Lorentz-covariant composite linear in one rotor field: by OM1, Psi^a = psi B^a with arbitrary B^a in Cl+ (labels enter the B^a)
Rmat = lambda B: op_matrix(lambda p, B=B: p * B)
inv_fail = 0
for _ in range(30):
    Bs = [mv_even(rng.normal(size=8)) for _ in range(3)]
    Mc = np.vstack([Rmat(B) for B in Bs])                                     # 24 x 8, Psi = Mc psi
    r0 = np.linalg.matrix_rank(Mc, tol=1e-9)
    lam = sum(rng.normal() * l for l in gell)
    r1 = np.linalg.matrix_rank(np.hstack([Mc, u3_op(lam) @ Mc]), tol=1e-9)
    inv_fail += (r1 > r0)
ok(f"OM8a (illustration; the general statement is OM9) composites Psi^a = psi B^a (B^a in Cl+ arbitrary, 30 samples): an su(3) generator "
   f"moves the composites out of their image in {inv_fail}/30 random cases (special B^a, e.g. B^1 = 1, B^2 = B^3 = 0, are kept by the diagonal generator); su(3) is simple of dimension 8 and the compact internal algebra is u(2) (dim 4), "
   "so no nontrivial su(3) action commutes with Lorentz on such composites", inv_fail == 30)

rk_before, rk_after = [], []
for _ in range(50):
    psi = rng.normal(size=8); v = rng.normal(size=3)
    M = np.outer(psi, v)                                                      # columns psi^a = psi v^a
    rk_before.append(np.linalg.matrix_rank(M, tol=1e-9))
    lam = sum(rng.normal() * l for l in gell)
    U = np.eye(24) + 0.3 * u3_op(lam)                                         # near-identity su(3) action on R^3 x Cl+
    vec = np.concatenate([M[:, a_] for a_ in range(3)])
    M2 = (U @ vec).reshape(3, 8).T
    rk_after.append(np.linalg.matrix_rank(M2, tol=1e-9))
ok(f"OM8 label-built copies psi^a = psi v^a have rank {max(rk_before)} (one rotor field); an su(3) action through I s3 gives rank "
   f"{min(rk_after)}..{max(rk_after)}: SU(3) does not act on label-built copies, and they carry no independent masses; colour and three "
   "generations need three independent rotor fields, which are not objects of the framework",
   max(rk_before) == 1 and min(rk_after) == 2)

# ---------- OM9 su(3) is perfect: no nontrivial map to the abelian symmetries ----------
flat8 = np.array([g_.ravel() for g_ in gell]).T
comms = [X @ Y - Y @ X for X in gell for Y in gell]
rank_comm = np.linalg.matrix_rank(np.array([c.ravel() for c in comms]), tol=1e-9)
ok(f"OM9 [su(3), su(3)] spans {rank_comm} of 8 dimensions (su(3) is perfect): every homomorphism to the abelian internal symmetries (U(1) of "
   f"the massive rotor field, OM3; U(1) duality of the force field) is trivial, and smooth actions on the 3-dim label space are trivial "
   f"(OM4e, dimension <= {iso_bound}): in the class of local point symmetries keeping the type of each field and acting linearly on psi "
   "(linear maps of psi = right multiplications, OM1) SU(3) acts trivially; nonlinear maps of psi not classified; nonlocal or field-mixing symmetries not considered; colour is an input",
   rank_comm == 8 and iso_bound < dim_su3)
