# 136: the signature (1,3) from the label medium. Matter = three comoving labels X^a on a 4-manifold with metric g of signature (t, s).
# SG1 three independent labels have a one-dimensional kernel u (the flow); the label current J = *(dX^1 ^ dX^2 ^ dX^3) is along u
#     (J . dX^a = 0) in every signature;
# SG2 (observation, not used for the selection) the material metric h^ab = g^-1(dX^a, dX^b) is definite iff g restricted to the annihilator
#     of u is definite: always in (0,4), iff u is timelike in (1,3), never in (2,2) (sampled configurations);
# SG3 (2,2): for every choice of the flow axis some label direction has omega^2/k^2 = -(nu - 1) < 0 (the symbol is ultrahyperbolic): the
#     evolution of a medium with pressure is ill-posed; the selection uses only real characteristic speeds in every direction;
# SG4 linearised label medium L = -F(n), F = n^nu, nu > 1 (any medium with pressure, doc 81): the longitudinal characteristic speed is
#     omega^2/k^2 = nu - 1 in (1,3) (hyperbolic, sound speed^2 = w) and -(nu - 1) in (0,4) (elliptic: Hadamard growth e^{|k| t}, the
#     evolution along the flow is ill-posed); the symbol is computed by finite differences from the action itself;
# SG4c for general F the longitudinal speed is omega^2/k^2 = +-n F''/F' (sign by the signature): the selection holds for n F''/F' > 0;
#     a concave F reverses it;
# SG5 (3,1) is (1,3) with g -> -g: the same |det h|, the same characteristic speeds: one class;
# SG6 dimension n = 3..6 with n - 1 labels (world-line matter): real characteristic speeds in every label direction for some flow axis
#     hold only for one time direction, (1, n-1) ~ (n-1, 1).
# Run: ~/.venvs/jev/bin/python ga_signature_from_labels_check.py
import itertools
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

rng = np.random.default_rng(136)

def metric(t, n):
    return np.diag([-1.0] * t + [1.0] * (n - t))                          # t negative directions ("time"), convention-free count

def kernel(P):
    return np.linalg.svd(P)[2][-1]                                          # u with P u = 0 (P = dX, (n-1) x n)

def hodge_current(P):
    # J^mu = eps^{mu a b c} P_1a P_2b P_3c (generalised: cofactor vector of the (n-1) x n matrix)
    n = P.shape[1]
    return np.array([(-1)**mu * np.linalg.det(np.delete(P, mu, axis=1)) for mu in range(n)])

# ---------- SG1 ----------
devs = []
for t in range(5):
    g = metric(t, 4)
    for _ in range(200):
        P = rng.normal(size=(3, 4))
        u = kernel(P); J = hodge_current(P)
        devs.append(max(np.abs(P @ u).max(), np.abs(P @ J).max() / np.linalg.norm(J),
                        1 - abs(u @ J) / np.linalg.norm(J)))
ok(f"SG1 three independent labels in 4D: one-dimensional kernel u (the flow), label current J along u (max deviation {max(devs):.1e}) in every "
   "signature", max(devs) < 1e-10)

# ---------- SG2, SG3 ----------
def stats(t, n=4, N=4000):
    g = metric(t, n); gi = np.linalg.inv(g)
    definite, timelike, dets = 0, 0, []
    agree = True
    for _ in range(N):
        P = rng.normal(size=(n - 1, n))
        h = P @ gi @ P.T
        ev = np.linalg.eigvalsh(h)
        d = np.all(ev > 0) or np.all(ev < 0)
        u = kernel(P); uu = u @ g @ u
        tl = (uu < 0) if t == 1 else (uu > 0) if t == n - 1 else None
        if t in (1, n - 1):
            agree &= (d == tl)
            timelike += tl
        definite += d
        dets.append(np.linalg.det(h))
    dets = np.array(dets)
    return definite / N, timelike / N, agree, (dets > 0).mean(), (dets < 0).mean()
S = {t: stats(t) for t in range(5)}
for t in range(5):
    print(f"   ({t},{4 - t}): definite h {S[t][0]:.3f}, det h > 0 {S[t][3]:.3f}, det h < 0 {S[t][4]:.3f}")
ok(f"SG2 (observation) h = g^-1(dX, dX) is definite in (0,4) always ({S[0][0]:.3f}), in (1,3) exactly when the flow is timelike (fraction {S[1][0]:.3f} = "
   f"{S[1][1]:.3f}, agreement {S[1][2]}), in (2,2) never ({S[2][0]:.3f})",
   S[0][0] == 1 and S[1][2] and S[3][2] and S[2][0] == 0 and 0 < S[1][0] < 1)
# ---------- SG4 characteristic speeds from the action ----------
def speeds(t, n=4, nu=4 / 3, eps=1e-4, f=0, all_dirs=False, Ffun=None):
    g = metric(t, n); gi = np.linalg.inv(g)
    # background: labels = the coordinates other than f, flow along coordinate f
    P0 = np.delete(np.eye(n), f, axis=0)
    sig = np.sign(np.linalg.det(P0 @ gi @ P0.T))                        # n^2 = sig det h, positive on the background
    def L(Pf):
        d = sig * np.linalg.det(Pf.reshape(n - 1, n) @ gi @ Pf.reshape(n - 1, n).T)
        return -(abs(d))**(nu / 2) if Ffun is None else -Ffun(np.sqrt(abs(d)))
    x0 = P0.ravel(); m = x0.size
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            e_i = np.zeros(m); e_i[i] = eps; e_j = np.zeros(m); e_j[j] = eps
            H[i, j] = H[j, i] = (L(x0 + e_i + e_j) - L(x0 + e_i - e_j) - L(x0 - e_i + e_j) + L(x0 - e_i - e_j)) / (4 * eps**2)
    H = H.reshape(n - 1, n, n - 1, n)
    sym = lambda k: np.einsum('ambn,m,n->ab', H, k, k)                 # M_ab(k) = d2L/dP_am dP_bn k_m k_n
    o = (f + 1) % n                                                      # a label direction
    A = sym(np.eye(n)[f]); C = sym(np.eye(n)[o])                        # omega^2 A + k^2 C (cross term vanishes on the background)
    Bx = sym(np.eye(n)[f] + np.eye(n)[o]) - A - C
    w2 = np.linalg.eigvals(-np.linalg.solve(A, C)).real
    if all_dirs:
        # longitudinal omega^2/k^2 along every label axis and along random label-space directions
        others = [d for d in range(n) if d != f]
        dirs = [np.eye(n)[d] for d in others]
        for _ in range(20):
            v = np.zeros(n); v[others] = rng.normal(size=n - 1); dirs.append(v / np.linalg.norm(v))
        lon = []
        for kv in dirs:
            ev = np.linalg.eigvals(-np.linalg.solve(A, sym(kv))).real
            lon.append(ev[np.argmax(np.abs(ev))])
        return np.array(lon)
    return w2, np.abs(Bx).max()
w13, b13 = speeds(1); w04, b04 = speeds(0); w31, b31 = speeds(3, f=3)       # in (3,1) the flow is along the single positive axis
nu = 4 / 3
long13 = w13[np.argmax(np.abs(w13))]; long04 = w04[np.argmax(np.abs(w04))]
print(f"   omega^2/k^2 (nu = 4/3): (1,3) {np.round(np.sort(w13), 4)}, (0,4) {np.round(np.sort(w04), 4)}, (3,1) {np.round(np.sort(w31), 4)}")
ok(f"SG4 from L = -n^nu (nu = 4/3, radiation-like): longitudinal omega^2/k^2 = {long13:.4f} in (1,3) (= nu - 1 = {nu - 1:.4f}, real speed, "
   f"hyperbolic) and {long04:.4f} in (0,4) (= -(nu - 1): imaginary omega, Hadamard growth e^(|k| t), ill-posed evolution along the flow); "
   "transverse modes are the SDiff zero modes (omega = 0) in both",
   abs(long13 - (nu - 1)) < 1e-3 and abs(long04 + (nu - 1)) < 1e-3 and b13 < 1e-6 and b04 < 1e-6)
nus = [1.2, 1.5, 2.0]
lw = [(speeds(1, nu=v)[0], speeds(0, nu=v)[0]) for v in nus]
ok("SG4b the same for nu = 1.2, 1.5, 2 (stiff): (1,3) gives nu - 1 > 0, (0,4) gives -(nu - 1) < 0: "
   + ", ".join(f"{a.max():.3f}/{b.min():.3f}" for a, b in lw),
   all(abs(a.max() - (v - 1)) < 2e-3 and abs(b.min() + (v - 1)) < 2e-3 for (a, b), v in zip(lw, nus)))

lon13 = speeds(1, f=0, all_dirs=True)
lon22 = {f: speeds(2, f=f, all_dirs=True) for f in range(4)}
lon13s = speeds(1, f=1, all_dirs=True)
print(f"   (1,3), timelike flow: min omega^2/k^2 over directions {lon13.min():.4f}; spacelike flow: {lon13s.min():.4f}")
print("   (2,2), flow along axis 0..3: min omega^2/k^2 " + ", ".join(f"{lon22[f].min():.4f}" for f in range(4)))
ok(f"SG3 (2,2): for every flow axis some label direction has omega^2/k^2 = {max(l.min() for l in lon22.values()):.4f} < 0 (ultrahyperbolic "
   "symbol, ill-posed evolution); (1,3) with a timelike flow has real speeds in every direction (min "
   f"{lon13.min():.4f} = nu - 1), with a spacelike flow it does not ({lon13s.min():.4f})",
   all(l.min() < -0.3 for l in lon22.values()) and abs(lon13.min() - 1 / 3) < 1e-3 and lon13s.min() < -0.3)

# ---------- SG4c general F: the sign is (signature) x sign(n F''/F') ----------
Fg = lambda n: n**1.5 + 0.3 * n**3; cg = (0.75 + 1.8) / (1.5 + 0.9)       # n F''/F' at n = 1
Fc = lambda n: n**0.5;               cc = -0.5                               # concave F: n F''/F' = -1/2
lg13, lg04 = speeds(1, Ffun=Fg)[0].max(), speeds(0, Ffun=Fg)[0].min()
lc13, lc04 = speeds(1, Ffun=Fc)[0].min(), speeds(0, Ffun=Fc)[0].max()
ok(f"SG4c for any F the longitudinal omega^2/k^2 is +-n F''/F': F = n^1.5 + 0.3 n^3 gives {lg13:.4f} in (1,3) and {lg04:.4f} in (0,4) (n F''/F' = "
   f"{cg:.4f}); a concave F = n^0.5 gives {lc13:.4f} in (1,3) and {lc04:.4f} in (0,4): the selection of (1,3) holds for media with n F''/F' > 0 "
   "(e.g. radiation n^(4/3), fixed by blindness to the dilation, doc 81); a concave medium would favour (0,4)",
   abs(lg13 - cg) < 2e-3 and abs(lg04 + cg) < 2e-3 and abs(lc13 - cc) < 2e-3 and abs(lc04 + cc) < 2e-3)

# ---------- SG5 mirror ----------
ok(f"SG5 (3,1) = (1,3) with g -> -g: the same characteristic speeds {np.round(np.sort(w31), 4)} = {np.round(np.sort(w13), 4)}; "
   "|det h| is unchanged: one signature class", np.allclose(np.sort(w31), np.sort(w13), atol=1e-4))

# ---------- SG6 other dimensions ----------
res = {}
for n in (3, 4, 5, 6):
    for t in range(n + 1):
        good = any(speeds(t, n, nu=1.5, f=f, all_dirs=True).min() > 1e-3 for f in range(n))
        res[(n, t)] = (good,)
passing = sorted(k for k, v in res.items() if all(v))
print("   (n, t) hyperbolic: " + ", ".join(f"({n},{t})" for n, t in passing))
ok("SG6 in n = 3..6 with n - 1 labels, real sound speeds in every direction (for some flow axis) hold only for t = 1 or t = n - 1 (one time "
   "direction, up to the overall sign of g)", passing == sorted([(n, 1) for n in (3, 4, 5, 6)] + [(n, n - 1) for n in (3, 4, 5, 6)]))
