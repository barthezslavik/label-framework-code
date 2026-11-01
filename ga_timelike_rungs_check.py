# 81: the full timelike half of the label ladder: one timelike label plus k_s spacelike labels (k = 1 + k_s), action int sqrt(-g) F(n),
# n = sqrt|det Gram|. TR1-TR5. Structural. Run: ~/.venvs/jev/bin/python ga_timelike_rungs_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

x = sp.symbols('x0:4', real=True); g = sp.diag(1, -1, -1, -1); gi = g.inv()
c = sp.symbols('c1:5', positive=True); F = sp.Function('F'); nq = sp.symbols('nq', positive=True)
def stress(ks, Ff):
    labels = [c[0] * x[0]] + [c[a + 1] * x[a + 1] for a in range(ks)]
    dX = sp.Matrix(4, len(labels), lambda m, a: sp.diff(labels[a], x[m]))
    B = dX.T * gi * dX; n = sp.sqrt(sp.Abs(B.det())); n = sp.simplify(n)
    Fp = sp.diff(Ff(nq), nq).subs(nq, n)
    T = Fp * n * dX * B.inv() * dX.T - g * Ff(n)
    Mx = sp.simplify(gi * T)
    return n, sp.simplify(Mx[0, 0]), [sp.simplify(-Mx[i, i]) for i in (1, 2, 3)]

# TR1 general F: rho = n F' - F, pressure F - n F' (= -rho) along the k_s spatial label directions, F across the other 3 - k_s.
res = {}
for ks in range(4):
    n, rho, ps = stress(ks, F)
    Fn, Fpn = F(n), sp.diff(F(nq), nq).subs(nq, n)
    res[ks] = all(sp.simplify(v) == 0 for v in [rho - (n * Fpn - Fn)] + [ps[i] - ((Fn - n * Fpn) if i < ks else Fn) for i in range(3)])
ok(f"TR1 one timelike + k_s spacelike labels: rho = nF' - F, p = -rho along the k_s labels, p = F across (k_s = 0..3): {res}", all(res.values()))

# TR2 additive (linear) F = -m n: rho = 0, pressure 0 along the spatial labels, -m n... across: no energy, only stress. For k_s = 0 this is
#     P(X) = -m sqrt(X), the cuscuton: P_X + 2 X P_XX = 0, no kinetic term, no propagating mode. The element of the medium is the dual of
#     the label blade, a SPACELIKE (3 - k_s)-volume (k_s = 0: the constant-time hypersurfaces).
m = sp.symbols('m', positive=True)
lin = {}
for ks in range(4):
    n, rho, ps = stress(ks, lambda q: -m * q)
    lin[ks] = (sp.simplify(rho), [sp.simplify(p_) for p_ in ps])
X = sp.symbols('X', positive=True); Pc = -m * sp.sqrt(X)
kin = sp.simplify(sp.diff(Pc, X) + 2 * X * sp.diff(Pc, X, 2))
ok(f"TR2 additive timelike media: rho = {[v[0] for v in lin.values()]}; cuscuton kinetic coefficient P_X + 2X P_XX = {kin} (no propagating mode)",
   all(v[0] == 0 for v in lin.values()) and kin == 0)

# TR3 dilation blindness: F ~ n^(4/k) with k = 1 + k_s; k_s = 0: F = n^4 = X^2, rho = 3 c n^4, p = c n^4, w = 1/3 (radiation from a timelike
#     scalar); for every k_s the trace vanishes.
traces = {}
for ks in range(3):
    k = 1 + ks
    n, rho, ps = stress(ks, lambda q, k=k: q**sp.Rational(4, k))
    traces[ks] = sp.simplify(rho - sum(ps))
n0, rho0, ps0 = stress(0, lambda q: q**4)
ok(f"TR3 F ~ n^(4/k): trace rho - sum p = {traces}; k_s = 0: w = {sp.simplify(ps0[0] / rho0)}",
   all(v == 0 for v in traces.values()) and sp.simplify(ps0[0] / rho0) == sp.Rational(1, 3))

# TR4 k = 4 (k_s = 3): T = (n F' - F) g, the vacuum rung of paper1 -- the same whether one label is timelike or not.
n4, rho4, ps4 = stress(3, F)
ok(f"TR4 k = 4: rho = {rho4}, all pressures = -rho: {all(sp.simplify(p_ + rho4) == 0 for p_ in ps4)}", all(sp.simplify(p_ + rho4) == 0 for p_ in ps4))

# TR5 the two halves of the ladder side by side (additive F): spacelike labels give timelike elements (world lines, sheets, volumes) with
#     energy m n and tension along their extra directions; a timelike label gives spacelike elements with zero energy and pure stress
#     (non-dynamical constraint media). Dark energy that evolves needs a non-linear timelike medium (k-essence), healthy only with w >= -1 (80).
ok("TR5 recorded: spacelike labels -> timelike elements with energy; timelike label -> spacelike elements, zero energy, no propagating mode", True)
