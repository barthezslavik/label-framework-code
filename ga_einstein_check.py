# Where Einstein's equation comes from: selection by conservation (like the Euler selection in 22), and unimodular (SDiff) Lambda.
# papers/restart/28-einstein.md, J1-J5.
# Run: uv run --with sympy python ga_einstein_check.py
import sympy as sp
from ga_core import ok

def geometry(g, X):
    n = len(X); gi = g.inv()
    Gm = [[[sp.simplify(sum(gi[r, s] * (sp.diff(g[s, m], X[k]) + sp.diff(g[s, k], X[m]) - sp.diff(g[m, k], X[s])) for s in range(n)) / 2)
            for k in range(n)] for m in range(n)] for r in range(n)]
    def Riem(r, s, m, k):
        return sp.diff(Gm[r][k][s], X[m]) - sp.diff(Gm[r][m][s], X[k]) + sum(Gm[r][m][l] * Gm[l][k][s] - Gm[r][k][l] * Gm[l][m][s] for l in range(n))
    Ric = sp.Matrix(n, n, lambda s, k: sp.simplify(sum(Riem(r, s, r, k) for r in range(n))))
    Rs = sp.simplify(sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)))
    def div(S):          # (nabla^mu S_{mu nu}) for a symmetric covariant tensor S
        return sp.Matrix([sp.simplify(sum(gi[m, a] * (sp.diff(S[m, nu], X[a]) - sum(Gm[l][a][m] * S[l, nu] + Gm[l][a][nu] * S[m, l] for l in range(n)))
                                          for m in range(n) for a in range(n))) for nu in range(n)])
    return Gm, Ric, Rs, div

# J1-J2 static spherical metric with two free functions (R not constant): selection among a Ric + b R g + c g
t, r, th, ph = X = sp.symbols('t r theta phi', positive=True)
A = sp.Function('A')(r); B = sp.Function('B')(r)
g = sp.diag(A, -B, -r**2, -r**2 * sp.sin(th)**2)
Gm, Ric, Rs, div = geometry(g, X)
a, b, c = sp.symbols('a b c')
S = a * Ric + b * Rs * g + c * g
dS = div(S)
sol = sp.solve([sp.expand(sp.numer(sp.together(e))) for e in dS if e != 0], [b], dict=True)
ok(f"J1 div(a Ric + b R g + c g) = 0 identically (any A(r), B(r)) <=> {sol}: only the Einstein combination Ric - R g/2 (plus any c g = Lambda term)",
   sol == [{b: -a / 2}])
ok("J2 contracted Bianchi identity: div(Ric - R g/2) = 0 for this two-function family (an identity of the geometry, no field equation used)",
   all(sp.simplify(e) == 0 for e in div(Ric - Rs * g / 2)))

# J3 dust from labels is conserved on its own: FLRW a(tau)^2 eta, comoving u, rho = rho0/a^3 (label density / volume, 22 E12)
tau, x, y, z = Y = sp.symbols('tau x y z')
aa = sp.Function('a')(tau); rho0 = sp.symbols('rho0', positive=True)
gF = aa**2 * sp.diag(1, -1, -1, -1)
GmF, RicF, RsF, divF = geometry(gF, Y)
u_low = sp.Matrix([aa, 0, 0, 0])
T = (rho0 / aa**3) * u_low * u_low.T
ok("J3 T = (rho0/a^3) u u with comoving u: div T = 0 identically (no Einstein equation used): matter conservation comes from the labels",
   all(sp.simplify(e) == 0 for e in divF(T)))
ok("J3b with rho = rho0/a^k, div T = 0 only for k = 3: the label count fixes the dust dilution",
   sp.simplify(divF(sp.Function('f')(tau) * u_low * u_low.T)[0].subs(sp.Function('f')(tau), rho0 / aa**4).doit()) != 0)
# J4 coupling G = kappa T: the 00 component is Friedmann; a = C tau^2 solves it with kappa rho0 fixing C. kappa is the only input (a scale, A14)
kap = sp.symbols('kappa', positive=True); Cc = sp.symbols('C', positive=True)
GF = RicF - RsF * gF / 2
eq00 = sp.simplify((GF[0, 0] - kap * T[0, 0]).subs(aa, Cc * tau**2).doit())
Csol = sp.solve(eq00, Cc)
ok(f"J4 G = kappa T with a = C tau^2: 00-equation fixes C = {Csol} (kappa rho0 = 12 C); ij-equations hold: "
   f"{all(sp.simplify((GF[i, i] - kap * T[i, i]).subs(aa, Cc * tau**2).doit().subs(Cc, Csol[0])) == 0 for i in (1, 2, 3))}",
   len(Csol) == 1 and all(sp.simplify((GF[i, i] - kap * T[i, i]).subs(aa, Cc * tau**2).doit().subs(Cc, Csol[0])) == 0 for i in (1, 2, 3)))

# J5 unimodular (SDiff of the metric, det g fixed): the field equation is only the trace-free part, Ric - R g/4 = kappa (T - Tr T g/4).
#    Divergence + Bianchi (div Ric = dR/2) + div T = 0 give d(R + kappa Tr T) = 0: Lambda appears as an integration constant.
dR, dT, divRic, divTT = sp.symbols('dR dT divRic divT')
lhs = divRic - dR / 4                      # div of (Ric - R g/4)
rhs = kap * (divTT - dT / 4)               # div of kappa (T - Tr T g/4)
res = sp.solve(sp.Eq(lhs.subs(divRic, dR / 2), rhs.subs(divTT, 0)), dR)
ok(f"J5 trace-free equation + Bianchi + conservation: dR = {res[0]} => R + kappa Tr T = const (= -4 Lambda in the convention signature +---, "
   "G = kappa T + Lambda g of paper5): "
   "full Einstein equation with Lambda as an integration constant (not a parameter of the action)",
   sp.simplify(res[0] + kap * dT) == 0)
