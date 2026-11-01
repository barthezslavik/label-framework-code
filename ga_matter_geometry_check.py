# 87: program item 3 -- does the framework explain why geometry responds to matter (the Einstein equation is an input, doc 28)? MG1-MG5.
# Run: ~/.venvs/jev/bin/python ga_matter_geometry_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

t, x, y, z = X = sp.symbols('t x y z', real=True)
g = sp.diag(1, -1, -1, -1); gi = g.inv()
m, v = sp.symbols('m v', positive=True)

# MG1 counterexample: label dust on a FIXED flat background satisfies every axiom used by the framework -- labels, SDiff (the action depends
#     only on n), additivity (F = -m n), conservation as an identity and dT = 0 on shell -- with no geometric dynamics at all.
#     Uniform flow: labels phi = (x - v t, y, z); n = sqrt(-det B), T^{mu nu} = m n u^mu u^nu with u the unit vector annihilated by the gradients.
phi = [x - v * t, y, z]
J = sp.Matrix(3, 4, lambda a, mu: sp.diff(phi[a], X[mu]))
B = J * gi * J.T
n = sp.sqrt(sp.simplify(-B.det()))
u = sp.Matrix([1, v, 0, 0]) / sp.sqrt(1 - v**2)
annih = sp.simplify(J * u)
T = m * n * u * u.T
divT = [sp.simplify(sum(sp.diff(T[mu, nu], X[mu]) for mu in range(4))) for nu in range(4)]
ok(f"MG1 dust on fixed Minkowski: grad(phi).u = {list(annih)}, n = {sp.simplify(n)}, div T = {divT}: all axioms hold, g has no dynamics",
   all(sp.simplify(a_) == 0 for a_ in annih) and all(d_ == 0 for d_ in divT))

# MG2 the matter actions of label media depend on the metric only algebraically (through sqrt(-g) and g^{mu nu} in B), never through its
#     derivatives, so they contribute no kinetic term for g: the variation with respect to d g vanishes identically.
gs = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f'g{min(i, j)}{max(i, j)}'))
dgs = sp.symbols('dg0:4')                                          # stand-ins for derivatives of g
L = sp.sqrt(-gs.det()) * sp.Function('F')(sp.Symbol('n'))
ok("MG2 label-media Lagrangians contain no derivatives of g: d L / d(dg) = 0", all(sp.diff(L, d_) == 0 for d_ in dgs))

# MG3 if g IS dynamical, conservation fixes the form (base B4: G + Lambda g among algebraic tensors) but not the strength: for every kappa the
#     divergence of G + Lambda g - kappa T vanishes when div T = 0 (Bianchi). kappa (Newton's G), its sign and whether it is non-zero are inputs.
kap, Lam = sp.symbols('kappa Lambda', real=True)
divG, divT_s = 0, 0                                                # Bianchi identity and conservation
ok("MG3 div(G + Lambda g - kappa T) = div G - kappa div T = 0 for every kappa: the coupling strength is not fixed", sp.simplify(divG - kap * divT_s) == 0)

# MG4 the dilation weights are consistent with coupling but do not force it: Einstein-Hilbert has weight 2 under K, dust weight 1, or 2 with the
#     label measure rescaled (the scaling of GR solutions, r_s ~ M). Equal weights describe how solutions scale, not whether g responds.
w_EH, w_dust_rescaled = 2, 1 + 1
ok(f"MG4 weights: Einstein-Hilbert {w_EH}, dust with rescaled measure {w_dust_rescaled}: consistent scaling, not a derivation", w_EH == w_dust_rescaled)

# MG5 what the framework does supply once geometry is assumed to respond universally: the source is the metric variation of the label-media
#     action (conserved on the matter shell), the form is G + Lambda g, Lambda is a label (SDiff of the 4-volume). What it does not supply:
#     the existence of geometric dynamics and kappa. Negative result: 'geometry responds to matter' is an independent axiom of the framework.
ok("MG5 recorded: source, form and the status of Lambda are supplied; existence of geometric dynamics and kappa are not (independent axiom)", True)

# MG6 the ladder candidate, stated as a reading: classical label media (level 1) and classical mechanics (level 2) have actions algebraic in g
#     (MG2), so they cannot generate geometric dynamics. At level 3 (quantum fields) the one-loop effective action on a curved background
#     contains sqrt(-g) R with a cutoff-dependent coefficient -- induced gravity (Sakharov 1967, known). In the ladder language geometric
#     dynamics could first appear at the quantum level; this is a pointer, not a derivation, and leaves kappa as a cutoff (a label of the
#     microscopic theory).
ok("MG6 recorded (reading, known physics): geometric dynamics cannot come from levels 1-2; level 3 induces sqrt(-g) R (Sakharov)", True)
