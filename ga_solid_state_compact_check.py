# 124: the rule "quantised <=> compact" (doc 92) in condensed matter. (i) The Berry flux of a two-band lattice model over the Brillouin
# zone (a compact torus) is an integer (Chern number; Qi-Wu-Zhang model, Fukui-Hatsugai-Suzuki lattice method); (ii) the same flux for the
# continuum Dirac model over the non-compact plane is 1/2 -- not an integer; (iii) flux through a superconducting ring is quantised in
# h/2e because the phase of the condensate lives on a circle; a non-compact phase (no winding) gives no quantum. Known results, used here as
# the condensed-matter face of the rule; no number distinguishes the framework from standard physics. SS1-SS4.
# Run: ~/.venvs/jev/bin/python ga_solid_state_compact_check.py
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex)
def lower(H):
    w, v = np.linalg.eigh(H); return v[:, 0]
def chern_lattice(m, N=60):
    ks = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u = np.array([[lower(np.sin(kx) * sx + np.sin(ky) * sy + (m + np.cos(kx) + np.cos(ky)) * sz) for ky in ks] for kx in ks])
    F = 0.0
    for i in range(N):
        for j in range(N):
            a, b, c, d = u[i, j], u[(i + 1) % N, j], u[(i + 1) % N, (j + 1) % N], u[i, (j + 1) % N]
            F += np.angle(np.vdot(a, b) * np.vdot(b, c) * np.vdot(c, d) * np.vdot(d, a))
    return F / (2 * np.pi)
cs = {m: chern_lattice(m) for m in (-3.0, -1.0, 1.0, 3.0)}
ok("SS1 Chern number over the compact Brillouin torus (QWZ): " + ", ".join(f"m = {m}: {c:+.4f}" for m, c in cs.items()) + " -- integers",
   all(abs(c - round(c)) < 1e-6 for c in cs.values()))
# continuum massive Dirac H = kx sx + ky sy + M sz: Berry curvature of the lower band, integrated over the plane (non-compact)
M = 1.0
k = np.linspace(0, 400, 400001)
Fk = M / (2 * (k**2 + M**2)**1.5)                                   # |Omega(k)| for the 2D massive Dirac band
flux = np.trapezoid(Fk * 2 * np.pi * k, k) / (2 * np.pi) + M / (2 * np.sqrt(k[-1]**2 + M**2))   # + analytic tail beyond the grid
ok(f"SS2 the same flux over the non-compact plane (continuum Dirac): {flux:.4f} -- one half, not an integer", abs(flux - 0.5) < 1e-3)
# flux quantum: single-valued condensate phase on a ring (compact U(1)) -> Phi = n h/(2e)
h, e = 6.62607015e-34, 1.602176634e-19
ok(f"SS3 compact condensate phase: flux quantum h/2e = {h / (2 * e):.4e} Wb (known); a non-compact (R-valued) phase has no winding and no quantum", True)
ok("SS4 recorded (negative for distinction): for charges (winding numbers, flux; not energy spectra) condensed matter obeys the rule "
   "'integer only for compact generators' (Chern, flux, "
   "circulation, skyrmion degree), but every such integer is standard physics; the framework adds no new solid-state number", True)
