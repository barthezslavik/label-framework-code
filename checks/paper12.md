# paper12: checks

Scripts that compute the statements of *Quantum Mechanics in the Label Framework: What Labels Derive, What They Postulate, and What
They Say About Interpretations* (in preparation); all print `PASS`.

- `ga_qm_labels_check` (QM1–QM8): relabelling invariance of the Fisher term; Bohm potential from its variation; Schrödinger equation
  from continuity and Bernoulli; linearity only at alpha = -1 with affine internal energy; Kelvin circulation; Wallstrom's Bessel
  solutions; n = |psi|^2 by construction; no closed one-body dynamics for an entangled state
- `ga_qm_modifications_check` (QD1–QD5, HH1): Diosi–Penrose with lengths of the series against Gran Sasso; localisation noise vs
  reversible label dynamics; guided density vs a second label medium; CHSH bound for local label values; non-linearity breaks
  superposition and isometry; hedgehog rotation equals label rotation, A = -1 at 2 pi
- `ga_qm_interpretations_check` (QI1–QI6): Bohm trajectories = label world lines; free world lines only for dust; projection rule vs
  unitary pointer records; ABL rule summed gives Born; decoherence functional; norm of a label-medium psi = label count
