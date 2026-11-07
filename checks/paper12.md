# paper12: checks

Scripts that compute the statements of *Quantum Mechanics in the Label Framework: One Body, Many Bodies, Statistics, Spin and Internal
Quantum Numbers* (in preparation); all print `PASS`.

- `ga_qm_labels_check` (QM1–QM8): relabelling invariance of the Fisher term; Bohm potential from its variation; Schrödinger equation
  from continuity and Bernoulli; linearity only at alpha = -1 with affine internal energy; Kelvin circulation; Wallstrom's Bessel
  solutions; n = |psi|^2 by construction; no closed one-body dynamics for an entangled state
- `ga_qm_modifications_check` (QD1–QD5, HH1): Diosi–Penrose with lengths of the series against Gran Sasso; localisation noise vs
  reversible label dynamics; guided density vs a second label medium; CHSH bound for local label values; non-linearity breaks
  superposition and isometry; hedgehog rotation equals label rotation, A = -1 at 2 pi
- `ga_qm_interpretations_check` (QI1–QI6): Bohm trajectories = label world lines; free world lines only for dust; projection rule vs
  unitary pointer records; ABL rule summed gives Born; decoherence functional; norm of a label-medium psi = label count
- `ga_dens_metric_check` (DM1–DM5): metrics on densities of configuration space under Diff(M)^N x S_N; Fisher–Rao is the only
  local metric without derivatives, but marginal Fisher–Rao metrics add at least N coefficients; Wasserstein needs the metric of M
- `ga_dens_dynamics_check` (DD1–DD5): marginal terms give the Bohm potential of marginal densities; superadditivity of Fisher
  information; non-linear evolution; for product states a change of hbar
- `ga_dens_signal_check` (DS1–DS4): marginal terms allow signalling, linear in c; no-signalling selects Fisher–Rao; estimate
  |c| < 7e-16 from the fine-structure constant
- `ga_dens_series_check` (DX1–DX3, ST1–ST4): the label medium as the marginal level; unitarity as the isometry of the
  Sasaki–Fisher–Rao metric; hbar between Wasserstein and Fisher–Rao; statistics invisible to densities, exchange phase (-1)^l
- `ga_internal_check` (IN1–IN7): relabellings of the finite algebra C + H + M_N(C) span 13 dimensions, unimodularity leaves the 12 of
  the Standard Model; hypercharges; the SU(2)^2 U(1) anomaly alone selects the unimodular line; unimodularity leaves the colour number
  free, the global SU(2) anomaly makes it odd; trace normalisation 5/3; one-loop couplings meet at no single scale
- `ga_internal_dictionary_check` (ID1–ID4): every inner relabelling keeps the trace (volume) of a finite space; unimodularity fixes the
  phase of det on H_F; its line equals the SU(2) instanton Jacobian line; det dX and Tr F F are exact top forms
- `ga_internal_spin_check` (SP1–SP3): soliton spins half-integer iff the colour number is odd; equivalence with the global SU(2)
  anomaly; Wess–Zumino five-form zero on SU(2), non-zero on SU(3)
- `ga_internal_budget_check` (IB1–IB3): anomaly planes of one generation and the unimodular line; physical parameters of D_F
  (10, 3, 21); Standard-Model labels 19 and 37, unchanged by unimodularity
