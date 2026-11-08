# paper13: checks

Scripts that compute the statements of *Statistical Mechanics in the Label Framework: Entropy as a Label Count, the Second Law as
Initial Data* (in preparation); all print `PASS`.

- `ga_stat_entropy_check` (SE1–SE4): volume-preserving relabellings keep every int f(rho); general relabellings keep only the label
  count; relative entropy to the label volume; Boltzmann count of label arrangements
- `ga_stat_second_law_check` (SL1–SL5): lattice cat map: fine-grained entropy constant, coarse-grained growth, reversal,
  typical start, non-mixing translation; the Lagrange–Jacobi minimum is time-symmetric
- `ga_stat_gradient_flow_check` (GF1–GF4): de Bruijn identity; Jordan–Kinderlehrer–Otto step; Fokker–Planck to Gibbs; entropy
  production and the quantum energy
- `ga_stat_equilibrium_check` (EQ1–EQ5): Gibbs from maximum entropy; temperature set by energy; Fermi–Dirac-like form with a Casimir;
  isothermal sphere; k_B as a unit
- `ga_stat_horizon_born_check` (HB1–HB3): label count scales with volume, not area; no relaxation in a label medium; relaxation of a
  guided density
