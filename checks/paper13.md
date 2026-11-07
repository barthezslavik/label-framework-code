# paper13: checks

Scripts that compute the statements of *The Internal Sector in the Label Framework: Relabellings of a Finite Space, Forms, Counts and
Labels* (in preparation); all print `PASS`.

- `ga_internal_check` (IN1–IN7): relabellings of the finite algebra C + H + M_N(C) span 13 dimensions, unimodularity leaves the 12 of
  the Standard Model; hypercharges; two of four anomaly conditions select the unimodular line; unimodularity leaves the colour number
  free, the global SU(2) anomaly makes it odd; trace normalisation 5/3; one-loop couplings meet at no single scale
- `ga_internal_dictionary_check` (ID1–ID4): every inner relabelling keeps the trace (volume) of a finite space; unimodularity fixes the
  phase of det on H_F; its line equals the SU(2) instanton Jacobian line; det dX and Tr F F are exact top forms
- `ga_internal_spin_check` (SP1–SP3): soliton spins half-integer iff the colour number is odd; equivalence with the global SU(2)
  anomaly; Wess–Zumino five-form zero on SU(2), non-zero on SU(3)
- `ga_internal_budget_check` (IB1–IB3): anomaly planes of one generation and the unimodular line; physical parameters of D_F
  (10, 3, 21); Standard-Model labels 19 and 37, unchanged by unimodularity
