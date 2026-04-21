# Readout Emptying Qualification Study

This study directory is the qualification-first companion to the analytic seed
API in `cqed_sim.optimal_control.readout_emptying`.

It answers a different question from the seed constructor:

- not only "can we synthesize a segmented emptying pulse?",
- but also "does that pulse survive Kerr, Lindblad readout dynamics, and hardware filtering well enough to be useful?"

## Entry points

- `linear_seed_validation.py`
  - exactness of the segmented linear construction
  - waveform, phase-space, terminal zoom, and matrix-vs-ODE evidence
- `kerr_replay_and_chirp.py`
  - nonlinear Kerr degradation and shared-chirp recovery
- `dispersive_lindblad_validation.py`
  - emitted-field separation, calibrated overlap-error validation, IQ clouds, and ringdown comparison
- `reduced_refinement.py`
  - reduced-parameter refinement plus disturbance-proxy amplitude sweep
- `hardware_sensitivity.py`
  - filter distortion, mismatch heatmap, and photon-number sensitivity
- `summary_benchmark.py`
  - consolidated square vs seed vs corrected vs refined benchmark

## Output locations

The scripts write stage outputs under:

- `outputs/readout_emptying_qualification/`

The final benchmark script also refreshes the website-ready tutorial figures under:

- `documentations/assets/images/tutorials/readout_emptying/`

That includes the summary benchmark figures plus the promoted stage plots used by
`documentations/tutorials/readout_emptying_qualification.md`.

The current shipped proof intentionally does **not** use raw assignment accuracy or raw
`non_qnd_total` as headline evidence. Instead it emphasizes:

- calibrated Gaussian overlap error,
- post-pulse ringdown time and tail energy,
- strong-readout disturbance proxies,
- Lindblad output separation,
- and robustness under mismatch and filtering.

## References

- D. T. McClure, H. Paik, L. S. Bishop, M. Steffen, J. M. Chow, and J. M. Gambetta, "Rapid Driven Reset of a Qubit Readout Resonator," Phys. Rev. Applied 5, 011001 (2016). DOI: 10.1103/PhysRevApplied.5.011001
- M. Jerger, F. Motzoi, Y. Gao, C. Dickel, L. Buchmann, A. Bengtsson, G. Tancredi, C. W. Warren, J. Bylander, D. DiVincenzo, R. Barends, and P. A. Bushev, "Dispersive Qubit Readout with Intrinsic Resonator Reset," arXiv:2406.04891 (2024). DOI: 10.48550/arXiv.2406.04891
