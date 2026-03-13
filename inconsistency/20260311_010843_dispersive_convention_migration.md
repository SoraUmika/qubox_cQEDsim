Status: fixed on 2026-03-13 after re-audit.

## Confirmed issues

### 1. Runtime dispersive sign is opposite to the new requested canonical convention

- What: The two-mode and three-mode runtime Hamiltonians currently implement dispersive terms as `-chi * n_boson * n_q`.
- Where:
  - `cqed_sim/core/model.py`
  - `cqed_sim/core/readout_model.py`
  - `cqed_sim/core/frequencies.py`
- Affected components:
  - `DispersiveTransmonCavityModel`
  - `DispersiveReadoutTransmonStorageModel`
  - manifold-frequency helpers
  - spectroscopy predictions
- Why inconsistent: The requested repository-wide convention is `+chi * n_c n_q`, so negative `chi` must lower the qubit transition frequency with photon number.
- Consequences:
  - negative `chi` currently raises the qubit transition frequency in the runtime model
  - helper frequencies and guide-line predictions inherit the opposite physical meaning

### 2. Runtime Kerr sign is opposite to the requested explicit cavity-ladder convention

- What: Runtime cavity self-Kerr terms currently enter as `-(K/2) n(n-1)` and higher falling-factorial Kerr terms also carry a leading minus sign.
- Where:
  - `cqed_sim/core/model.py`
  - `cqed_sim/core/readout_model.py`
  - `cqed_sim/sim/couplings.py`
- Affected components:
  - two-mode and three-mode static Hamiltonians
  - generic self-Kerr helper
  - any test or example interpreting adjacent cavity transition spacing
- Why inconsistent: The requested canonical documentation target is `+(K/2) n(n-1)` with an explicit physical interpretation for the sign of `K`.
- Consequences:
  - positive `K` currently lowers the cavity ladder spacing instead of raising it
  - existing user-facing sign language around Kerr is ambiguous unless the code is audited directly

### 3. Parameter translation layer encodes the old runtime sign and compensates with a separate synthesis mapping

- What: The parameter translator extracts `chi` from `omega0 - omega1`, uses the perturbative formula with the old sign, and publishes `synthesis_chi = -0.5 * chi`.
- Where:
  - `cqed_sim/analysis/parameter_translation.py`
- Affected components:
  - bare-to-dressed translation
  - measured-parameter inversion
  - runtime-to-synthesis interoperability
- Why inconsistent: After migrating runtime semantics to `omega_ge(n) = omega_ge(0) + n chi`, the current extraction and mapping formulas would silently return the wrong sign and preserve an avoidable convention split.
- Consequences:
  - translated parameters would disagree with the Hamiltonian actually simulated
  - calibration and synthesis code could target the wrong conditioned frequencies

### 4. Unitary synthesis drift semantics are intentionally different from runtime semantics

- What: `DriftPhaseModel` currently uses a sigma-z style split with `E_e - E_g = delta_q + 2 * (chi*n + chi2*n(n-1))`, while runtime uses an `n_q` projector convention.
- Where:
  - `cqed_sim/unitary_synthesis/sequence.py`
  - tests under `tests/unitary_synthesis/`
  - API and physics documentation
- Affected components:
  - drift-phase reports
  - conditional-phase synthesis helpers
  - parameter translation outputs
- Why inconsistent: The repository currently requires an explicit sign and factor conversion between runtime and synthesis, but the requested migration asks for one canonical convention across the repository.
- Consequences:
  - users can move between runtime simulation and synthesis with the correct magnitude but the wrong sign or factor unless they know the hidden mapping
  - docs must currently explain a permanent mismatch instead of one shared convention

### 5. Documentation and examples currently mix physical-detuning clarity with old Hamiltonian semantics

- What: Some recent example/test updates already separate physical transition detuning from raw `Pulse.carrier`, but README/API/physics docs still describe the old runtime `chi` meaning.
- Where:
  - `README.md`
  - `API_REFERENCE.md`
  - `physics_and_conventions/physics_conventions_report.tex`
  - spectroscopy examples and diagnostics
- Affected components:
  - user-facing docs
  - example labels and guide-line expectations
  - convention-audit narrative
- Why inconsistent: Carrier-axis clarity was improved without completing the Hamiltonian-sign migration.
- Consequences:
  - the codebase can now describe the x-axis correctly while still attributing the wrong physical meaning to `chi`
  - readers can see two different stories depending on whether they inspect the example or the model implementation

## Suspected or follow-up items

### A. Readout-chain dispersive sign may need a coordinated migration decision

- What: `ReadoutResonator.dispersive_shift()` currently returns `0` for `|g>` and `-chi` for `|e>`.
- Where:
  - `cqed_sim/experiment/readout_chain.py`
- Affected components:
  - resonator-response helpers
  - Purcell and measurement-rate examples/tests
- Resolution on 2026-03-13:
  - `cqed_sim/measurement/readout_chain.py` now uses the same canonical sign language as the runtime Hamiltonian: the excited state shifts the resonator by `+chi` relative to `|g>`.
  - This follow-up item is resolved rather than left open.

### B. Example artifacts and study scripts likely encode old sign assumptions outside the main test path

- What: Several study/example files compute conditional detunings directly from `st_chi_hz`, `st_chi2_hz`, or hand-written formulas.
- Where:
  - `examples/`
  - calibration/study helpers
- Affected components:
  - saved JSON/PNG artifacts
  - study-specific reports
- Open question: Some of these are smoke-test assets or archived studies and may need either migration or a historical note, depending on whether they are intended to remain active user-facing references.

## Resolution summary

- Runtime dispersive and Kerr semantics are now centralized in `UniversalCQEDModel` and inherited by the compatibility wrappers in `cqed_sim/core/model.py` and `cqed_sim/core/readout_model.py`.
- `cqed_sim/analysis/parameter_translation.py` now exposes the same canonical `chi` and `chi_2` values for runtime and synthesis callers instead of a separate sign-converted mapping.
- `cqed_sim/unitary_synthesis/sequence.py`, `README.md`, `API_REFERENCE.md`, and the physics documentation now describe the same projector-based `+chi * n * |e><e|` and `+(K/2) n(n-1)` conventions.
- The readout-chain follow-up item was re-audited and confirmed aligned with the canonical sign convention.
