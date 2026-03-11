## Confirmed issues

### 1. `usage_examples.ipynb` mixed raw carrier semantics with physical-detuning plotting

- Where:
  - `usage_examples.ipynb`, bare-spectroscopy section
  - `usage_examples.ipynb`, displacement-conditioned qubit spectroscopy section
  - `usage_examples.ipynb`, AC-Stark / centroid section
- What:
  - The notebook scanned `carrier=MHz(detuning_mhz)` directly while presenting the x-axis as a qubit detuning in the rotating frame.
  - In `cqed_sim`, `Pulse.carrier` is the negative of the rotating-frame transition frequency it resonantly addresses.
- Why inconsistent:
  - The core library and physics documentation already define user-facing spectroscopy predictions in terms of transition frequency, not raw waveform carrier.
  - With negative `chi`, physical qubit transition lines move to lower frequency, but raw carrier resonances move to the opposite sign.
- Consequence:
  - The notebook could appear to disagree with the documented dispersive sign even when the simulator core was correct.

### 2. The selective-spectroscopy markdown still described the old `-chi` wording

- Where:
  - `usage_examples.ipynb`, markdown introducing displacement-conditioned spectroscopy
- What:
  - The text said each photon number shifts the qubit transition by roughly `-chi`.
- Why inconsistent:
  - The canonical repo convention is now `omega_ge(n) = omega_ge(0) + n * chi`.
  - Negative `chi` lowers the transition frequency directly; the shift is not described as `-chi`.
- Consequence:
  - The text could reinforce the same sign confusion as the stale plotting cell.

## Suspected / unresolved issues

- None at this stage. The core Hamiltonian, frame subtraction, and carrier helper path are internally consistent; the inconsistency was confined to the notebook/tutorial layer.
