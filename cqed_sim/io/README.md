# `cqed_sim.io` — Gate Sequence I/O

## What this module does

`cqed_sim.io` provides gate sequence deserialization for reading externally-produced
calibration gate files. It bridges external calibration tooling (which writes gate JSON
files) to the waveform bridge and unitary synthesis layers inside `cqed_sim`.

## Current scope: read-only

This module currently supports **deserialization only**. There is no save or serialize
path. To convert a gate back to a plain dict for external tooling, use
`gate_to_record(gate)`, but there is no function that writes a gate list to disk.

## Main functions

| Function | Description |
|---|---|
| `load_gate_sequence(path_like)` | Load a JSON gate file; returns `(chosen_path, [Gate, ...])` |
| `validate_gate_entry(entry, gate_index)` | Validate a single gate dict and return a typed gate object |
| `gate_summary_text(gate)` | Human-readable one-line summary of a gate |
| `render_gate_table(gates)` | Print a formatted table of a gate list |
| `gate_to_record(gate)` | Convert a typed gate object back to a plain dict |

## Gate types

| Type | Target | Key params |
|---|---|---|
| `DisplacementGate` | `"storage"` | `re`, `im` (complex displacement alpha = re + i*im) |
| `RotationGate` | `"qubit"` | `theta` (rotation angle), `phi` (axis angle) |
| `SQRGate` | `"qubit"` | `theta` (per-Fock angles), `phi` (per-Fock phases) |

## When to use

Use `load_gate_sequence(path)` when loading SQR, Rotation, or Displacement gate
sequences from JSON files produced by calibration workflows. The returned gate objects
are typed dataclasses that feed directly into pulse builders such as
`build_rotation_pulse(gate, ...)`, `build_displacement_pulse(gate, ...)`, and
`build_sqr_multitone_pulse(gate, ...)`.

## Format

Gate JSON files must be a list of gate dictionaries, each with:
- `"type"`: `"Displacement"`, `"Rotation"`, or `"SQR"`
- `"target"`: `"storage"` (Displacement) or `"qubit"` (Rotation, SQR)
- `"params"`: dict of numeric parameters as described above
- `"name"` (optional): human-readable label
- `"index"` (optional): integer position in the sequence

## Limitations

- Only JSON format is supported; no HDF5, CSV, or binary formats.
- Only three gate types are supported: Displacement, Rotation, SQR.
- No save/serialize path exists for writing gate lists to disk.

## Relationship to the rest of `cqed_sim`

`cqed_sim.io` sits at the boundary between external calibration tooling and the
pulse-level simulation stack. The typical workflow is:

```
external calibration tool → JSON gate file → load_gate_sequence() → gate objects
  → build_rotation_pulse() / build_displacement_pulse() / build_sqr_multitone_pulse()
  → SequenceCompiler → simulate_sequence()
```

For ideal (unitary) gate operators, see `cqed_sim.core.ideal_gates`. For gate synthesis,
see `cqed_sim.unitary_synthesis`.
