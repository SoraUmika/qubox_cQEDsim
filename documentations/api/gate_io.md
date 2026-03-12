# API Reference — Gate I/O (`cqed_sim.io`)

The I/O module handles loading and serializing gate sequences from JSON files.

---

## Gate Dataclasses

**Module path:** `cqed_sim.io.gates`

```python
@dataclass(frozen=True)
class DisplacementGate:
    index: int; name: str; re: float; im: float
    # Properties: type="Displacement", target="storage", alpha=complex(re, im)

@dataclass(frozen=True)
class RotationGate:
    index: int; name: str; theta: float; phi: float
    # Properties: type="Rotation", target="qubit"

@dataclass(frozen=True)
class SQRGate:
    index: int; name: str; theta: tuple[float, ...]; phi: tuple[float, ...]
    # Properties: type="SQR", target="qubit"
```

**Union type:** `Gate = DisplacementGate | RotationGate | SQRGate`

---

## Functions

| Function | Signature | Description |
|---|---|---|
| `load_gate_sequence(path_like)` | `(str \| Path) -> tuple[Path, list[Gate]]` | Load and validate JSON gate sequence. Typo-tolerant for `.json`/`.josn` extensions. |
| `render_gate_table(gates, max_rows=20)` | `(list[Gate], int) -> None` | Print formatted ASCII gate table |
| `gate_to_record(gate)` | `(Gate) -> dict` | Convert gate to dict record |
| `gate_summary_text(gate)` | `(Gate) -> str` | One-line summary of gate params |

---

## JSON Format

Array of objects, each with:

```json
[
  {
    "type": "Displacement",
    "target": "storage",
    "name": "D_0",
    "params": {"re": 1.5, "im": 0.3}
  },
  {
    "type": "Rotation",
    "target": "qubit",
    "name": "R_0",
    "params": {"theta": 3.14159, "phi": 0.0}
  },
  {
    "type": "SQR",
    "target": "qubit",
    "name": "SQR_0",
    "params": {"theta": [0.0, 1.57, 3.14], "phi": [0.0, 0.0, 0.0]}
  }
]
```

SQR gates accept both `"theta"`/`"thetas"` and `"phi"`/`"phis"` keys.
