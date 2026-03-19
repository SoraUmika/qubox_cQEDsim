"""Tests for cqed_sim.io.gates — gate sequence I/O and validation."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from cqed_sim.io.gates import (
    DisplacementGate,
    RotationGate,
    SQRGate,
    gate_path_candidates,
    gate_to_record,
    gate_summary_text,
    validate_gate_entry,
    load_gate_sequence,
    render_gate_table,
)

# ---- helpers ----


def _write_json(data, path):
    path.write_text(json.dumps(data), encoding="utf-8")


def _displacement_entry(index=0):
    return {
        "index": index,
        "type": "Displacement",
        "target": "storage",
        "name": f"D_{index}",
        "params": {"re": 1.0, "im": 0.5},
    }


def _rotation_entry(index=0):
    return {
        "index": index,
        "type": "Rotation",
        "target": "qubit",
        "name": f"R_{index}",
        "params": {"theta": 1.5708, "phi": 0.0},
    }


def _sqr_entry(index=0, n=4):
    return {
        "index": index,
        "type": "SQR",
        "target": "qubit",
        "name": f"SQR_{index}",
        "params": {"theta": [1.5708] * n, "phi": [0.0] * n},
    }


# ---- load_gate_sequence ----


class TestLoadGateSequence:
    def test_loads_displacement_gate(self, tmp_path):
        p = tmp_path / "gates.json"
        _write_json([_displacement_entry()], p)
        _, gates = load_gate_sequence(p)
        assert len(gates) == 1
        assert isinstance(gates[0], DisplacementGate)
        assert gates[0].alpha == complex(1.0, 0.5)

    def test_loads_rotation_gate(self, tmp_path):
        p = tmp_path / "gates.json"
        _write_json([_rotation_entry()], p)
        _, gates = load_gate_sequence(p)
        assert isinstance(gates[0], RotationGate)
        assert abs(gates[0].theta - 1.5708) < 1e-10

    def test_loads_sqr_gate(self, tmp_path):
        p = tmp_path / "gates.json"
        _write_json([_sqr_entry(n=3)], p)
        _, gates = load_gate_sequence(p)
        assert isinstance(gates[0], SQRGate)
        assert len(gates[0].theta) == 3

    def test_loads_mixed_sequence(self, tmp_path):
        p = tmp_path / "gates.json"
        _write_json([_rotation_entry(0), _displacement_entry(1), _sqr_entry(2)], p)
        _, gates = load_gate_sequence(p)
        assert len(gates) == 3
        assert isinstance(gates[0], RotationGate)
        assert isinstance(gates[1], DisplacementGate)
        assert isinstance(gates[2], SQRGate)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_gate_sequence(tmp_path / "nonexistent.json")

    def test_not_a_list_raises(self, tmp_path):
        p = tmp_path / "gates.json"
        _write_json({"key": "value"}, p)
        with pytest.raises(TypeError):
            load_gate_sequence(p)

    def test_missing_required_key_raises(self, tmp_path):
        p = tmp_path / "gates.json"
        # Missing 'target' and 'params' — validate_gate_entry will raise KeyError
        _write_json([{"index": 0, "type": "Rotation"}], p)
        with pytest.raises(KeyError):
            load_gate_sequence(p)

    def test_unsupported_gate_type_raises(self, tmp_path):
        p = tmp_path / "gates.json"
        entry = {
            "index": 0,
            "type": "SNAP",
            "target": "storage",
            "params": {"phases": [0.0]},
        }
        _write_json([entry], p)
        with pytest.raises(ValueError):
            load_gate_sequence(p)

    def test_returns_chosen_path(self, tmp_path):
        p = tmp_path / "gates.json"
        _write_json([_rotation_entry()], p)
        chosen, _ = load_gate_sequence(p)
        assert chosen == p


# ---- validate_gate_entry ----


class TestValidateGateEntry:
    def test_validates_displacement(self):
        gate = validate_gate_entry(_displacement_entry(), 0)
        assert isinstance(gate, DisplacementGate)

    def test_validates_rotation(self):
        gate = validate_gate_entry(_rotation_entry(), 0)
        assert isinstance(gate, RotationGate)

    def test_validates_sqr(self):
        gate = validate_gate_entry(_sqr_entry(), 0)
        assert isinstance(gate, SQRGate)

    def test_wrong_target_raises(self):
        entry = _displacement_entry()
        entry["target"] = "qubit"  # Displacement must target storage
        with pytest.raises(ValueError, match="storage"):
            validate_gate_entry(entry, 0)

    def test_non_numeric_field_raises(self):
        entry = _rotation_entry()
        entry["params"]["theta"] = "not_a_number"
        with pytest.raises(TypeError):
            validate_gate_entry(entry, 0)


# ---- gate_summary_text ----


class TestGateSummaryText:
    def test_displacement_summary(self):
        gate = DisplacementGate(index=0, name="D", re=1.0, im=0.5)
        text = gate_summary_text(gate)
        assert "alpha" in text

    def test_rotation_summary(self):
        gate = RotationGate(index=0, name="R", theta=1.5708, phi=0.0)
        text = gate_summary_text(gate)
        assert "theta" in text
        assert "phi" in text

    def test_sqr_summary_active_count(self):
        import numpy as np

        gate = SQRGate(
            index=0, name="SQR", theta=(np.pi, 0.0, np.pi, 0.0), phi=(0.0,) * 4
        )
        text = gate_summary_text(gate)
        assert "active=2" in text


# ---- gate_to_record ----


class TestGateToRecord:
    def test_displacement_roundtrip(self):
        gate = DisplacementGate(index=0, name="D", re=1.0, im=0.5)
        record = gate_to_record(gate)
        assert record["type"] == "Displacement"
        assert record["target"] == "storage"
        assert record["params"]["re"] == 1.0

    def test_rotation_roundtrip(self):
        gate = RotationGate(index=0, name="R", theta=1.5708, phi=0.0)
        record = gate_to_record(gate)
        assert record["type"] == "Rotation"


# ---- render_gate_table (smoke) ----


def test_render_gate_table_smoke(capsys):
    gates = [
        RotationGate(0, "R", 1.5708, 0.0),
        DisplacementGate(1, "D", 1.0, 0.0),
    ]
    render_gate_table(gates)
    captured = capsys.readouterr()
    assert "Rotation" in captured.out
    assert "Displacement" in captured.out


# ---- gate_path_candidates ----


class TestGatePathCandidates:
    def test_json_extension_includes_josn_fallback(self):
        paths = gate_path_candidates("my_gates.json")
        suffixes = [p.suffix for p in paths]
        assert ".json" in suffixes
        assert ".josn" in suffixes

    def test_josn_typo_includes_json_fallback(self):
        paths = gate_path_candidates("my_gates.josn")
        suffixes = [p.suffix for p in paths]
        assert ".json" in suffixes
        assert ".josn" in suffixes

    def test_no_extension_appends_json_and_josn(self):
        paths = gate_path_candidates("my_gates")
        suffixes = [p.suffix for p in paths]
        assert ".json" in suffixes
        assert ".josn" in suffixes
        assert "" in suffixes

    def test_josn_typo_loads_json_file(self, tmp_path):
        """Requesting .josn should fall back to .json if .json exists."""
        p = tmp_path / "gates.json"
        _write_json([_rotation_entry()], p)
        chosen, gates = load_gate_sequence(tmp_path / "gates.josn")
        assert chosen == p
        assert len(gates) == 1


# ---- SQR alternate key names ----


class TestSQRAlternateKeys:
    def test_sqr_with_thetas_and_phis_keys(self):
        entry = {
            "type": "SQR",
            "target": "qubit",
            "params": {"thetas": [1.0, 2.0], "phis": [0.0, 0.5]},
        }
        gate = validate_gate_entry(entry, 0)
        assert isinstance(gate, SQRGate)
        assert len(gate.theta) == 2

    def test_sqr_missing_theta_raises(self):
        entry = {
            "type": "SQR",
            "target": "qubit",
            "params": {"bad_key": [1.0]},
        }
        with pytest.raises(KeyError, match="theta"):
            validate_gate_entry(entry, 0)


# ---- params validation ----


class TestParamsValidation:
    def test_params_not_dict_raises(self):
        entry = {"type": "Rotation", "target": "qubit", "params": [1.0, 2.0]}
        with pytest.raises(TypeError, match="dictionary"):
            validate_gate_entry(entry, 0)

    def test_entry_not_dict_raises(self):
        with pytest.raises(TypeError, match="dictionary"):
            validate_gate_entry("not_a_dict", 0)

    def test_rotation_wrong_target_raises(self):
        entry = _rotation_entry()
        entry["target"] = "storage"
        with pytest.raises(ValueError, match="qubit"):
            validate_gate_entry(entry, 0)
