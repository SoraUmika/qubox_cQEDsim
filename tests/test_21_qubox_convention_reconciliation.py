from __future__ import annotations

import numpy as np

import examples.audits.sqr_convention_metric_audit as audit


def test_qubox_rotation_and_sqr_sign_mapping():
    scan = audit._sign_scan_for_equivalence()
    best_rot = scan["best_rotation_match"]
    best_sqr = scan["best_sqr_match"]

    # qubox: exp(+i phi_eff) exp(+i omega t)
    assert best_rot["phase_sign"] == 1
    assert best_rot["omega_sign"] == 1
    assert best_rot["process_fidelity_vs_qubox"] > 0.999999

    # unified SQR path now matches qubox with same omega sign semantics as Rotation.
    assert best_sqr["phase_sign"] == 1
    assert best_sqr["omega_sign"] == 1
    assert best_sqr["process_fidelity_vs_qubox"] > 0.999999


def test_qubox_detuning_sign_vs_rotation_and_sqr():
    det = audit._detuning_sign_check()
    rot = det["rotation_like"]
    sqr = det["sqr_like"]

    assert np.sign(rot["+delta_axis_z"]) == -np.sign(rot["-delta_axis_z"])
    assert np.sign(sqr["+delta_axis_z"]) == -np.sign(sqr["-delta_axis_z"])
    assert sqr["relative_to_rotation_sign"] == "same"
