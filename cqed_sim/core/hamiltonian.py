from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import qutip as qt

from cqed_sim.sim.couplings import cross_kerr, exchange, self_kerr


@dataclass(frozen=True)
class CrossKerrSpec:
    left: str
    right: str
    chi: float


@dataclass(frozen=True)
class SelfKerrSpec:
    mode: str
    kerr: float


@dataclass(frozen=True)
class ExchangeSpec:
    left: str
    right: str
    coupling: float | complex


def coupling_term_key(
    cross_kerr_terms: Sequence[CrossKerrSpec] = (),
    self_kerr_terms: Sequence[SelfKerrSpec] = (),
    exchange_terms: Sequence[ExchangeSpec] = (),
) -> tuple[tuple[str, str, float], tuple[str, float], tuple[str, str, complex]]:
    return (
        tuple((term.left, term.right, float(term.chi)) for term in cross_kerr_terms),
        tuple((term.mode, float(term.kerr)) for term in self_kerr_terms),
        tuple((term.left, term.right, complex(term.coupling)) for term in exchange_terms),
    )


def resolve_operator(operators: Mapping[str, qt.Qobj], label: str) -> qt.Qobj:
    try:
        return operators[label]
    except KeyError as exc:
        available = ", ".join(sorted(operators))
        raise KeyError(f"Operator label '{label}' is not available. Known labels: {available}.") from exc


def additional_coupling_terms(
    operators: Mapping[str, qt.Qobj],
    *,
    cross_kerr_terms: Sequence[CrossKerrSpec] = (),
    self_kerr_terms: Sequence[SelfKerrSpec] = (),
    exchange_terms: Sequence[ExchangeSpec] = (),
) -> list[qt.Qobj]:
    terms: list[qt.Qobj] = []
    for spec in cross_kerr_terms:
        terms.append(cross_kerr(resolve_operator(operators, spec.left), resolve_operator(operators, spec.right), spec.chi))
    for spec in self_kerr_terms:
        terms.append(self_kerr(resolve_operator(operators, spec.mode), spec.kerr))
    for spec in exchange_terms:
        terms.append(exchange(resolve_operator(operators, spec.left), resolve_operator(operators, spec.right), spec.coupling))
    return terms


def assemble_static_hamiltonian(
    base_hamiltonian: qt.Qobj,
    operators: Mapping[str, qt.Qobj],
    *,
    cross_kerr_terms: Sequence[CrossKerrSpec] = (),
    self_kerr_terms: Sequence[SelfKerrSpec] = (),
    exchange_terms: Sequence[ExchangeSpec] = (),
) -> qt.Qobj:
    hamiltonian = base_hamiltonian
    for term in additional_coupling_terms(
        operators,
        cross_kerr_terms=cross_kerr_terms,
        self_kerr_terms=self_kerr_terms,
        exchange_terms=exchange_terms,
    ):
        hamiltonian += term
    return hamiltonian


__all__ = [
    "CrossKerrSpec",
    "SelfKerrSpec",
    "ExchangeSpec",
    "additional_coupling_terms",
    "assemble_static_hamiltonian",
    "coupling_term_key",
]
