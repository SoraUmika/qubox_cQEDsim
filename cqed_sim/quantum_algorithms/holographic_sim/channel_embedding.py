from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .channel import HolographicChannel
from .observables import PhysicalObservable, as_observable
from .results import MeasurementOutcome
from .utils import conditional_bond_state, partial_trace_joint


@dataclass
class PurifiedChannelStep:
    """The prepare-apply-measure-reset primitive for one holographic step."""

    channel: HolographicChannel

    def propagate(self, bond_state: Any) -> np.ndarray:
        return self.channel.apply(bond_state)

    def sample_measurement(
        self,
        bond_state: Any,
        observable: PhysicalObservable | Any,
        *,
        rng: np.random.Generator | None = None,
        atol: float = 1.0e-15,
    ) -> MeasurementOutcome:
        branches = self.enumerate_measurement_branches(bond_state, observable, atol=atol)
        if not branches:
            raise ValueError("No nonzero-probability measurement branches were found.")
        probabilities = np.asarray([branch.probability for branch in branches], dtype=float)
        probabilities = probabilities / probabilities.sum()
        generator = np.random.default_rng() if rng is None else rng
        idx = int(generator.choice(len(branches), p=probabilities))
        return branches[idx]

    def enumerate_measurement_branches(
        self,
        bond_state: Any,
        observable: PhysicalObservable | Any,
        *,
        atol: float = 1.0e-15,
    ) -> list[MeasurementOutcome]:
        obs = as_observable(observable)
        if obs.dim != self.channel.physical_dim:
            raise ValueError(
                f"Observable dimension {obs.dim} does not match the physical dimension {self.channel.physical_dim}."
            )
        eigenvalues, eigenvectors = obs.eigendecomposition()
        joint = self.channel.joint_output_state(bond_state)
        outcomes: list[MeasurementOutcome] = []
        for idx, eigval in enumerate(eigenvalues):
            vec = eigenvectors[:, idx]
            projector = np.outer(vec, vec.conj())
            next_state, probability = conditional_bond_state(
                joint,
                projector,
                physical_dim=self.channel.physical_dim,
                bond_dim=self.channel.bond_dim,
                atol=atol,
            )
            if next_state is None or probability <= atol:
                continue
            outcomes.append(
                MeasurementOutcome(
                    eigenvalue=complex(eigval),
                    probability=float(probability),
                    bond_state=next_state,
                    outcome_index=idx,
                )
            )
        return outcomes

    def trace_out(self, bond_state: Any) -> MeasurementOutcome:
        joint = self.channel.joint_output_state(bond_state)
        next_state = partial_trace_joint(
            joint,
            physical_dim=self.channel.physical_dim,
            bond_dim=self.channel.bond_dim,
            trace_out="physical",
        )
        return MeasurementOutcome(eigenvalue=None, probability=1.0, bond_state=next_state, outcome_index=None)
