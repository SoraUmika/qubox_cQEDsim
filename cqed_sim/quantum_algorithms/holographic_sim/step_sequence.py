from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .channel import HolographicChannel
from .step_unitary import StepUnitarySpec
from .utils import json_ready


@dataclass(frozen=True)
class HolographicChannelSequence:
    """Validated finite holographic step sequence.

    All steps share the same `physical_dim` and `bond_dim`, but each step may
    carry a different joint unitary or Kraus realization.
    """

    channels: Sequence[HolographicChannel]
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = tuple(self.channels)
        if len(normalized) == 0:
            raise ValueError("HolographicChannelSequence requires at least one channel.")
        first = normalized[0]
        for channel in normalized[1:]:
            if channel.physical_dim != first.physical_dim:
                raise ValueError("All sequence channels must have the same physical_dim.")
            if channel.bond_dim != first.bond_dim:
                raise ValueError("All sequence channels must have the same bond_dim.")
        object.__setattr__(self, "channels", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_steps(self) -> int:
        return len(self.channels)

    @property
    def physical_dim(self) -> int:
        return self.channels[0].physical_dim

    @property
    def bond_dim(self) -> int:
        return self.channels[0].bond_dim

    def channel_for_step(self, step: int) -> HolographicChannel:
        step_index = int(step)
        if step_index <= 0 or step_index > self.num_steps:
            raise ValueError(f"step={step_index} is outside [1, {self.num_steps}].")
        return self.channels[step_index - 1]

    @classmethod
    def from_unitaries(
        cls,
        unitaries: Sequence[Any],
        *,
        physical_dim: int,
        bond_dim: int,
        reference_state: Any | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "HolographicChannelSequence":
        resolved_channels: list[HolographicChannel] = []
        for step_index, item in enumerate(unitaries, start=1):
            spec = item if isinstance(item, StepUnitarySpec) else StepUnitarySpec(item)
            channel_metadata = {"sequence_step": int(step_index), "acts_on": spec.acts_on}
            if spec.metadata:
                channel_metadata["unitary_metadata"] = dict(spec.metadata)
            channel = HolographicChannel.from_unitary(
                spec,
                physical_dim=physical_dim,
                bond_dim=bond_dim,
                reference_state=reference_state,
                label=spec.label if spec.label is not None else f"step_{step_index}",
                metadata=channel_metadata,
            )
            resolved_channels.append(channel)
        return cls(
            channels=tuple(resolved_channels),
            label=label,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_mps_state(
        cls,
        state: Any,
        *,
        complete: bool = True,
        chi_max: int | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "HolographicChannelSequence":
        from .mps import MatrixProductState

        mps = state if isinstance(state, MatrixProductState) else MatrixProductState(state)
        needs_right_canonical = mps.tensors is None
        needs_completion = bool(complete and mps.uniform_tensors is None)
        if needs_right_canonical or needs_completion:
            mps.make_right_canonical(cast_complete=complete, chi_max=chi_max)
        resolved_metadata = {"source": "MatrixProductState", "num_sites": int(mps.num_sites)}
        if chi_max is not None:
            resolved_metadata["chi_max"] = int(chi_max)
        if metadata is not None:
            resolved_metadata.update(dict(metadata))
        return cls(
            channels=tuple(
                mps.to_holographic_channel(
                    site=site,
                    complete=complete,
                    label=f"mps_site_{site}",
                )
                for site in range(mps.num_sites)
            ),
            label=label,
            metadata=resolved_metadata,
        )

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "label": self.label,
                "num_steps": int(self.num_steps),
                "physical_dim": int(self.physical_dim),
                "bond_dim": int(self.bond_dim),
                "channels": [channel.to_record() for channel in self.channels],
                "metadata": dict(self.metadata),
            }
        )