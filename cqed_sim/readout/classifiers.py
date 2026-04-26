from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class TimeResolvedClassifier(Protocol):
    labels: tuple[int, ...]

    def classify(self, records: np.ndarray) -> np.ndarray:
        ...


def confusion_matrix(
    predicted: Sequence[int],
    prepared: Sequence[int],
    *,
    labels: Sequence[int] = (0, 1),
) -> np.ndarray:
    labels = tuple(int(label) for label in labels)
    index = {label: idx for idx, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=float)
    for pred, prep in zip(predicted, prepared):
        if int(prep) in index and int(pred) in index:
            counts[index[int(pred)], index[int(prep)]] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = counts / np.maximum(np.sum(counts, axis=0, keepdims=True), 1.0)
    return np.asarray(norm, dtype=float)


@dataclass
class MatchedFilterClassifier:
    templates: dict[int, np.ndarray]
    labels: tuple[int, ...]

    @classmethod
    def fit(cls, mean_records_by_label: dict[int, Sequence[complex]]) -> "MatchedFilterClassifier":
        templates = {int(label): np.asarray(record, dtype=np.complex128).reshape(-1) for label, record in mean_records_by_label.items()}
        if not templates:
            raise ValueError("At least one template is required.")
        sizes = {record.size for record in templates.values()}
        if len(sizes) != 1:
            raise ValueError("All templates must have the same length.")
        return cls(templates=templates, labels=tuple(sorted(templates)))

    def scores(self, records: np.ndarray) -> np.ndarray:
        data = np.asarray(records, dtype=np.complex128)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        out = np.empty((data.shape[0], len(self.labels)), dtype=float)
        for col, label in enumerate(self.labels):
            template = self.templates[int(label)]
            if template.size != data.shape[1]:
                raise ValueError("Record length does not match the fitted template length.")
            out[:, col] = np.real(data @ np.conjugate(template)) - 0.5 * float(np.vdot(template, template).real)
        return out

    def classify(self, records: np.ndarray) -> np.ndarray:
        score = self.scores(records)
        return np.asarray([self.labels[int(idx)] for idx in np.argmax(score, axis=1)], dtype=int)

    def confusion(self, records: np.ndarray, prepared: Sequence[int]) -> np.ndarray:
        return confusion_matrix(self.classify(records), prepared, labels=self.labels)


@dataclass
class GaussianMLClassifier:
    means: dict[int, np.ndarray]
    covariances: dict[int, np.ndarray]
    priors: dict[int, float]
    labels: tuple[int, ...]
    regularization: float = 1.0e-9

    @classmethod
    def fit(
        cls,
        samples_by_label: dict[int, np.ndarray],
        *,
        priors: dict[int, float] | None = None,
        regularization: float = 1.0e-9,
    ) -> "GaussianMLClassifier":
        means: dict[int, np.ndarray] = {}
        covs: dict[int, np.ndarray] = {}
        labels = tuple(sorted(int(label) for label in samples_by_label))
        for label in labels:
            data = np.asarray(samples_by_label[label], dtype=float)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.shape[0] < 1:
                raise ValueError("Each label must contain at least one sample.")
            means[label] = np.mean(data, axis=0)
            centered = data - means[label][None, :]
            if data.shape[0] <= 1:
                cov = np.eye(data.shape[1], dtype=float) * float(regularization)
            else:
                cov = centered.T @ centered / float(data.shape[0] - 1)
                cov += np.eye(data.shape[1], dtype=float) * float(regularization)
            covs[label] = cov
        if priors is None:
            prior = {label: 1.0 / len(labels) for label in labels}
        else:
            total = sum(float(priors.get(label, 0.0)) for label in labels)
            if total <= 0.0:
                raise ValueError("Prior weights must contain positive total mass.")
            prior = {label: float(priors.get(label, 0.0)) / total for label in labels}
        return cls(means=means, covariances=covs, priors=prior, labels=labels, regularization=float(regularization))

    def log_likelihoods(self, samples: np.ndarray) -> np.ndarray:
        data = np.asarray(samples, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        out = np.empty((data.shape[0], len(self.labels)), dtype=float)
        for col, label in enumerate(self.labels):
            mean = self.means[label]
            cov = self.covariances[label]
            diff = data - mean[None, :]
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0.0:
                cov = cov + np.eye(cov.shape[0]) * self.regularization
                sign, logdet = np.linalg.slogdet(cov)
            inv = np.linalg.pinv(cov)
            quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
            out[:, col] = -0.5 * (quad + logdet + data.shape[1] * np.log(2.0 * np.pi)) + np.log(max(self.priors[label], 1.0e-300))
        return out

    def classify(self, samples: np.ndarray) -> np.ndarray:
        ll = self.log_likelihoods(samples)
        return np.asarray([self.labels[int(idx)] for idx in np.argmax(ll, axis=1)], dtype=int)

    def confusion(self, samples: np.ndarray, prepared: Sequence[int]) -> np.ndarray:
        return confusion_matrix(self.classify(samples), prepared, labels=self.labels)


@dataclass
class PathClassifierAdapter:
    classifier: TimeResolvedClassifier

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(int(label) for label in self.classifier.labels)

    def classify(self, records: np.ndarray) -> np.ndarray:
        return self.classifier.classify(records)

    def confusion(self, records: np.ndarray, prepared: Sequence[int]) -> np.ndarray:
        return confusion_matrix(self.classify(records), prepared, labels=self.labels)


__all__ = [
    "GaussianMLClassifier",
    "MatchedFilterClassifier",
    "PathClassifierAdapter",
    "TimeResolvedClassifier",
    "confusion_matrix",
]
