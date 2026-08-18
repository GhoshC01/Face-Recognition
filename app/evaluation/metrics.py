from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

REJECT_LABEL = "<reject>"
IMPOSTOR_LABEL = "<impostor>"


@dataclass
class ProbeRecord:
    """One evaluated probe, independent of any particular threshold. The raw
    top-1 gallery match and its similarity score are computed once by
    actually running the production pipeline; PASS/FAIL and correctness are
    derived per-threshold afterwards in `evaluate_at_threshold` without
    re-running any recognition.
    """

    probe_path: str
    ground_truth_id: str | None  # None => impostor / not enrolled anywhere
    predicted_id: str | None  # top-1 gallery match, or None if no candidate exists
    similarity: float | None  # None if the pipeline failed on this probe entirely
    valid: bool
    rejection_reason: str | None = None


@dataclass
class ThresholdMetrics:
    """Accuracy is NOT the similarity score -- it is the fraction of labeled
    probes the pipeline got right at a given threshold. This is computed
    using an open-set identification model with a reject option:

    - TP: a genuine probe, accepted, matched to its own correct identity.
    - FP: accepted but wrong -- either a genuine probe matched to the WRONG
      identity (a substitution error) or an impostor probe matched to ANY
      identity at all (a false accept).
    - FN: a genuine probe not accepted (below threshold, or the pipeline
      couldn't even produce a usable embedding for it).
    - TN: an impostor probe correctly rejected (matched to no one).

    FAR is restricted to impostor probes specifically -- the strict
    biometric definition of "rate at which non-enrolled people are wrongly
    accepted." Substitution errors (a genuine person accepted as someone
    else) are tracked separately since FAR alone doesn't capture that
    distinct, also-important failure mode.
    """

    threshold: float
    total: int
    correct: int
    incorrect: int
    rejected: int
    accuracy: float
    precision: float
    recall: float
    far: float
    frr: float
    substitution_error_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)


def evaluate_at_threshold(records: list[ProbeRecord], threshold: float) -> ThresholdMetrics:
    total = len(records)
    correct = incorrect = rejected = 0
    tp = fp = fn = tn = 0
    substitution_errors = 0
    genuine_total = 0
    impostor_total = 0
    confusion: dict[str, dict[str, int]] = {}

    def _record_confusion(row: str, col: str) -> None:
        confusion.setdefault(row, {}).setdefault(col, 0)
        confusion[row][col] += 1

    for record in records:
        is_genuine = record.ground_truth_id is not None
        genuine_total += 1 if is_genuine else 0
        impostor_total += 0 if is_genuine else 1

        # `predicted_id is not None` is a defensive redundancy: the real
        # pipeline only ever produces (similarity, predicted_id) together
        # from the same search call, but this guards against inconsistent
        # data regardless -- there is no "accept" without an identity to
        # accept as.
        accepted = (
            record.valid
            and record.similarity is not None
            and record.predicted_id is not None
            and record.similarity >= threshold
        )
        predicted_label = record.predicted_id if accepted else None

        truth_row = record.ground_truth_id if is_genuine else IMPOSTOR_LABEL
        predicted_col = predicted_label if predicted_label is not None else REJECT_LABEL
        _record_confusion(truth_row, predicted_col)

        if is_genuine:
            if accepted and predicted_label == record.ground_truth_id:
                correct += 1
                tp += 1
            elif accepted:
                incorrect += 1
                fp += 1
                substitution_errors += 1
            else:
                rejected += 1
                fn += 1
        else:
            if accepted:
                incorrect += 1
                fp += 1
            else:
                correct += 1
                tn += 1

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    frr = fn / genuine_total if genuine_total else 0.0
    far = (fp - substitution_errors) / impostor_total if impostor_total else 0.0
    substitution_error_rate = substitution_errors / genuine_total if genuine_total else 0.0

    return ThresholdMetrics(
        threshold=threshold,
        total=total,
        correct=correct,
        incorrect=incorrect,
        rejected=rejected,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        far=far,
        frr=frr,
        substitution_error_rate=substitution_error_rate,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        confusion_matrix=confusion,
    )


def select_best_threshold(
    records: list[ProbeRecord],
    thresholds: list[float],
    objective: Literal["accuracy", "f1", "min_far_frr_gap"] = "accuracy",
) -> tuple[float, ThresholdMetrics, list[ThresholdMetrics]]:
    """Evaluates every candidate threshold against the measured probe
    results and returns whichever one best satisfies `objective` -- a
    threshold selected from data, never a guessed constant. Returns the full
    swept metrics too, so the tradeoff curve (not just the winner) is
    available for inspection.
    """
    if not thresholds:
        raise ValueError("At least one candidate threshold is required")

    sweep = [evaluate_at_threshold(records, t) for t in sorted(set(thresholds))]

    def _score(metrics: ThresholdMetrics) -> float:
        if objective == "accuracy":
            return metrics.accuracy
        if objective == "f1":
            denom = metrics.precision + metrics.recall
            return (2 * metrics.precision * metrics.recall / denom) if denom else 0.0
        if objective == "min_far_frr_gap":
            return -abs(metrics.far - metrics.frr)
        raise ValueError(f"Unknown objective: {objective!r}")

    best = max(sweep, key=_score)
    return best.threshold, best, sweep
