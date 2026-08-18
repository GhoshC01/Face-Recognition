from __future__ import annotations

import pytest

from app.evaluation.metrics import (
    IMPOSTOR_LABEL,
    REJECT_LABEL,
    ProbeRecord,
    evaluate_at_threshold,
    select_best_threshold,
)

THRESHOLD = 0.5


def _records() -> list[ProbeRecord]:
    return [
        # genuine A: both correctly matched -> 2 true positives
        ProbeRecord("p1", ground_truth_id="A", predicted_id="A", similarity=0.9, valid=True),
        ProbeRecord("p2", ground_truth_id="A", predicted_id="A", similarity=0.8, valid=True),
        # genuine B: correctly matched -> true positive
        ProbeRecord("p3", ground_truth_id="B", predicted_id="B", similarity=0.7, valid=True),
        # genuine B: matched to the WRONG identity -> substitution error (a kind of false positive)
        ProbeRecord("p4", ground_truth_id="B", predicted_id="C", similarity=0.6, valid=True),
        # genuine C: below threshold -> false negative (false reject)
        ProbeRecord("p5", ground_truth_id="C", predicted_id="C", similarity=0.3, valid=True),
        # genuine C: pipeline failed entirely (e.g. no face) -> also a false negative
        ProbeRecord("p6", ground_truth_id="C", predicted_id=None, similarity=None, valid=False, rejection_reason="no_face_detected"),
        # impostor: falsely accepted as A -> false positive (this is the strict FAR case)
        ProbeRecord("p7", ground_truth_id=None, predicted_id="A", similarity=0.55, valid=True),
        # impostor: no gallery match at all -> true negative
        ProbeRecord("p8", ground_truth_id=None, predicted_id=None, similarity=None, valid=True),
    ]


def test_metrics_match_hand_computed_example():
    metrics = evaluate_at_threshold(_records(), THRESHOLD)

    assert metrics.total == 8
    assert metrics.correct == 4  # 3 genuine TP + 1 impostor TN
    assert metrics.incorrect == 2  # 1 substitution + 1 impostor false-accept
    assert metrics.rejected == 2  # 2 genuine FN (one below threshold, one invalid)
    assert metrics.true_positives == 3
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 2
    assert metrics.true_negatives == 1

    assert metrics.accuracy == pytest.approx(4 / 8)
    assert metrics.precision == pytest.approx(3 / 5)
    assert metrics.recall == pytest.approx(3 / 5)
    assert metrics.frr == pytest.approx(2 / 6)  # 2 FN over 6 genuine probes
    assert metrics.far == pytest.approx(1 / 2)  # 1 impostor false-accept over 2 impostor probes
    assert metrics.substitution_error_rate == pytest.approx(1 / 6)


def test_confusion_matrix_is_appropriate_for_open_set_identification():
    metrics = evaluate_at_threshold(_records(), THRESHOLD)
    cm = metrics.confusion_matrix

    assert cm["A"] == {"A": 2}
    assert cm["B"] == {"B": 1, "C": 1}
    assert cm["C"] == {REJECT_LABEL: 2}
    assert cm[IMPOSTOR_LABEL] == {"A": 1, REJECT_LABEL: 1}


def test_similarity_alone_is_not_accuracy():
    """A record can have a high raw similarity yet still be an incorrect
    (wrong-identity) match -- accuracy must come from comparing the
    prediction to ground truth, not from the similarity score itself."""
    records = [
        ProbeRecord("p1", ground_truth_id="A", predicted_id="B", similarity=0.99, valid=True),
    ]
    metrics = evaluate_at_threshold(records, threshold=0.5)

    assert metrics.correct == 0
    assert metrics.incorrect == 1
    assert metrics.accuracy == 0.0


def test_higher_threshold_increases_rejections_and_can_change_accuracy():
    records = _records()

    lenient = evaluate_at_threshold(records, threshold=0.1)  # almost everything clears the bar
    strict = evaluate_at_threshold(records, threshold=0.95)  # almost nothing does

    assert lenient.correct == 5  # 4 genuine TP (p1,p2,p3,p5) + 1 impostor TN (p8)
    assert lenient.rejected == 1  # only the invalid probe (p6) is unconditionally rejected
    assert strict.correct == 2  # 0 genuine TP + 2 impostor TN (p7, p8 both now rejected)
    assert strict.rejected == 6  # every genuine probe now fails to clear the threshold
    assert strict.rejected > lenient.rejected
    assert strict.correct < lenient.correct


def test_select_best_threshold_picks_the_measured_optimum_by_accuracy():
    records = _records()

    best_threshold, best_metrics, sweep = select_best_threshold(records, [0.2, 0.5, 0.9], objective="accuracy")

    assert len(sweep) == 3
    assert best_metrics.accuracy == max(m.accuracy for m in sweep)
    assert best_threshold == best_metrics.threshold


def test_select_best_threshold_requires_at_least_one_candidate():
    with pytest.raises(ValueError):
        select_best_threshold(_records(), [], objective="accuracy")


def test_select_best_threshold_rejects_unknown_objective():
    with pytest.raises(ValueError):
        select_best_threshold(_records(), [0.5], objective="not_a_real_objective")


def test_empty_records_do_not_crash():
    metrics = evaluate_at_threshold([], threshold=0.5)

    assert metrics.total == 0
    assert metrics.accuracy == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.far == 0.0
    assert metrics.frr == 0.0
