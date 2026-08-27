"""DER and JER against hand-computed cases.

Every expected value in this file is derived by hand in the test's own docstring
or comment. That is the only way to test a metric: a test that compares the
implementation to itself, or to a "golden" number produced by the same code,
catches nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from streamdiar.metrics.assignment import greedy_assignment, linear_sum_assignment
from streamdiar.metrics.der import (
    _boundary_collar_mask,
    der,
    jer,
    labels_to_matrix,
)


class TestPerfectAndTrivial:
    def test_identical_reference_and_hypothesis_is_zero(self, two_speaker_reference):
        r = der(two_speaker_reference, two_speaker_reference)
        assert r.der == 0.0
        assert (r.miss, r.false_alarm, r.confusion) == (0.0, 0.0, 0.0)

    def test_permuted_labels_still_score_zero(self, two_speaker_reference):
        """A relabelling is not an error: that is what the optimal mapping is for."""
        hyp = two_speaker_reference[:, ::-1].copy()
        assert der(two_speaker_reference, hyp).der == 0.0

    def test_mapping_is_the_permutation(self, two_speaker_reference):
        hyp = two_speaker_reference[:, ::-1].copy()
        assert der(two_speaker_reference, hyp).mapping == {0: 1, 1: 0}

    def test_empty_hypothesis_is_all_miss(self, two_speaker_reference):
        """No hypothesis speech: every reference frame is a miss, so DER is exactly 1."""
        r = der(two_speaker_reference, np.zeros((10, 0), dtype=bool))
        assert r.der == 1.0
        assert r.miss == 1.0
        assert r.false_alarm == 0.0
        assert r.confusion == 0.0

    def test_denominator_is_reference_speaker_frames(self, two_speaker_reference):
        assert der(two_speaker_reference, two_speaker_reference).total_ref_frames == 10


class TestHandComputedDecomposition:
    def test_single_speaker_hypothesis_is_pure_confusion(self, two_speaker_reference):
        """Ref: A on 0-4, B on 5-9. Hyp: one speaker everywhere.

        Every frame has one reference and one hypothesis speaker, so miss and
        false alarm are 0. The mapping can only match one of A/B, so 5 of the 10
        reference frames are confusions: DER = 5/10 = 0.5.
        """
        hyp = np.ones((10, 1), dtype=bool)
        r = der(two_speaker_reference, hyp)
        assert r.confusion == pytest.approx(0.5)
        assert r.miss == 0.0
        assert r.false_alarm == 0.0
        assert r.der == pytest.approx(0.5)

    def test_pure_false_alarm(self):
        """Ref: one speaker on frames 0-4 only. Hyp: that speaker on all 10 frames.

        Reference speaker-frames = 5. Frames 5-9 have a hypothesis speaker and no
        reference speaker: 5 false alarms. DER = 5/5 = 1.0, and DER exceeding the
        speech fraction is normal.
        """
        ref = np.zeros((10, 1), dtype=bool)
        ref[:5, 0] = True
        hyp = np.ones((10, 1), dtype=bool)
        r = der(ref, hyp)
        assert r.false_alarm == pytest.approx(1.0)
        assert r.miss == 0.0
        assert r.confusion == 0.0
        assert r.der == pytest.approx(1.0)

    def test_pure_miss(self):
        """Ref: speaker on 0-9. Hyp: the same speaker on 0-4 only. Miss = 5/10."""
        ref = np.ones((10, 1), dtype=bool)
        hyp = np.zeros((10, 1), dtype=bool)
        hyp[:5, 0] = True
        r = der(ref, hyp)
        assert r.miss == pytest.approx(0.5)
        assert r.false_alarm == 0.0
        assert r.der == pytest.approx(0.5)

    def test_der_can_exceed_one(self):
        """Ref: 1 speaker on 0-1. Hyp: 3 speakers everywhere over 10 frames.

        Reference frames = 2. Frame 0 and 1: 1 ref vs 3 hyp -> 2 FA each = 4.
        Frames 2-9: 0 ref vs 3 hyp -> 3 FA each = 24. Total FA = 28.
        Confusion = sum(min(r,h)) - correct = 2 - 2 = 0. DER = 28/2 = 14.0.
        """
        ref = np.zeros((10, 1), dtype=bool)
        ref[:2, 0] = True
        hyp = np.ones((10, 3), dtype=bool)
        r = der(ref, hyp)
        assert r.false_alarm == pytest.approx(14.0)
        assert r.der == pytest.approx(14.0)

    def test_components_sum_to_der(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            ref = rng.random((40, 3)) < 0.3
            hyp = rng.random((40, 4)) < 0.3
            r = der(ref, hyp)
            if np.isfinite(r.der):
                assert r.der == pytest.approx(r.miss + r.false_alarm + r.confusion)

    def test_components_are_never_negative(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            ref = rng.random((30, 4)) < 0.4
            hyp = rng.random((30, 2)) < 0.4
            r = der(ref, hyp)
            if np.isfinite(r.der):
                assert min(r.miss, r.false_alarm, r.confusion) >= -1e-12


class TestOverlap:
    def test_overlap_doubles_the_denominator(self, overlapping_reference):
        """Ref: A on 0-5 (6), B on 4-9 (6). Total reference speaker-frames = 12."""
        assert der(overlapping_reference, overlapping_reference).total_ref_frames == 12

    def test_perfect_overlap_hypothesis_is_zero(self, overlapping_reference):
        assert der(overlapping_reference, overlapping_reference).der == 0.0

    def test_single_label_system_pays_the_overlap_floor(self, overlapping_reference):
        """Ref frames 4 and 5 have two speakers; a one-speaker-per-frame hypothesis
        misses one of them at each, so DER >= 2/12 = 0.1667 no matter what."""
        hyp = np.zeros((10, 2), dtype=bool)
        hyp[:5, 0] = True
        hyp[5:, 1] = True
        r = der(overlapping_reference, hyp)
        assert r.miss == pytest.approx(2 / 12)
        assert r.der == pytest.approx(2 / 12)

    def test_excluding_overlap_removes_that_floor(self, overlapping_reference):
        hyp = np.zeros((10, 2), dtype=bool)
        hyp[:5, 0] = True
        hyp[5:, 1] = True
        assert der(overlapping_reference, hyp, score_overlap=False).der == 0.0

    def test_excluding_overlap_reduces_scored_frames(self, overlapping_reference):
        full = der(overlapping_reference, overlapping_reference)
        excl = der(overlapping_reference, overlapping_reference, score_overlap=False)
        assert excl.scored_frames == full.scored_frames - 2

    def test_all_overlap_recording(self):
        """Every frame has both speakers. A single-label hypothesis loses exactly half."""
        ref = np.ones((10, 2), dtype=bool)
        hyp = np.ones((10, 1), dtype=bool)
        r = der(ref, hyp)
        assert r.total_ref_frames == 20
        assert r.miss == pytest.approx(0.5)


class TestOptimalMappingMatters:
    """The 2x2 case from ``assignment.py``, embedded in a real reference/hypothesis.

    Ref A on frames 0-18 (19 frames), ref B on 19-27 (9 frames).
    Hyp 0 on 0-9 and 19-27 (19 frames), hyp 1 on 10-18 (9 frames).
    Co-occurrence matrix is [[10, 9], [9, 0]].
    Every frame has exactly one reference and one hypothesis speaker, so
    miss = false alarm = 0 and DER is pure confusion over 28 reference frames.

    Optimal mapping A->1, B->0: correct = 9 + 9 = 18, confusion = 10, DER = 10/28.
    Greedy mapping  A->0, B->1: correct = 10 + 0 = 10, confusion = 18, DER = 18/28.
    """

    @staticmethod
    def _build() -> tuple[np.ndarray, np.ndarray]:
        ref = np.zeros((28, 2), dtype=bool)
        ref[0:19, 0] = True
        ref[19:28, 1] = True
        hyp = np.zeros((28, 2), dtype=bool)
        hyp[0:10, 0] = True
        hyp[19:28, 0] = True
        hyp[10:19, 1] = True
        return ref, hyp

    def test_cooccurrence_matrix_is_the_counterexample(self):
        ref, hyp = self._build()
        overlap = ref.astype(int).T @ hyp.astype(int)
        assert overlap.tolist() == [[10, 9], [9, 0]]

    def test_der_uses_the_optimal_mapping(self):
        ref, hyp = self._build()
        r = der(ref, hyp)
        assert r.mapping == {0: 1, 1: 0}
        assert r.der == pytest.approx(10 / 28)

    def test_greedy_mapping_would_inflate_der(self):
        ref, hyp = self._build()
        overlap = (ref.astype(int).T @ hyp.astype(int)).astype(float)
        g_rows, g_cols = greedy_assignment(overlap, maximize=True)
        o_rows, o_cols = linear_sum_assignment(overlap, maximize=True)
        greedy_correct = overlap[g_rows, g_cols].sum()
        optimal_correct = overlap[o_rows, o_cols].sum()
        assert greedy_correct == 10
        assert optimal_correct == 18
        assert (28 - greedy_correct) / 28 == pytest.approx(18 / 28)

    def test_miss_and_false_alarm_are_zero_here(self):
        ref, hyp = self._build()
        r = der(ref, hyp)
        assert (r.miss, r.false_alarm) == (0.0, 0.0)


class TestJer:
    def test_hand_computed_jer(self):
        """Same construction as above.

        Ref A (19 frames) maps to hyp 1 (9 frames): intersection 9,
          miss 10, false alarm 0, total 19 -> 10/19.
        Ref B (9 frames) maps to hyp 0 (19 frames): intersection 9,
          miss 0, false alarm 10, total 9 -> 10/9.
        JER = mean(10/19, 10/9) = 0.818713...
        """
        ref, hyp = TestOptimalMappingMatters._build()
        r = der(ref, hyp)
        expected = ((10 / 19) + (10 / 9)) / 2
        assert r.jer == pytest.approx(expected)

    def test_perfect_jer_is_zero(self, two_speaker_reference):
        assert der(two_speaker_reference, two_speaker_reference).jer == 0.0

    def test_unmapped_reference_speaker_scores_one(self):
        """Two reference speakers, one hypothesis speaker: the unmapped ref scores 1.0."""
        ref = np.zeros((10, 2), dtype=bool)
        ref[:5, 0] = True
        ref[5:, 1] = True
        hyp = np.zeros((10, 1), dtype=bool)
        hyp[:5, 0] = True
        # Ref 0 -> hyp 0 perfectly (error 0); ref 1 unmapped (error 1). JER = 0.5.
        assert der(ref, hyp).jer == pytest.approx(0.5)

    def test_jer_ignores_reference_speakers_with_no_speech(self):
        ref = np.zeros((10, 3), dtype=bool)
        ref[:5, 0] = True
        ref[5:, 1] = True  # column 2 is empty
        assert der(ref, ref).jer == 0.0

    def test_jer_weights_speakers_not_time(self):
        """A dominant speaker cannot hide a completely missed short one.

        Ref: A on 0-89 (90 frames), B on 90-99 (10 frames). Hyp: A everywhere.
        DER = 10/100 = 0.10, but B is entirely lost, so JER is much larger.
        """
        ref = np.zeros((100, 2), dtype=bool)
        ref[:90, 0] = True
        ref[90:, 1] = True
        hyp = np.ones((100, 1), dtype=bool)
        r = der(ref, hyp)
        assert r.der == pytest.approx(0.10)
        assert r.jer > 0.5

    def test_jer_of_no_reference_speakers_is_nan(self):
        assert np.isnan(jer(np.zeros((5, 0), dtype=bool), np.zeros((5, 1), dtype=bool), {}))


class TestCollar:
    def test_zero_collar_scores_every_frame(self, two_speaker_reference):
        assert der(two_speaker_reference, two_speaker_reference).scored_frames == 10

    def test_collar_excludes_frames_around_boundaries(self, two_speaker_reference):
        """Boundaries at 0, 5 and 10. A 2-frame collar removes 0-1, 3-6 and 8-9,
        leaving frames 2 and 7 scored."""
        keep = _boundary_collar_mask(two_speaker_reference, 2)
        assert keep.tolist() == [False, False, True, False, False,
                                 False, False, True, False, False]

    def test_collar_finds_speaker_changes_inside_continuous_speech(self):
        """Activity never drops to zero at frame 5, but the speaker changes there,
        so it is still a boundary."""
        ref = np.zeros((10, 2), dtype=bool)
        ref[:5, 0] = True
        ref[5:, 1] = True
        keep = _boundary_collar_mask(ref, 1)
        assert not keep[4] and not keep[5]

    def test_collar_can_forgive_a_boundary_error(self):
        """Hypothesis boundary one frame late. Scored strictly that is 1 error in 10;
        with a 2-frame collar the offending frame is not scored at all."""
        ref = np.zeros((10, 2), dtype=bool)
        ref[:5, 0] = True
        ref[5:, 1] = True
        hyp = np.zeros((10, 2), dtype=bool)
        hyp[:6, 0] = True
        hyp[6:, 1] = True
        assert der(ref, hyp).der == pytest.approx(0.1)
        assert der(ref, hyp, collar_frames=2).der == 0.0

    def test_a_large_collar_can_score_nothing_and_yield_nan(self):
        ref = np.zeros((10, 2), dtype=bool)
        ref[:5, 0] = True
        ref[5:, 1] = True
        r = der(ref, ref, collar_frames=20)
        assert r.scored_frames == 0
        assert np.isnan(r.der)

    def test_negative_collar_is_treated_as_zero(self, two_speaker_reference):
        keep = _boundary_collar_mask(two_speaker_reference, -3)
        assert keep.all()


class TestUndefinedIsNan:
    def test_no_reference_speech_gives_nan_not_zero(self):
        """An empty reference has no DER. Returning 0.0 would drag down a sweep mean."""
        hyp = np.ones((10, 1), dtype=bool)
        r = der(np.zeros((10, 2), dtype=bool), hyp)
        assert np.isnan(r.der)
        assert np.isnan(r.miss) and np.isnan(r.false_alarm) and np.isnan(r.confusion)
        assert np.isnan(r.jer)

    def test_nan_result_still_reports_speaker_counts(self):
        r = der(np.zeros((10, 2), dtype=bool), np.ones((10, 3), dtype=bool))
        assert r.n_ref_speakers == 2
        assert r.n_hyp_speakers == 3
        assert r.speaker_count_error == 1

    def test_both_empty_is_nan(self):
        r = der(np.zeros((5, 0), dtype=bool), np.zeros((5, 0), dtype=bool))
        assert np.isnan(r.der)


class TestSpeakerCount:
    def test_signed_count_error(self):
        ref = np.zeros((10, 3), dtype=bool)
        ref[:, 0] = True
        hyp = np.ones((10, 1), dtype=bool)
        assert der(ref, hyp).speaker_count_error == -2

    def test_over_clustering_is_positive(self):
        ref = np.ones((10, 1), dtype=bool)
        hyp = np.zeros((10, 4), dtype=bool)
        hyp[:, 0] = True
        assert der(ref, hyp).speaker_count_error == 3


class TestValidation:
    def test_frame_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="frame count mismatch"):
            der(np.zeros((10, 2), dtype=bool), np.zeros((9, 2), dtype=bool))

    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            der(np.zeros(10, dtype=bool), np.zeros((10, 1), dtype=bool))

    def test_non_bool_input_is_coerced(self):
        ref = np.zeros((10, 2), dtype=np.int8)
        ref[:5, 0] = 1
        ref[5:, 1] = 1
        assert der(ref, ref).der == 0.0

    def test_str_repr_contains_the_components(self, two_speaker_reference):
        text = str(der(two_speaker_reference, two_speaker_reference))
        for token in ["DER=", "miss=", "fa=", "conf=", "ref=", "hyp="]:
            assert token in text

    def test_to_dict_has_int_keyed_mapping(self, two_speaker_reference):
        d = der(two_speaker_reference, two_speaker_reference).to_dict()
        assert all(isinstance(k, int) for k in d["mapping"])


class TestLabelsToMatrix:
    def test_non_speech_becomes_an_all_false_row(self):
        out = labels_to_matrix(np.array([-1, 0, 0]), 3)
        assert out[0].sum() == 0

    def test_labels_are_densified_by_first_appearance(self):
        """Sparse ids 3 and 7 become columns 0 and 1, in the order they first appear."""
        out = labels_to_matrix(np.array([-1, 3, 3, 7, -1, 7]), 6)
        assert out.shape == (6, 2)
        assert out[:, 0].tolist() == [False, True, True, False, False, False]
        assert out[:, 1].tolist() == [False, False, False, True, False, True]

    def test_unused_spawned_id_does_not_widen_the_matrix(self):
        """A diarizer that allocated id 9 but never used it must not be charged
        with a speaker-count error."""
        out = labels_to_matrix(np.array([0, 0, 2, 2]), 4)
        assert out.shape[1] == 2

    def test_all_non_speech_gives_zero_columns(self):
        out = labels_to_matrix(np.array([-1, -1, -1]), 3)
        assert out.shape == (3, 0)

    def test_explicit_n_speakers_is_respected(self):
        out = labels_to_matrix(np.array([0, 1]), 2, n_speakers=5)
        assert out.shape == (2, 5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels length"):
            labels_to_matrix(np.array([0, 1]), 5)

    def test_exactly_one_speaker_per_speech_frame(self):
        out = labels_to_matrix(np.array([0, 1, 2, -1]), 4)
        assert out.sum(axis=1).tolist() == [1, 1, 1, 0]


class TestInvariants:
    def test_der_is_invariant_to_hypothesis_relabelling(self):
        rng = np.random.default_rng(5)
        ref = rng.random((60, 3)) < 0.35
        hyp = rng.random((60, 3)) < 0.35
        base = der(ref, hyp).der
        perm = rng.permutation(3)
        assert der(ref, hyp[:, perm]).der == pytest.approx(base)

    def test_der_is_invariant_to_reference_relabelling(self):
        rng = np.random.default_rng(6)
        ref = rng.random((60, 3)) < 0.35
        hyp = rng.random((60, 4)) < 0.35
        base = der(ref, hyp).der
        perm = rng.permutation(3)
        assert der(ref[:, perm], hyp).der == pytest.approx(base)

    def test_extra_unused_hypothesis_column_costs_nothing(self, two_speaker_reference):
        hyp = np.concatenate(
            [two_speaker_reference, np.zeros((10, 1), dtype=bool)], axis=1
        )
        assert der(two_speaker_reference, hyp).der == 0.0

    def test_duplicating_a_hypothesis_speaker_creates_false_alarms(self, two_speaker_reference):
        """Emitting the same speech twice under two labels is a false alarm, not free."""
        hyp = np.concatenate(
            [two_speaker_reference, two_speaker_reference[:, :1]], axis=1
        )
        assert der(two_speaker_reference, hyp).false_alarm > 0

    def test_time_reversal_leaves_der_unchanged(self):
        rng = np.random.default_rng(8)
        ref = rng.random((50, 2)) < 0.4
        hyp = rng.random((50, 3)) < 0.4
        assert der(ref[::-1], hyp[::-1]).der == pytest.approx(der(ref, hyp).der)

    def test_repeating_every_frame_leaves_der_unchanged(self):
        """DER is a rate, so uniformly rescaling time must not change it."""
        rng = np.random.default_rng(9)
        ref = rng.random((30, 2)) < 0.4
        hyp = rng.random((30, 2)) < 0.4
        assert der(np.repeat(ref, 3, axis=0), np.repeat(hyp, 3, axis=0)).der == pytest.approx(
            der(ref, hyp).der
        )
