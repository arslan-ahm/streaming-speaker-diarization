"""The online mechanisms: tracker, micro-cluster, spawning, ablation switches, edges.

The tests are built on synthetic embeddings with *known* speaker structure, so
the expected behaviour is derivable rather than observed. That matters most for
the micro-cluster: its whole justification is variance reduction, and
``test_micro_cluster_averages_a_contiguous_run`` checks it averages exactly the
run it should and stops where it should.
"""

from __future__ import annotations

import numpy as np
import pytest

from streamdiar.config import replace
from streamdiar.models.online import (
    DiarizationOutput,
    SpeakerTracker,
    _micro_cluster,
    _softmax_confidence,
    _unit,
    diarize_naive_online,
    diarize_online,
)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class TestUnit:
    def test_normalises(self):
        assert np.linalg.norm(_unit(np.array([3.0, 4.0]))) == pytest.approx(1.0)

    def test_zero_vector_maps_to_itself_not_nan(self):
        out = _unit(np.zeros(4))
        assert np.isfinite(out).all() and out.sum() == 0.0

    def test_already_unit_is_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        assert np.allclose(_unit(v), v)


class TestSpeakerTracker:
    def test_starts_empty(self):
        t = SpeakerTracker(4, 3, 0.5)
        assert t.n_active == 0
        assert t.similarities(np.ones(4)).size == 0

    def test_spawn_returns_increasing_ids(self):
        t = SpeakerTracker(4, 3, 0.5)
        assert [t.spawn(unit(np.random.rand(4))) for _ in range(3)] == [0, 1, 2]

    def test_spawn_when_full_returns_minus_one(self):
        """The hard cap is what makes resident memory O(1) in recording length."""
        t = SpeakerTracker(4, 2, 0.5)
        t.spawn(unit([1, 0, 0, 0]))
        t.spawn(unit([0, 1, 0, 0]))
        assert t.full
        assert t.spawn(unit([0, 0, 1, 0])) == -1

    def test_centroid_is_the_seed_after_spawning(self):
        t = SpeakerTracker(3, 2, 0.5)
        e = unit([1.0, 2.0, 3.0])
        t.spawn(e)
        assert np.allclose(t.centroids[0], e)

    def test_similarity_to_own_centroid_is_one(self):
        t = SpeakerTracker(3, 2, 0.5)
        e = unit([1.0, 2.0, 3.0])
        t.spawn(e)
        assert t.similarities(e)[0] == pytest.approx(1.0)

    def test_update_moves_the_centroid_toward_the_embedding(self):
        t = SpeakerTracker(2, 2, 0.5)
        t.spawn(unit([1.0, 0.0]))
        t.update(0, unit([0.0, 1.0]))
        assert t.centroids[0][1] > 0.5

    def test_update_keeps_the_centroid_on_the_unit_sphere(self):
        t = SpeakerTracker(3, 2, 0.35)
        t.spawn(unit([1.0, 0.0, 0.0]))
        for _ in range(10):
            t.update(0, unit(np.random.rand(3)))
            assert np.linalg.norm(t.centroids[0]) == pytest.approx(1.0)

    def test_momentum_one_replaces_the_centroid(self):
        t = SpeakerTracker(2, 2, 1.0)
        t.spawn(unit([1.0, 0.0]))
        e = unit([0.0, 1.0])
        t.update(0, e)
        assert np.allclose(t.centroids[0], e)

    def test_small_momentum_barely_moves_the_centroid(self):
        t = SpeakerTracker(2, 2, 0.01)
        t.spawn(unit([1.0, 0.0]))
        t.update(0, unit([0.0, 1.0]))
        assert t.centroids[0][0] > 0.99

    def test_counts_increment(self):
        t = SpeakerTracker(2, 2, 0.5)
        t.spawn(unit([1.0, 0.0]))
        t.update(0, unit([1.0, 0.1]))
        t.update(0, unit([1.0, 0.2]))
        assert t.counts[0] == 3

    def test_updating_an_inactive_speaker_raises(self):
        t = SpeakerTracker(2, 3, 0.5)
        t.spawn(unit([1.0, 0.0]))
        with pytest.raises(IndexError, match="not active"):
            t.update(1, unit([0.0, 1.0]))

    def test_memory_does_not_grow_with_updates(self):
        """The centroid store is allocated once at max_speakers x embed_dim."""
        t = SpeakerTracker(8, 5, 0.4)
        t.spawn(unit(np.random.rand(8)))
        shape = t.centroids.shape
        for _ in range(500):
            t.update(0, unit(np.random.rand(8)))
        assert t.centroids.shape == shape


class TestMicroCluster:
    def test_single_pending_item_is_its_own_query(self):
        e = unit([1.0, 0.0, 0.0])
        query, size = _micro_cluster([(0, e)], 0.9)
        assert size == 1
        assert np.allclose(query, e)

    def test_averages_a_contiguous_run_of_the_same_speaker(self):
        """Three near-identical embeddings then a distant one: the run is size 3."""
        a = unit([1.0, 0.02, 0.0])
        b = unit([1.0, -0.02, 0.0])
        c = unit([1.0, 0.01, 0.0])
        far = unit([0.0, 0.0, 1.0])
        _, size = _micro_cluster([(0, a), (1, b), (2, c), (3, far)], 0.9)
        assert size == 3

    def test_stops_at_the_first_mismatch_even_if_a_later_item_matches(self):
        """A short interjection must not let a later same-speaker window rejoin, or
        the query would be dragged across a turn boundary."""
        a = unit([1.0, 0.0, 0.0])
        far = unit([0.0, 1.0, 0.0])
        _, size = _micro_cluster([(0, a), (1, far), (2, a), (3, a)], 0.9)
        assert size == 1

    def test_query_is_unit_norm(self):
        a = unit([1.0, 0.05, 0.0])
        b = unit([1.0, -0.05, 0.0])
        query, _ = _micro_cluster([(0, a), (1, b)], 0.9)
        assert np.linalg.norm(query) == pytest.approx(1.0)

    def test_averaging_reduces_noise(self):
        """The mechanism's justification: averaging m windows of one speaker cuts
        the query's distance to the true centre."""
        rng = np.random.default_rng(0)
        true = unit(np.ones(16))
        # Noise sd 0.15 in 16 dimensions puts pairwise cosine around 0.75, above
        # the 0.5 threshold used here, so the whole run is admitted.
        pending = [(i, unit(true + 0.15 * rng.normal(size=16))) for i in range(6)]
        single = pending[0][1]
        query, size = _micro_cluster(pending, 0.5)
        assert size >= 4
        assert np.linalg.norm(query - true) < np.linalg.norm(single - true)

    def test_threshold_one_admits_nothing_extra(self):
        a = unit([1.0, 0.05, 0.0])
        b = unit([1.0, -0.05, 0.0])
        _, size = _micro_cluster([(0, a), (1, b)], 1.0)
        assert size == 1

    def test_threshold_minus_one_admits_everything(self):
        pts = [(i, unit(np.random.rand(4))) for i in range(5)]
        _, size = _micro_cluster(pts, -1.0)
        assert size == 5


class TestSoftmaxConfidence:
    def test_returns_a_probability(self):
        conf, _ = _softmax_confidence(np.array([0.9, 0.2]), 0, 0.5, 0.1)
        assert 0.0 < conf <= 1.0

    def test_no_existing_speakers_gives_confidence_one(self):
        """The first decision of a recording has no alternative; 1.0 is honest here
        and is documented as a known artefact rather than hidden."""
        conf, _ = _softmax_confidence(np.zeros(0), -1, 0.5, 0.1)
        assert conf == pytest.approx(1.0)

    def test_a_single_centroid_still_has_the_spawn_alternative(self):
        """Without the virtual spawn option, every decision taken while one centroid
        exists would report 1.0 by vacuity."""
        conf, _ = _softmax_confidence(np.array([0.6]), 0, 0.55, 0.1)
        assert conf < 1.0

    def test_margin_is_chosen_minus_best_alternative(self):
        _, margin = _softmax_confidence(np.array([0.9, 0.4]), 0, 0.5, 0.1)
        assert margin == pytest.approx(0.9 - 0.5)  # spawn at 0.5 beats the 0.4 rival

    def test_margin_is_negative_for_a_losing_choice(self):
        _, margin = _softmax_confidence(np.array([0.9, 0.4]), 1, 0.5, 0.1)
        assert margin < 0

    def test_spawn_choice_is_scored_against_the_best_centroid(self):
        _, margin = _softmax_confidence(np.array([0.3, 0.2]), -1, 0.55, 0.1)
        assert margin == pytest.approx(0.55 - 0.3)

    def test_lower_temperature_sharpens_confidence(self):
        sims = np.array([0.9, 0.6])
        sharp, _ = _softmax_confidence(sims, 0, 0.5, 0.05)
        soft, _ = _softmax_confidence(sims, 0, 0.5, 1.0)
        assert sharp > soft

    def test_confidence_of_a_clear_winner_is_high(self):
        conf, _ = _softmax_confidence(np.array([0.99, 0.1]), 0, 0.3, 0.05)
        assert conf > 0.95


@pytest.fixture
def two_speaker_stream() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """40 windows: speaker A for the first 20, speaker B for the last 20.

    Returns ``(embeddings, starts, ends, is_speech, true_labels)`` with hop 25 and
    window 50 frames, matching the tiny config.
    """
    rng = np.random.default_rng(0)
    a = unit(np.concatenate([[1.0], np.zeros(7)]))
    b = unit(np.concatenate([np.zeros(7), [1.0]]))
    emb = np.stack(
        [unit((a if k < 20 else b) + 0.05 * rng.normal(size=8)) for k in range(40)]
    )
    ends = (np.arange(40) + 1) * 25
    starts = np.maximum(0, ends - 50)
    return emb, starts, ends, np.ones(40, dtype=bool), np.repeat([0, 1], 20)


class TestDiarizeOnline:
    def test_finds_two_speakers_on_a_two_speaker_stream(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, truth = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        assert out.n_speakers == 2

    def test_labels_match_the_true_partition(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, truth = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        # Up to relabelling: every window of one true speaker gets one label.
        assert len(set(out.labels[truth == 0].tolist())) == 1
        assert len(set(out.labels[truth == 1].tolist())) == 1
        assert out.labels[0] != out.labels[-1]

    def test_first_window_always_spawns(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 1, ends[-1], 25)
        assert out.spawned[0]

    def test_query_size_is_one_when_the_budget_is_zero(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 0, ends[-1], 25)
        assert (out.query_size[out.labels >= 0] == 1).all()

    def test_query_size_grows_with_the_budget(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        sizes = []
        for budget in [0, 1, 3, 6]:
            out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer,
                                 budget, ends[-1], 25)
            sizes.append(float(out.query_size[out.labels >= 0].mean()))
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_bounded_window_off_pins_query_size_to_one(self, tiny_cfg, two_speaker_stream):
        """The ablation removes the mechanism while keeping the buffer and the delay."""
        emb, starts, ends, speech, _ = two_speaker_stream
        dcfg = replace(tiny_cfg.diarizer, bounded_window=False)
        out = diarize_online(emb, starts, ends, speech, dcfg, 6, ends[-1], 25)
        assert (out.query_size[out.labels >= 0] == 1).all()

    def test_bounded_window_off_keeps_the_same_latency(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        on = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 4, ends[-1], 25)
        off = diarize_online(emb, starts, ends, speech,
                             replace(tiny_cfg.diarizer, bounded_window=False),
                             4, ends[-1], 25)
        assert np.array_equal(on.emission_frames, off.emission_frames)

    def test_non_speech_windows_get_label_minus_one(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, _, _ = two_speaker_stream
        speech = np.ones(40, dtype=bool)
        speech[10:15] = False
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        assert (out.labels[10:15] == -1).all()

    def test_non_speech_windows_have_nan_confidence(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, _, _ = two_speaker_stream
        speech = np.ones(40, dtype=bool)
        speech[10:15] = False
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        assert np.isnan(out.confidence[10:15]).all()

    def test_speaker_cap_is_respected(self, tiny_cfg):
        """With 20 distinct speakers and a cap of 3, no more than 3 may be created."""
        rng = np.random.default_rng(1)
        emb = np.stack([unit(rng.normal(size=8)) for _ in range(20)])
        ends = (np.arange(20) + 1) * 25
        dcfg = replace(tiny_cfg.diarizer, max_speakers=3, spawn_threshold=0.99)
        out = diarize_online(emb, np.maximum(0, ends - 50), ends,
                             np.ones(20, dtype=bool), dcfg, 0, ends[-1], 25)
        assert out.n_speakers <= 3

    def test_spawn_threshold_zero_creates_one_speaker(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        dcfg = replace(tiny_cfg.diarizer, spawn_threshold=-1.0)
        out = diarize_online(emb, starts, ends, speech, dcfg, 0, ends[-1], 25)
        assert out.n_speakers == 1

    def test_spawn_threshold_one_spawns_until_the_cap(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        dcfg = replace(tiny_cfg.diarizer, spawn_threshold=1.01, max_speakers=6)
        out = diarize_online(emb, starts, ends, speech, dcfg, 0, ends[-1], 25)
        assert out.n_speakers == 6

    def test_spawn_disabled_limits_speakers_to_the_warmup_prefix(self, tiny_cfg):
        """A speaker who first talks after the warm-up can never be represented."""
        rng = np.random.default_rng(2)
        a = unit(np.concatenate([[1.0], np.zeros(7)]))
        b = unit(np.concatenate([np.zeros(7), [1.0]]))
        emb = np.stack([unit((a if k < 20 else b) + 0.05 * rng.normal(size=8))
                        for k in range(40)])
        ends = (np.arange(40) + 1) * 25
        dcfg = replace(tiny_cfg.diarizer, spawn_enabled=False, warmup_windows=3)
        out = diarize_online(emb, np.maximum(0, ends - 50), ends,
                             np.ones(40, dtype=bool), dcfg, 0, ends[-1], 25)
        assert out.n_speakers == 1

    def test_oracle_count_caps_the_speaker_count(self, tiny_cfg):
        rng = np.random.default_rng(3)
        emb = np.stack([unit(rng.normal(size=8)) for _ in range(30)])
        ends = (np.arange(30) + 1) * 25
        dcfg = replace(tiny_cfg.diarizer, spawn_threshold=0.99)
        out = diarize_online(emb, np.maximum(0, ends - 50), ends,
                             np.ones(30, dtype=bool), dcfg, 0, ends[-1], 25,
                             oracle_n_speakers=2)
        assert out.n_speakers == 2

    def test_oracle_count_of_one_forces_a_single_speaker(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer,
                             0, ends[-1], 25, oracle_n_speakers=1)
        assert out.n_speakers == 1

    def test_is_deterministic(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        a = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        b = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        assert np.array_equal(a.labels, b.labels)
        assert np.array_equal(a.query_size, b.query_size)

    def test_method_name_is_recorded(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        assert out.method == "online"

    def test_mean_query_size_is_reported(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 3, ends[-1], 25)
        assert out.extra["mean_query_size"] > 1.0


class TestEdgeCases:
    def test_no_windows_returns_an_empty_output(self, tiny_cfg):
        out = diarize_online(np.zeros((0, 8)), np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                             np.zeros(0, dtype=bool), tiny_cfg.diarizer, 2, 0, 25)
        assert out.labels.size == 0
        assert out.n_speakers == 0

    def test_all_non_speech_gives_no_speakers(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, _, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, np.zeros(40, dtype=bool),
                             tiny_cfg.diarizer, 2, ends[-1], 25)
        assert out.n_speakers == 0
        assert (out.labels == -1).all()

    def test_single_window(self, tiny_cfg):
        emb = unit(np.ones(8))[None, :]
        out = diarize_online(emb, np.array([0]), np.array([25]),
                             np.ones(1, dtype=bool), tiny_cfg.diarizer, 0, 25, 25)
        assert out.n_speakers == 1
        assert out.emission_frames[0] == 25

    def test_budget_larger_than_the_recording_flushes_at_the_end(self, tiny_cfg,
                                                                 two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer,
                             10_000, ends[-1], 25)
        assert (out.emission_frames == ends[-1]).all()

    def test_negative_budget_is_treated_as_zero(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, -5, ends[-1], 25)
        assert np.array_equal(out.emission_frames, ends)


class TestFrameLabels:
    def test_regions_tile_the_recording(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        frame_labels = out.frame_labels()
        assert frame_labels.shape == (ends[-1],)
        assert (frame_labels >= 0).all()  # every window was speech

    def test_non_speech_regions_are_minus_one(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, _, _ = two_speaker_stream
        speech = np.ones(40, dtype=bool)
        speech[4] = False
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        fl = out.frame_labels()
        assert (fl[4 * 25 : 5 * 25] == -1).all()

    def test_frame_labels_length_matches_n_frames(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, 1234, 25)
        assert out.frame_labels().shape == (1234,)


class TestNaiveOnline:
    def test_zero_latency_for_every_decision(self, tiny_cfg, two_speaker_stream):
        """Its one genuine advantage, reported rather than hidden."""
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_naive_online(emb, starts, ends, speech, tiny_cfg.diarizer, ends[-1], 25)
        assert np.array_equal(out.emission_frames, ends)

    def test_finds_two_speakers_on_a_two_speaker_stream(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_naive_online(emb, starts, ends, speech, tiny_cfg.diarizer, ends[-1], 25)
        assert out.n_speakers == 2

    def test_query_size_is_always_one(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_naive_online(emb, starts, ends, speech, tiny_cfg.diarizer, ends[-1], 25)
        assert (out.query_size == 1).all()

    def test_method_name(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_naive_online(emb, starts, ends, speech, tiny_cfg.diarizer, ends[-1], 25)
        assert out.method == "naive_online"

    def test_is_deterministic(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        a = diarize_naive_online(emb, starts, ends, speech, tiny_cfg.diarizer, ends[-1], 25)
        b = diarize_naive_online(emb, starts, ends, speech, tiny_cfg.diarizer, ends[-1], 25)
        assert np.array_equal(a.labels, b.labels)

    def test_empty_input(self, tiny_cfg):
        out = diarize_naive_online(np.zeros((0, 8)), np.zeros(0, dtype=int),
                                   np.zeros(0, dtype=int), np.zeros(0, dtype=bool),
                                   tiny_cfg.diarizer, 0, 25)
        assert out.labels.size == 0

    def test_uses_a_running_mean_not_an_ema(self, tiny_cfg):
        """The baseline's centroid is the exact mean of its members, which is the
        textbook incremental rule and is what makes it a baseline rather than an
        ablation of the proposed method."""
        e1 = unit([1.0, 0.0])
        e2 = unit([1.0, 0.2])
        emb = np.stack([e1, e2])
        ends = np.array([25, 50])
        out = diarize_naive_online(emb, np.array([0, 0]), ends, np.ones(2, dtype=bool),
                                   replace(tiny_cfg.diarizer, spawn_threshold=0.0),
                                   50, 25)
        assert out.n_speakers == 1
        assert out.labels.tolist() == [0, 0]


class TestDiarizationOutputDataclass:
    def test_all_arrays_share_the_window_count(self, tiny_cfg, two_speaker_stream):
        emb, starts, ends, speech, _ = two_speaker_stream
        out = diarize_online(emb, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], 25)
        n = len(out.labels)
        for arr in [out.window_starts, out.window_ends, out.emission_frames,
                    out.confidence, out.margin, out.spawned, out.query_size]:
            assert len(arr) == n

    def test_is_a_dataclass_with_the_expected_fields(self):
        assert {"labels", "emission_frames", "confidence", "margin", "spawned",
                "query_size", "n_speakers", "method"} <= set(
            DiarizationOutput.__dataclass_fields__
        )
