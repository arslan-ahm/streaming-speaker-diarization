"""The generator: ground truth by construction, determinism, and the nuisance structure.

The central claim these tests defend is that the reference is *exact*. Turns are
sampled first and features rendered from them, so there is no annotation error
anywhere in the evaluation — but that is only true if the rendering actually
respects the turn list, which is what
``test_features_are_louder_inside_turns`` and
``test_overlap_is_summed_in_the_power_domain`` check.
"""

from __future__ import annotations

import numpy as np
import pytest

from streamdiar.config import DataConfig
from streamdiar.data.generator import (
    BACKGROUND_LOG_POWER,
    Turn,
    dataset_stats,
    feature_statistics,
    generate_recording,
    generate_split,
    get_world,
    sample_segment,
    sample_training_batch,
    sample_turns,
)
from streamdiar.utils.seeding import rng_for


class TestTurn:
    def test_n_frames(self):
        assert Turn(10, 25, 0).n_frames == 15

    def test_reversed_turn_has_zero_frames(self):
        assert Turn(25, 10, 0).n_frames == 0

    def test_seconds_conversion(self):
        assert Turn(100, 250, 0).seconds(100) == (1.0, 2.5)

    def test_is_hashable_and_frozen(self):
        t = Turn(0, 5, 1)
        assert hash(t) == hash(Turn(0, 5, 1))
        with pytest.raises(Exception):
            t.start = 3  # type: ignore[misc]


class TestSampleTurns:
    def test_turns_are_within_the_recording(self, tiny_data_cfg):
        turns = sample_turns(tiny_data_cfg, 3, rng_for(0, "t"))
        total = int(tiny_data_cfg.duration_s * tiny_data_cfg.frame_rate)
        assert all(0 <= t.start < t.end <= total for t in turns)

    def test_at_least_one_turn(self, tiny_data_cfg):
        assert len(sample_turns(tiny_data_cfg, 2, rng_for(0, "t"))) >= 1

    def test_speakers_are_within_range(self, tiny_data_cfg):
        turns = sample_turns(tiny_data_cfg, 3, rng_for(1, "t"))
        assert all(0 <= t.speaker < 3 for t in turns)

    def test_no_speaker_succeeds_itself(self, tiny_data_cfg):
        """Two consecutive turns by the same speaker are one turn; forbidding
        self-succession is what makes the sampled log-normal the true marginal."""
        for seed in range(8):
            turns = sample_turns(tiny_data_cfg, 3, rng_for(seed, "t"))
            speakers = [t.speaker for t in turns]
            assert all(a != b for a, b in zip(speakers, speakers[1:]))

    def test_turns_are_in_time_order(self, tiny_data_cfg):
        turns = sample_turns(tiny_data_cfg, 3, rng_for(2, "t"))
        assert [t.start for t in turns] == sorted(t.start for t in turns)

    def test_single_speaker_is_allowed(self, tiny_data_cfg):
        turns = sample_turns(tiny_data_cfg, 1, rng_for(3, "t"))
        assert all(t.speaker == 0 for t in turns)

    def test_zero_overlap_probability_produces_no_overlap(self, tiny_data_cfg):
        cfg = DataConfig(**{**tiny_data_cfg.__dict__, "overlap_prob": 0.0})
        for seed in range(6):
            turns = sample_turns(cfg, 3, rng_for(seed, "t"))
            for a, b in zip(turns, turns[1:]):
                assert b.start >= a.end

    def test_high_overlap_probability_produces_overlap(self, tiny_data_cfg):
        cfg = DataConfig(**{**tiny_data_cfg.__dict__, "overlap_prob": 1.0,
                            "overlap_frac_max": 0.6})
        found = False
        for seed in range(6):
            turns = sample_turns(cfg, 3, rng_for(seed, "t"))
            if any(b.start < a.end for a, b in zip(turns, turns[1:])):
                found = True
        assert found

    def test_a_degenerate_short_recording_still_returns_a_turn(self):
        cfg = DataConfig(duration_s=0.05, frame_rate=100)
        assert len(sample_turns(cfg, 2, rng_for(0, "t"))) >= 1


class TestRecording:
    def test_shape_and_dtype(self, one_recording, tiny_data_cfg):
        assert one_recording.features.shape == (
            int(tiny_data_cfg.duration_s * tiny_data_cfg.frame_rate),
            tiny_data_cfg.n_features,
        )
        assert one_recording.features.dtype == np.float32

    def test_reference_matrix_matches_the_turns(self, one_recording):
        ref = one_recording.reference_matrix()
        for t in one_recording.turns:
            assert ref[t.start : t.end, t.speaker].all()

    def test_reference_matrix_is_false_outside_every_turn(self, one_recording):
        ref = one_recording.reference_matrix()
        covered = np.zeros_like(ref)
        for t in one_recording.turns:
            covered[t.start : t.end, t.speaker] = True
        assert np.array_equal(ref, covered)

    def test_reference_matrix_width_is_the_speaker_count(self, one_recording):
        assert one_recording.reference_matrix().shape[1] == one_recording.n_speakers

    def test_speaker_count_is_within_the_configured_range(self, tiny_data_cfg):
        lo, hi = tiny_data_cfg.n_speakers_range
        for i in range(8):
            rec = generate_recording(tiny_data_cfg, 0, i, pool="test")
            assert lo <= rec.n_speakers <= hi

    def test_explicit_speaker_count_is_honoured(self, tiny_data_cfg):
        rec = generate_recording(tiny_data_cfg, 0, 0, pool="test", n_speakers=2)
        assert rec.n_speakers == 2

    def test_duration_seconds(self, one_recording, tiny_data_cfg):
        assert one_recording.duration_s == pytest.approx(tiny_data_cfg.duration_s)

    def test_speech_fraction_is_a_fraction(self, tiny_recordings):
        assert all(0.0 <= r.speech_fraction() <= 1.0 for r in tiny_recordings)

    def test_overlap_fraction_is_a_fraction(self, tiny_recordings):
        assert all(0.0 <= r.overlap_fraction() <= 1.0 for r in tiny_recordings)

    def test_pool_ids_are_distinct(self, tiny_recordings):
        """Sampling speakers without replacement: two reference columns are never
        the same voice, or a 'confusion' error would be unfalsifiable."""
        for r in tiny_recordings:
            assert len(set(r.pool_ids)) == len(r.pool_ids)

    def test_rttm_round_trips_the_turn_count(self, one_recording):
        from streamdiar.data.real import parse_rttm

        parsed = parse_rttm(one_recording.to_rttm())[one_recording.name]
        assert len(parsed) == len(one_recording.turns)

    def test_rttm_times_match_the_turns(self, one_recording):
        from streamdiar.data.real import parse_rttm

        parsed = sorted(parse_rttm(one_recording.to_rttm())[one_recording.name])
        turns = sorted(one_recording.turns, key=lambda t: t.start)
        for (start, end, _), t in zip(parsed, turns):
            assert start == pytest.approx(t.start / one_recording.frame_rate, abs=1e-3)
            assert end == pytest.approx(t.end / one_recording.frame_rate, abs=1e-3)


class TestGroundTruthByConstruction:
    def test_features_are_louder_inside_turns(self, tiny_data_cfg):
        """The rendering must actually respect the turn list, or the 'exact
        reference' claim is empty. Speech frames sit near log(1 + bg) = 0.13 and
        non-speech near log(bg) = -2.0, so the means must be well separated."""
        rec = generate_recording(tiny_data_cfg, 0, 0, pool="test")
        ref = rec.reference_matrix().any(axis=1)
        assert ref.any() and (~ref).any()
        speech_energy = rec.features[ref].mean()
        silence_energy = rec.features[~ref].mean()
        assert speech_energy > silence_energy + 1.0

    def test_silence_sits_near_the_background_level(self, tiny_data_cfg):
        rec = generate_recording(tiny_data_cfg, 0, 0, pool="test")
        silence = ~rec.reference_matrix().any(axis=1)
        assert rec.features[silence].mean() == pytest.approx(BACKGROUND_LOG_POWER, abs=0.5)

    def test_overlap_is_summed_in_the_power_domain(self, tiny_data_cfg):
        """Two simultaneous speakers must combine as log(exp(a) + exp(b)), not a+b.
        Getting that wrong would make overlap a trivially detectable amplitude
        spike. With power summation, doubling the power adds only log(2) ~ 0.69.
        """
        cfg = DataConfig(**{**tiny_data_cfg.__dict__, "overlap_prob": 1.0,
                            "overlap_frac_max": 0.6})
        for i in range(12):
            rec = generate_recording(cfg, 0, i, pool="test")
            counts = rec.reference_matrix().sum(axis=1)
            if (counts >= 2).sum() >= 10 and (counts == 1).sum() >= 10:
                two = rec.features[counts >= 2].mean()
                one = rec.features[counts == 1].mean()
                # Power summation: bounded well below an additive log-domain sum.
                assert 0.0 < two - one < 1.5
                return
        pytest.skip("no recording with enough overlapped and single-speaker frames")


class TestDeterminism:
    def test_same_arguments_give_bit_identical_features(self, tiny_data_cfg):
        a = generate_recording(tiny_data_cfg, 0, 3, pool="test")
        b = generate_recording(tiny_data_cfg, 0, 3, pool="test")
        assert np.array_equal(a.features, b.features)

    def test_same_arguments_give_identical_turns(self, tiny_data_cfg):
        a = generate_recording(tiny_data_cfg, 0, 3, pool="test")
        b = generate_recording(tiny_data_cfg, 0, 3, pool="test")
        assert a.turns == b.turns

    def test_different_index_gives_different_features(self, tiny_data_cfg):
        a = generate_recording(tiny_data_cfg, 0, 3, pool="test")
        b = generate_recording(tiny_data_cfg, 0, 4, pool="test")
        assert not np.array_equal(a.features, b.features)

    def test_different_seed_gives_different_features(self, tiny_data_cfg):
        a = generate_recording(tiny_data_cfg, 0, 3, pool="test")
        b = generate_recording(tiny_data_cfg, 1, 3, pool="test")
        assert not np.array_equal(a.features, b.features)

    def test_noise_level_does_not_perturb_the_turn_structure(self, tiny_data_cfg):
        """Turns and acoustics use separate RNG streams, so a difficulty sweep is a
        controlled experiment rather than a new dataset."""
        loud = DataConfig(**{**tiny_data_cfg.__dict__, "noise_sd": 0.1})
        quiet = DataConfig(**{**tiny_data_cfg.__dict__, "noise_sd": 2.0})
        assert generate_recording(loud, 0, 5, "test").turns == generate_recording(
            quiet, 0, 5, "test"
        ).turns

    def test_speaker_timbre_is_addressed_by_name_not_call_order(self, tiny_data_cfg):
        """rng_for(0, pool, id) means speaker 7 sounds the same regardless of how
        many recordings preceded it."""
        world = get_world(tiny_data_cfg)
        first = world.speaker_timbre("test", 7).copy()
        for i in range(5):
            world.speaker_timbre("test", i)
        assert np.array_equal(world.speaker_timbre("test", 7), first)

    def test_train_and_test_pools_give_different_voices(self, tiny_data_cfg):
        world = get_world(tiny_data_cfg)
        assert not np.array_equal(
            world.speaker_timbre("train", 3), world.speaker_timbre("test", 3)
        )

    def test_world_is_cached_by_shape(self, tiny_data_cfg):
        assert get_world(tiny_data_cfg) is get_world(tiny_data_cfg)

    def test_world_basis_is_split_independent(self, tiny_data_cfg):
        """The basis is a property of the world, so train and test speakers inhabit
        the same feature geometry."""
        world = get_world(tiny_data_cfg)
        assert world.speaker_basis.shape == (tiny_data_cfg.n_features, 8)
        assert world.phone_templates.shape == (tiny_data_cfg.n_phones,
                                               tiny_data_cfg.n_features)


class TestSplits:
    def test_dev_and_test_are_disjoint_recordings(self, tiny_data_cfg):
        dev = {r.name for r in generate_split(tiny_data_cfg, 0, "dev")}
        test = {r.name for r in generate_split(tiny_data_cfg, 0, "test")}
        assert dev.isdisjoint(test)

    def test_dev_and_test_features_differ(self, tiny_data_cfg):
        dev = generate_split(tiny_data_cfg, 0, "dev")[0]
        test = generate_split(tiny_data_cfg, 0, "test")[0]
        assert not np.array_equal(dev.features, test.features)

    def test_split_sizes_follow_the_config(self, tiny_data_cfg):
        assert len(generate_split(tiny_data_cfg, 0, "dev")) == tiny_data_cfg.n_dev_recordings
        assert len(generate_split(tiny_data_cfg, 0, "test")) == tiny_data_cfg.n_test_recordings

    def test_unknown_split_raises(self, tiny_data_cfg):
        with pytest.raises(ValueError, match="dev.*test"):
            generate_split(tiny_data_cfg, 0, "train")

    def test_split_is_deterministic(self, tiny_data_cfg):
        a = generate_split(tiny_data_cfg, 0, "test")
        b = generate_split(tiny_data_cfg, 0, "test")
        assert all(np.array_equal(x.features, y.features) for x, y in zip(a, b))


class TestTrainingSegments:
    def test_batch_shape(self, tiny_data_cfg):
        feats, labels = sample_training_batch(tiny_data_cfg, 0, 0, 4, 3, 40)
        assert feats.shape == (12, 40, tiny_data_cfg.n_features)
        assert labels.shape == (12,)

    def test_labels_are_batch_local_and_grouped(self, tiny_data_cfg):
        _, labels = sample_training_batch(tiny_data_cfg, 0, 0, 4, 3, 40)
        assert labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    def test_batch_is_deterministic_in_the_step(self, tiny_data_cfg):
        a, _ = sample_training_batch(tiny_data_cfg, 0, 7, 4, 2, 40)
        b, _ = sample_training_batch(tiny_data_cfg, 0, 7, 4, 2, 40)
        assert np.array_equal(a, b)

    def test_different_steps_give_different_batches(self, tiny_data_cfg):
        a, _ = sample_training_batch(tiny_data_cfg, 0, 7, 4, 2, 40)
        b, _ = sample_training_batch(tiny_data_cfg, 0, 8, 4, 2, 40)
        assert not np.array_equal(a, b)

    def test_pool_larger_than_requested_speakers_is_respected(self, tiny_data_cfg):
        feats, labels = sample_training_batch(tiny_data_cfg, 0, 0, 999, 2, 20)
        assert len(set(labels.tolist())) == tiny_data_cfg.n_train_speakers

    def test_segments_of_one_speaker_have_different_channels(self, tiny_data_cfg):
        """If two segments of the same speaker shared a channel offset, encoding the
        channel would be the cheapest solution to the metric-learning objective and
        the embedder would collapse on real recordings."""
        a = sample_segment(tiny_data_cfg, 0, "train", 3, 0, 60)
        b = sample_segment(tiny_data_cfg, 0, "train", 3, 1, 60)
        assert not np.allclose(a.mean(axis=0), b.mean(axis=0), atol=0.05)

    def test_segment_dtype_and_shape(self, tiny_data_cfg):
        seg = sample_segment(tiny_data_cfg, 0, "train", 0, 0, 33)
        assert seg.shape == (33, tiny_data_cfg.n_features)
        assert seg.dtype == np.float32

    def test_segment_is_deterministic(self, tiny_data_cfg):
        a = sample_segment(tiny_data_cfg, 0, "train", 5, 2, 40)
        b = sample_segment(tiny_data_cfg, 0, "train", 5, 2, 40)
        assert np.array_equal(a, b)

    def test_segments_match_the_recording_distribution(self, tiny_data_cfg):
        """A training segment and a single-speaker test window must be draws from
        the same distribution, or a model trained on one cannot be evaluated on
        the other. Compare per-channel means."""
        segs = np.concatenate(
            [sample_segment(tiny_data_cfg, 0, "test", i, 0, 100) for i in range(8)]
        )
        rec = generate_recording(tiny_data_cfg, 0, 0, pool="test")
        single = rec.reference_matrix().sum(axis=1) == 1
        assert segs.mean() == pytest.approx(rec.features[single].mean(), abs=0.6)


class TestFeatureStatistics:
    def test_shapes(self, tiny_data_cfg):
        mean, sd = feature_statistics(tiny_data_cfg, 0, n_segments=8)
        assert mean.shape == sd.shape == (tiny_data_cfg.n_features,)

    def test_sd_is_positive(self, tiny_data_cfg):
        _, sd = feature_statistics(tiny_data_cfg, 0, n_segments=8)
        assert (sd > 0).all()

    def test_is_deterministic(self, tiny_data_cfg):
        a = feature_statistics(tiny_data_cfg, 0, n_segments=8)
        b = feature_statistics(tiny_data_cfg, 0, n_segments=8)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_uses_the_training_pool_only(self, tiny_data_cfg):
        """Statistics estimated from test data would be a causality leak dressed up
        as normalisation. Different seeds must give different statistics, which
        they only do because they resample the training pool."""
        a, _ = feature_statistics(tiny_data_cfg, 0, n_segments=8)
        b, _ = feature_statistics(tiny_data_cfg, 1, n_segments=8)
        assert not np.array_equal(a, b)


class TestDatasetStats:
    def test_reports_the_expected_keys(self, tiny_recordings):
        stats = dataset_stats(tiny_recordings)
        for key in ["n_recordings", "total_hours", "mean_speakers", "min_speakers",
                    "max_speakers", "mean_turns_per_recording", "median_turn_s",
                    "mean_overlap_fraction", "mean_speech_fraction"]:
            assert key in stats

    def test_counts_match(self, tiny_recordings):
        assert dataset_stats(tiny_recordings)["n_recordings"] == len(tiny_recordings)

    def test_empty_list_gives_an_empty_dict(self):
        assert dataset_stats([]) == {}

    def test_min_max_bracket_the_mean(self, tiny_recordings):
        s = dataset_stats(tiny_recordings)
        assert s["min_speakers"] <= s["mean_speakers"] <= s["max_speakers"]
