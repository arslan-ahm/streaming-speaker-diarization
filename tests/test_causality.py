"""Causality, asserted by perturbation and demanded to be **bit-identical**.

This is the file that makes the word "online" mean something. The test is always
the same shape: run the system on features ``X``, then on ``X'`` which is
identical up to frame ``t0`` and pure noise afterwards, and require that
everything the system committed to at or before ``t0`` is unchanged — not close,
*equal*.

Three separate places can leak the future, and each gets its own test:

1. the convolution stack (centred padding),
2. the VAD state machine (a hangover looking forward instead of back),
3. the diarizer's decision timing (a pending decision drifting past its deadline
   across a silence and buying unbudgeted lookahead — a real bug during
   development, which is why ``test_decisions_do_not_drift_past_the_deadline``
   exists).

``test_noncausal_ablation_actually_leaks`` is the control: it asserts the causal
model's clean result is not vacuous by showing the ``causal=False`` variant fails
the identical check.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from streamdiar.config import EmbedderConfig, replace
from streamdiar.engine.diarize import run_diarizer
from streamdiar.models.embedder import build_embedder
from streamdiar.models.online import diarize_online
from streamdiar.models.vad import energy_vad, frame_energy, windows_are_speech


@pytest.fixture
def perturbed_pair(tiny_data_cfg) -> tuple[np.ndarray, np.ndarray, int]:
    """``(X, X', t0)``: identical up to ``t0``, independent noise after it."""
    rng = np.random.RandomState(0)
    n, f = 800, tiny_data_cfg.n_features
    x = rng.randn(n, f).astype(np.float32)
    t0 = 500
    x2 = x.copy()
    x2[t0:] = np.random.RandomState(1).randn(n - t0, f)
    return x, x2, t0


class TestEmbedderCausality:
    def test_frame_features_are_bit_identical_up_to_t0(self, tiny_cfg, feature_stats,
                                                       perturbed_pair):
        x, x2, t0 = perturbed_pair
        mean, sd = feature_stats
        model = build_embedder(tiny_cfg.embedder, tiny_cfg.data.n_features, mean, sd, seed=0)
        model.eval()
        with torch.no_grad():
            a = model.frame_features(torch.from_numpy(x).unsqueeze(0))[0]
            b = model.frame_features(torch.from_numpy(x2).unsqueeze(0))[0]
        assert torch.equal(a[:t0], b[:t0])

    def test_frame_features_do_change_after_t0(self, tiny_cfg, feature_stats, perturbed_pair):
        """The test would be vacuous if the model ignored its input."""
        x, x2, t0 = perturbed_pair
        mean, sd = feature_stats
        model = build_embedder(tiny_cfg.embedder, tiny_cfg.data.n_features, mean, sd, seed=0)
        model.eval()
        with torch.no_grad():
            a = model.frame_features(torch.from_numpy(x).unsqueeze(0))[0]
            b = model.frame_features(torch.from_numpy(x2).unsqueeze(0))[0]
        assert not torch.equal(a[t0:], b[t0:])

    def test_window_embeddings_are_bit_identical_for_windows_ending_at_or_before_t0(
        self, tiny_cfg, feature_stats, perturbed_pair
    ):
        x, x2, t0 = perturbed_pair
        mean, sd = feature_stats
        model = build_embedder(tiny_cfg.embedder, tiny_cfg.data.n_features, mean, sd, seed=0)
        e1, _, ends = model.embed_recording(x, tiny_cfg.frames_per_window, tiny_cfg.frames_per_hop)
        e2, _, _ = model.embed_recording(x2, tiny_cfg.frames_per_window, tiny_cfg.frames_per_hop)
        early = ends <= t0
        assert early.sum() > 5
        assert np.array_equal(e1[early], e2[early])
        assert np.abs(e1[early] - e2[early]).max() == 0.0

    def test_prefix_sum_pooling_does_not_leak_the_future(self, tiny_cfg, feature_stats):
        """Window pooling uses prefix sums over the whole sequence; the prefix at
        index e must depend only on frames < e, including in floating point."""
        mean, sd = feature_stats
        model = build_embedder(tiny_cfg.embedder, tiny_cfg.data.n_features, mean, sd, seed=0)
        rng = np.random.RandomState(2)
        short = rng.randn(300, tiny_cfg.data.n_features).astype(np.float32)
        long = np.concatenate([short, rng.randn(500, tiny_cfg.data.n_features).astype(np.float32)])
        e_short, _, ends_short = model.embed_recording(
            short, tiny_cfg.frames_per_window, tiny_cfg.frames_per_hop
        )
        e_long, _, _ = model.embed_recording(
            long, tiny_cfg.frames_per_window, tiny_cfg.frames_per_hop
        )
        k = len(ends_short) - 1  # the last short window may be truncated; drop it
        assert np.array_equal(e_short[:k], e_long[:k])

    def test_noncausal_ablation_actually_leaks(self, tiny_cfg, feature_stats, perturbed_pair):
        """The control: centred padding must fail the very test the causal model passes,
        or the causal result proves nothing."""
        x, x2, t0 = perturbed_pair
        mean, sd = feature_stats
        cfg = replace(tiny_cfg.embedder, causal=False)
        model = build_embedder(cfg, tiny_cfg.data.n_features, mean, sd, seed=0)
        e1, _, ends = model.embed_recording(x, tiny_cfg.frames_per_window, tiny_cfg.frames_per_hop)
        e2, _, _ = model.embed_recording(x2, tiny_cfg.frames_per_window, tiny_cfg.frames_per_hop)
        early = ends <= t0
        assert not np.array_equal(e1[early], e2[early])

    def test_receptive_field_matches_the_dilations(self):
        cfg = EmbedderConfig(kernel_size=3, dilations=(1, 2, 4, 8))
        model = build_embedder(cfg, 12)
        assert model.receptive_field == 1 + 2 * (1 + 2 + 4 + 8)

    def test_receptive_field_of_a_single_block(self):
        model = build_embedder(EmbedderConfig(kernel_size=3, dilations=(1,)), 12)
        assert model.receptive_field == 3

    def test_a_frame_beyond_the_receptive_field_has_no_influence(self, tiny_cfg, feature_stats):
        """Changing frame 0 must leave a frame far past the receptive field untouched."""
        mean, sd = feature_stats
        model = build_embedder(tiny_cfg.embedder, tiny_cfg.data.n_features, mean, sd, seed=0)
        model.eval()
        rng = np.random.RandomState(3)
        x = rng.randn(200, tiny_cfg.data.n_features).astype(np.float32)
        x2 = x.copy()
        x2[0] += 50.0
        with torch.no_grad():
            a = model.frame_features(torch.from_numpy(x).unsqueeze(0))[0]
            b = model.frame_features(torch.from_numpy(x2).unsqueeze(0))[0]
        far = model.receptive_field + 5
        assert torch.equal(a[far:], b[far:])


class TestVadCausality:
    def test_vad_is_causal(self):
        """A hangover extends a decision forward in time; it must never look ahead."""
        rng = np.random.RandomState(0)
        e = rng.randn(400)
        t0 = 250
        e2 = e.copy()
        e2[t0:] = np.random.RandomState(1).randn(400 - t0)
        a = energy_vad(e, 0.0, hangover_frames=12, onset_frames=3)
        b = energy_vad(e2, 0.0, hangover_frames=12, onset_frames=3)
        assert np.array_equal(a[:t0], b[:t0])

    def test_vad_output_changes_after_the_perturbation(self):
        """Not vacuous: driving the tail firmly below threshold must close speech.

        Note the direction matters. Setting the tail *above* threshold changes
        nothing, because random energy around a threshold of 0 already keeps the
        hangover state in speech -- which is itself a useful reminder that this
        VAD is sticky by design.
        """
        rng = np.random.RandomState(0)
        e = rng.randn(400)
        e2 = e.copy()
        e2[250:] = -10.0
        a = energy_vad(e, 0.0)
        b = energy_vad(e2, 0.0)
        assert not np.array_equal(a[250:], b[250:])
        assert not b[275:].any()

    def test_onset_requires_consecutive_active_frames(self):
        """A single spike must not open speech when onset_frames is 3."""
        e = np.zeros(20)
        e[5] = 10.0
        assert not energy_vad(e, 1.0, hangover_frames=0, onset_frames=3).any()

    def test_onset_opens_after_enough_consecutive_frames(self):
        e = np.zeros(20)
        e[5:9] = 10.0
        out = energy_vad(e, 1.0, hangover_frames=0, onset_frames=3)
        assert not out[6] and out[7]

    def test_hangover_extends_speech_forward(self):
        e = np.zeros(20)
        e[2:8] = 10.0
        out = energy_vad(e, 1.0, hangover_frames=4, onset_frames=1)
        assert out[8] and out[9]

    def test_hangover_eventually_closes(self):
        e = np.zeros(30)
        e[2:8] = 10.0
        assert not energy_vad(e, 1.0, hangover_frames=4, onset_frames=1)[-1]

    def test_all_silence_gives_no_speech(self):
        assert not energy_vad(np.full(50, -5.0), 0.0).any()

    def test_all_speech_gives_all_speech_after_the_onset(self):
        out = energy_vad(np.full(50, 5.0), 0.0, onset_frames=3)
        assert out[3:].all()

    def test_empty_input_gives_empty_output(self):
        assert energy_vad(np.zeros(0), 0.0).shape == (0,)

    def test_frame_energy_uses_training_statistics(self, tiny_data_cfg, feature_stats):
        """Standardising by the recording's own statistics would make every frame
        depend on the whole recording."""
        mean, sd = feature_stats
        x = np.zeros((10, tiny_data_cfg.n_features), dtype=np.float32)
        expected = float(((0.0 - mean) / sd).mean())
        assert frame_energy(x, mean, sd)[0] == pytest.approx(expected)

    def test_windows_are_speech_uses_the_region_not_the_window(self):
        """Using the whole embedding window would produce a false-alarm tail of
        (window - hop) frames after every turn."""
        speech = np.zeros(100, dtype=bool)
        speech[:50] = True
        region_starts = np.array([50, 75])
        region_ends = np.array([75, 100])
        assert windows_are_speech(speech, region_starts, region_ends).tolist() == [False, False]

    def test_windows_are_speech_threshold(self):
        speech = np.zeros(20, dtype=bool)
        speech[:6] = True
        # Region [0,10) is 60% speech -> True at the 0.5 threshold.
        assert windows_are_speech(speech, np.array([0]), np.array([10]))[0]
        assert not windows_are_speech(speech, np.array([0]), np.array([20]))[0]

    def test_empty_region_is_not_speech(self):
        assert not windows_are_speech(np.ones(10, dtype=bool), np.array([5]), np.array([5]))[0]


class TestDiarizerCausality:
    @staticmethod
    def _embeddings(n_windows: int, dim: int = 8, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        e = rng.normal(size=(n_windows, dim))
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def test_labels_are_bit_identical_when_the_future_changes(self, tiny_cfg):
        """The heart of the online claim: a decision emitted at or before t0 cannot
        depend on any embedding that arrives after t0."""
        n, hop = 60, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        speech = np.ones(n, dtype=bool)
        budget = 2

        e1 = self._embeddings(n, seed=0)
        e2 = e1.copy()
        t0_window = 30
        e2[t0_window + budget + 1 :] = self._embeddings(n - t0_window - budget - 1, seed=99)

        a = diarize_online(e1, starts, ends, speech, tiny_cfg.diarizer, budget, ends[-1], hop)
        b = diarize_online(e2, starts, ends, speech, tiny_cfg.diarizer, budget, ends[-1], hop)
        # Decisions emitted at or before window t0_window + budget.
        k = t0_window + 1
        assert np.array_equal(a.labels[:k], b.labels[:k])
        assert np.array_equal(a.confidence[:k], b.confidence[:k])
        assert np.array_equal(a.emission_frames[:k], b.emission_frames[:k])

    def test_later_decisions_do_change(self, tiny_cfg):
        n, hop = 60, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        speech = np.ones(n, dtype=bool)
        e1 = self._embeddings(n, seed=0)
        e2 = e1.copy()
        e2[35:] = self._embeddings(n - 35, seed=99)
        a = diarize_online(e1, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], hop)
        b = diarize_online(e2, starts, ends, speech, tiny_cfg.diarizer, 2, ends[-1], hop)
        assert not np.array_equal(a.labels, b.labels)

    def test_zero_budget_gives_exactly_zero_delay(self, tiny_cfg):
        n, hop = 30, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        out = diarize_online(
            self._embeddings(n), starts, ends, np.ones(n, dtype=bool),
            tiny_cfg.diarizer, 0, ends[-1], hop,
        )
        assert np.array_equal(out.emission_frames, ends)

    def test_emission_never_precedes_the_window_end(self, tiny_cfg):
        """A decision emitted before its own audio arrived would be a causality
        violation; metrics/latency.py raises on it, so this must hold for every budget."""
        n, hop = 40, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        rng = np.random.default_rng(0)
        speech = rng.random(n) < 0.8
        for budget in [0, 1, 2, 4, 8, 100]:
            out = diarize_online(
                self._embeddings(n), starts, ends, speech, tiny_cfg.diarizer,
                budget, ends[-1], hop,
            )
            assert (out.emission_frames >= out.window_ends).all(), budget

    def test_delay_never_exceeds_the_budget(self, tiny_cfg):
        """Window k is emitted when window k+budget arrives, so the delay is exactly
        budget*hop except in the tail, where it is smaller."""
        n, hop = 40, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        speech = np.ones(n, dtype=bool)
        for budget in [0, 1, 3, 7]:
            out = diarize_online(
                self._embeddings(n), starts, ends, speech, tiny_cfg.diarizer,
                budget, ends[-1], hop,
            )
            delay = out.emission_frames - out.window_ends
            assert delay.max() <= budget * hop

    def test_delay_equals_the_budget_away_from_the_tail(self, tiny_cfg):
        n, hop = 40, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        speech = np.ones(n, dtype=bool)
        budget = 3
        out = diarize_online(
            self._embeddings(n), starts, ends, speech, tiny_cfg.diarizer,
            budget, ends[-1], hop,
        )
        delay = out.emission_frames - out.window_ends
        assert (delay[: n - budget] == budget * hop).all()

    def test_decisions_do_not_drift_past_the_deadline(self, tiny_cfg):
        """The deadline check runs on every window, speech or not. Checking only on
        speech windows let a pending decision drift across a silence and buy
        unbudgeted lookahead -- a real bug, and it made the B=0 point impossible."""
        n, hop = 40, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        speech = np.ones(n, dtype=bool)
        speech[10:25] = False  # a long silence right after a speech window
        budget = 2
        out = diarize_online(
            self._embeddings(n), starts, ends, speech, tiny_cfg.diarizer,
            budget, ends[-1], hop,
        )
        spoke = out.labels >= 0
        delay = (out.emission_frames - out.window_ends)[spoke]
        assert delay.max() <= budget * hop

    def test_non_speech_windows_emit_immediately(self, tiny_cfg):
        n, hop = 20, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        speech = np.ones(n, dtype=bool)
        speech[5:10] = False
        out = diarize_online(
            self._embeddings(n), starts, ends, speech, tiny_cfg.diarizer, 3, ends[-1], hop
        )
        assert (out.emission_frames[5:10] == ends[5:10]).all()
        assert (out.labels[5:10] == -1).all()

    def test_tail_windows_flush_at_the_final_frame(self, tiny_cfg):
        n, hop = 20, tiny_cfg.frames_per_hop
        ends = (np.arange(n) + 1) * hop
        starts = np.maximum(0, ends - tiny_cfg.frames_per_window)
        out = diarize_online(
            self._embeddings(n), starts, ends, np.ones(n, dtype=bool),
            tiny_cfg.diarizer, 5, ends[-1], hop,
        )
        assert out.emission_frames[-1] == ends[-1]
        assert out.emission_frames.max() == ends[-1]


class TestEndToEndCausality:
    def test_full_pipeline_decisions_are_stable_under_future_perturbation(
        self, untrained_model, tiny_cfg, one_recording
    ):
        """The whole pipeline -- features, VAD, embedder, diarizer -- run twice on a
        recording whose second half is replaced by noise."""
        import copy

        rec_a = one_recording
        rec_b = copy.deepcopy(one_recording)
        t0 = rec_b.n_frames // 2
        rng = np.random.RandomState(7)
        feats = rec_b.features.copy()
        feats[t0:] = rng.randn(rec_b.n_frames - t0, feats.shape[1]).astype(np.float32) * 2.0
        rec_b.features = feats

        out_a, _ = run_diarizer(untrained_model, tiny_cfg, rec_a, method="online")
        out_b, _ = run_diarizer(untrained_model, tiny_cfg, rec_b, method="online")

        budget_frames = tiny_cfg.budget_windows * tiny_cfg.frames_per_hop
        safe = out_a.emission_frames <= t0 - budget_frames
        assert safe.sum() > 2
        assert np.array_equal(out_a.labels[safe], out_b.labels[safe])

    def test_offline_is_not_causal_and_the_test_shows_it(
        self, untrained_model, tiny_cfg, one_recording
    ):
        """The offline reference must fail the same check -- it is the whole point of
        the comparison. Its emission frame is the recording's last frame."""
        import copy

        rec_b = copy.deepcopy(one_recording)
        t0 = rec_b.n_frames // 2
        rng = np.random.RandomState(8)
        feats = rec_b.features.copy()
        feats[t0:] = rng.randn(rec_b.n_frames - t0, feats.shape[1]).astype(np.float32) * 2.0
        rec_b.features = feats

        a, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, method="offline_ahc")
        b, _ = run_diarizer(untrained_model, tiny_cfg, rec_b, method="offline_ahc")
        assert (a.emission_frames == one_recording.n_frames).all()
        assert not np.array_equal(a.labels, b.labels)

    def test_repeated_runs_are_bit_identical(self, untrained_model, tiny_cfg, one_recording):
        a, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, method="online")
        b, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, method="online")
        assert np.array_equal(a.labels, b.labels)
        # Non-speech windows carry NaN confidence by design, so compare where finite.
        finite = np.isfinite(a.confidence)
        assert np.array_equal(finite, np.isfinite(b.confidence))
        assert np.abs(a.confidence[finite] - b.confidence[finite]).max() == 0.0
        assert np.array_equal(a.emission_frames, b.emission_frames)
