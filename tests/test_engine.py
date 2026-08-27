"""Engine: loss behaviour, checkpoint round-trip, scoring wiring, and the smoke path.

The end-to-end test at the bottom is the one the build standard specifically asks
for: run the real pipeline at tiny scale and assert the output table is
**populated**. The sibling project shipped a bug where the results table printed
empty, and an exit-code-zero test would not have caught it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from streamdiar.config import replace
from streamdiar.engine import (
    METHODS,
    aggregate,
    evaluate,
    fit_vad,
    load_model,
    per_recording_metric,
    pooled_calibration,
    run_diarizer,
    save_model,
    score_output,
    train_embedder,
)
from streamdiar.engine.train import PrototypicalLoss, _batch_accuracy


def _grouped(n_speakers: int, n_segments: int, dim: int, separated: bool) -> torch.Tensor:
    """Embeddings grouped speaker-major; one-hot per speaker when ``separated``."""
    if separated:
        eye = torch.eye(n_speakers)[:, None, :].repeat(1, n_segments, 1)
        pad = torch.zeros(n_speakers, n_segments, max(0, dim - n_speakers))
        out = torch.cat([eye, pad], dim=-1)
    else:
        torch.manual_seed(0)
        out = torch.randn(n_speakers, n_segments, dim)
    return torch.nn.functional.normalize(out.reshape(n_speakers * n_segments, dim), dim=-1)


class TestPrototypicalLoss:
    def test_perfectly_separated_embeddings_give_near_zero_loss(self):
        loss = PrototypicalLoss()(_grouped(6, 4, 16, True), 6, 4)
        assert float(loss.detach()) < 0.01

    def test_random_embeddings_give_loss_near_or_above_log_n(self):
        loss = float(PrototypicalLoss()(_grouped(6, 4, 16, False), 6, 4).detach())
        assert loss > np.log(6) * 0.8

    def test_batch_accuracy_is_one_when_separated(self):
        assert _batch_accuracy(_grouped(6, 4, 16, True), 6, 4) == 1.0

    def test_batch_accuracy_is_near_chance_when_random(self):
        acc = _batch_accuracy(_grouped(8, 4, 16, False), 8, 4)
        assert acc < 0.5

    def test_one_segment_per_speaker_raises(self):
        """With M=1 the self-excluded prototype does not exist, and silently using
        the inclusive one is the classic GE2E collapse bug."""
        with pytest.raises(ValueError, match="n_segments >= 2"):
            PrototypicalLoss()(_grouped(4, 1, 8, True), 4, 1)

    def test_loss_is_differentiable(self):
        emb = _grouped(4, 2, 8, False).clone().requires_grad_(True)
        PrototypicalLoss()(emb, 4, 2).backward()
        assert emb.grad is not None and torch.isfinite(emb.grad).all()

    def test_scale_and_bias_are_learnable(self):
        loss_fn = PrototypicalLoss(init_scale=10.0, init_bias=-5.0)
        names = {n for n, _ in loss_fn.named_parameters()}
        assert names == {"log_scale", "bias"}
        assert float(loss_fn.log_scale.exp().detach()) == pytest.approx(10.0, rel=1e-4)

    def test_positive_prototype_excludes_the_query_itself(self):
        """Closed form. Speaker 0 has orthogonal segments e00=(1,0), e01=(0,1);
        speaker 1 has e10=(-1,0), e11=(0,-1). With scale 1 and bias 0:

        * self-excluded prototype for e00 is normalize(e01) = (0,1), so its own
          similarity is exactly **0**;
        * the negative prototype is normalize(e10+e11) = (-1,-1)/sqrt(2), so
          sim(e00, .) = -1/sqrt(2).

        By symmetry every segment sees logits ``[0, -1/sqrt(2)]``, so the loss is
        ``log(1 + exp(-1/sqrt(2))) = 0.400834``. An *inclusive* prototype would
        make the own similarity +1/sqrt(2) and the loss 0.217706 -- lower, which
        is exactly why the bug looks like an improvement.
        """
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
        emb = torch.nn.functional.normalize(torch.cat([a, b]), dim=-1)
        loss = float(PrototypicalLoss(init_scale=1.0, init_bias=0.0)(emb, 2, 2).detach())
        expected_excluded = math.log(1.0 + math.exp(-1.0 / math.sqrt(2.0)))
        expected_inclusive = math.log(1.0 + math.exp(-2.0 / math.sqrt(2.0)))
        assert loss == pytest.approx(expected_excluded, abs=1e-5)
        assert expected_inclusive < expected_excluded


class TestTraining:
    @pytest.mark.slow
    def test_loss_decreases_over_a_short_run(self, tiny_cfg):
        cfg = replace(tiny_cfg, train=replace(tiny_cfg.train, steps=40, log_every=5))
        model = train_embedder(cfg, seed=0)
        losses = [h["loss"] for h in model.history]
        assert np.mean(losses[-2:]) < np.mean(losses[:2])

    def test_history_is_recorded(self, tiny_cfg):
        model = train_embedder(tiny_cfg, seed=0)
        assert len(model.history) >= 1
        for key in ["step", "loss", "batch_accuracy", "grad_norm", "lr", "elapsed_s"]:
            assert key in model.history[0]

    def test_parameter_count_is_recorded(self, tiny_cfg):
        model = train_embedder(tiny_cfg, seed=0)
        assert model.n_parameters == sum(p.numel() for p in model.embedder.parameters())

    def test_seed_is_recorded(self, tiny_cfg):
        assert train_embedder(tiny_cfg, seed=7).seed == 7

    def test_model_is_left_in_eval_mode(self, tiny_cfg):
        assert not train_embedder(tiny_cfg, seed=0).embedder.training

    def test_same_seed_gives_identical_weights(self, tiny_cfg):
        a = train_embedder(tiny_cfg, seed=3)
        b = train_embedder(tiny_cfg, seed=3)
        for pa, pb in zip(a.embedder.parameters(), b.embedder.parameters()):
            assert torch.equal(pa, pb)

    def test_different_seeds_give_different_weights(self, tiny_cfg):
        a = train_embedder(tiny_cfg, seed=3)
        b = train_embedder(tiny_cfg, seed=4)
        diffs = [not torch.equal(pa, pb)
                 for pa, pb in zip(a.embedder.parameters(), b.embedder.parameters())]
        assert any(diffs)

    def test_history_is_written_to_jsonl(self, tiny_cfg, tmp_path):
        path = tmp_path / "history.jsonl"
        train_embedder(tiny_cfg, seed=0, history_path=str(path))
        assert path.exists()
        assert b"\r\n" not in path.read_bytes()


class TestVadFitting:
    def test_threshold_is_set_and_error_reported(self, untrained_model, tiny_cfg,
                                                 tiny_recordings):
        model = fit_vad(untrained_model, tiny_cfg, list(tiny_recordings))
        assert np.isfinite(model.vad_threshold)
        assert 0.0 <= model.vad_frame_error <= 1.0

    def test_fitted_threshold_beats_an_absurd_one(self, untrained_model, tiny_cfg,
                                                  tiny_recordings):
        from streamdiar.models.vad import energy_vad, frame_energy

        model = fit_vad(untrained_model, tiny_cfg, list(tiny_recordings))
        rec = tiny_recordings[0]
        energy = frame_energy(rec.features, model.feature_mean, model.feature_sd)
        truth = rec.reference_matrix().any(axis=1)
        fitted_err = (energy_vad(energy, model.vad_threshold) != truth).mean()
        silly_err = (energy_vad(energy, 100.0) != truth).mean()
        assert fitted_err <= silly_err

    def test_empty_dev_split_does_not_crash(self, untrained_model, tiny_cfg):
        model = fit_vad(untrained_model, tiny_cfg, [])
        assert np.isfinite(model.vad_threshold)


class TestCheckpoint:
    def test_round_trip_preserves_weights(self, untrained_model, tiny_cfg, tmp_path):
        path = save_model(untrained_model, tmp_path / "m.pt")
        again = load_model(path, tiny_cfg)
        for a, b in zip(untrained_model.embedder.parameters(), again.embedder.parameters()):
            assert torch.equal(a, b)

    def test_round_trip_preserves_the_vad_threshold(self, untrained_model, tiny_cfg, tmp_path):
        """A checkpoint that dropped this would load without error and score several
        DER points worse -- the worst kind of bug."""
        untrained_model.vad_threshold = -0.777
        path = save_model(untrained_model, tmp_path / "m.pt")
        assert load_model(path, tiny_cfg).vad_threshold == pytest.approx(-0.777)

    def test_round_trip_preserves_feature_statistics(self, untrained_model, tiny_cfg, tmp_path):
        path = save_model(untrained_model, tmp_path / "m.pt")
        again = load_model(path, tiny_cfg)
        assert np.array_equal(untrained_model.feature_mean, again.feature_mean)
        assert np.array_equal(untrained_model.feature_sd, again.feature_sd)

    def test_loaded_model_is_in_eval_mode(self, untrained_model, tiny_cfg, tmp_path):
        path = save_model(untrained_model, tmp_path / "m.pt")
        assert not load_model(path, tiny_cfg).embedder.training

    def test_loaded_model_gives_identical_embeddings(self, untrained_model, tiny_cfg,
                                                     one_recording, tmp_path):
        path = save_model(untrained_model, tmp_path / "m.pt")
        again = load_model(path, tiny_cfg)
        a, _, _ = untrained_model.embedder.embed_recording(
            one_recording.features, tiny_cfg.frames_per_window, tiny_cfg.frames_per_hop
        )
        b, _, _ = again.embedder.embed_recording(
            one_recording.features, tiny_cfg.frames_per_window, tiny_cfg.frames_per_hop
        )
        assert np.array_equal(a, b)

    def test_creates_parent_directories(self, untrained_model, tmp_path):
        path = save_model(untrained_model, tmp_path / "a" / "b" / "m.pt")
        assert path.exists()


class TestRunDiarizer:
    def test_all_methods_run(self, untrained_model, tiny_cfg, one_recording):
        for method in METHODS:
            out, timing = run_diarizer(untrained_model, tiny_cfg, one_recording, method=method)
            assert out.method == method
            assert timing["total_s"] > 0.0

    def test_unknown_method_raises(self, untrained_model, tiny_cfg, one_recording):
        with pytest.raises(ValueError, match="unknown method"):
            run_diarizer(untrained_model, tiny_cfg, one_recording, method="telepathy")

    def test_timing_splits_embed_from_cluster(self, untrained_model, tiny_cfg, one_recording):
        """Reported separately because they scale differently: embedding is linear
        for every method, offline clustering is the super-linear term."""
        _, timing = run_diarizer(untrained_model, tiny_cfg, one_recording)
        assert timing["total_s"] == pytest.approx(
            timing["embed_s"] + timing["vad_s"] + timing["cluster_s"]
        )

    def test_oracle_vad_uses_the_reference(self, untrained_model, tiny_cfg, one_recording):
        out, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, oracle_vad=True)
        ref_speech = one_recording.reference_matrix().any(axis=1)
        hop = tiny_cfg.frames_per_hop
        # Every window whose region is entirely silent in the reference must be
        # labelled non-speech under an oracle VAD.
        for k, end in enumerate(out.window_ends):
            if not ref_speech[max(0, int(end) - hop) : int(end)].any():
                assert out.labels[k] == -1

    def test_oracle_count_caps_the_speaker_count(self, untrained_model, tiny_cfg,
                                                one_recording):
        out, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, oracle_count=True)
        assert out.n_speakers <= one_recording.n_speakers

    def test_budget_override_changes_the_measured_latency(self, untrained_model, tiny_cfg,
                                                          one_recording):
        a, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, latency_budget_ms=0.0)
        b, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, latency_budget_ms=1000.0)
        da = (a.emission_frames - a.window_ends)[a.labels >= 0]
        db = (b.emission_frames - b.window_ends)[b.labels >= 0]
        assert da.max() == 0
        assert db.max() > 0


class TestScoreOutput:
    def test_reports_the_expected_metrics(self, untrained_model, tiny_cfg, one_recording):
        out, timing = run_diarizer(untrained_model, tiny_cfg, one_recording)
        res = score_output(one_recording, out, tiny_cfg, timing)
        for key in ["der", "miss", "false_alarm", "confusion", "jer", "der_no_overlap",
                    "der_collar250ms", "overlap_miss_floor", "n_speakers_true",
                    "n_speakers_pred", "latency_median_ms", "turn_accuracy", "rtf"]:
            assert key in res.metrics

    def test_der_is_a_finite_rate(self, untrained_model, tiny_cfg, one_recording):
        out, _ = run_diarizer(untrained_model, tiny_cfg, one_recording)
        res = score_output(one_recording, out, tiny_cfg)
        assert 0.0 <= res.metrics["der"] < 5.0

    def test_true_speaker_count_matches_the_recording(self, untrained_model, tiny_cfg,
                                                     one_recording):
        out, _ = run_diarizer(untrained_model, tiny_cfg, one_recording)
        res = score_output(one_recording, out, tiny_cfg)
        assert res.metrics["n_speakers_true"] == one_recording.n_speakers

    def test_calibration_arrays_are_aligned(self, untrained_model, tiny_cfg, one_recording):
        out, _ = run_diarizer(untrained_model, tiny_cfg, one_recording)
        res = score_output(one_recording, out, tiny_cfg)
        assert len(res.confidence) == len(res.margin) == len(res.correct)

    def test_excluded_decisions_are_counted_not_hidden(self, untrained_model, tiny_cfg,
                                                       one_recording):
        """Overlapped and silent regions are excluded from the calibration set; the
        count has to be visible."""
        out, _ = run_diarizer(untrained_model, tiny_cfg, one_recording)
        res = score_output(one_recording, out, tiny_cfg)
        n_speech_decisions = int((out.labels >= 0).sum())
        assert len(res.correct) + res.n_excluded_decisions == n_speech_decisions

    def test_no_overlap_der_is_at_most_the_full_der(self, untrained_model, tiny_cfg,
                                                   tiny_recordings):
        """Excluding overlap removes an irreducible miss, so it can only flatter."""
        for rec in tiny_recordings:
            out, _ = run_diarizer(untrained_model, tiny_cfg, rec)
            m = score_output(rec, out, tiny_cfg).metrics
            if np.isfinite(m["der"]) and np.isfinite(m["der_no_overlap"]):
                assert m["der_no_overlap"] <= m["der"] + 1e-9

    def test_overlap_floor_is_reported(self, untrained_model, tiny_cfg, one_recording):
        out, _ = run_diarizer(untrained_model, tiny_cfg, one_recording)
        m = score_output(one_recording, out, tiny_cfg).metrics
        assert 0.0 <= m["overlap_miss_floor"] <= 1.0

    def test_offline_latency_is_much_larger_than_online(self, untrained_model, tiny_cfg,
                                                        one_recording):
        on, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, method="online")
        off, _ = run_diarizer(untrained_model, tiny_cfg, one_recording, method="offline_ahc")
        m_on = score_output(one_recording, on, tiny_cfg).metrics
        m_off = score_output(one_recording, off, tiny_cfg).metrics
        assert m_off["latency_median_ms"] > m_on["latency_median_ms"]


class TestEvaluateAndAggregate:
    def test_one_result_per_recording_in_order(self, untrained_model, tiny_cfg,
                                               tiny_recordings):
        results = evaluate(untrained_model, tiny_cfg, list(tiny_recordings))
        assert [r.recording for r in results] == [r.name for r in tiny_recordings]

    def test_aggregate_is_populated(self, untrained_model, tiny_cfg, tiny_recordings):
        agg = aggregate(evaluate(untrained_model, tiny_cfg, list(tiny_recordings)))
        assert agg["n_recordings"] == len(tiny_recordings)
        assert np.isfinite(agg["der"])

    def test_aggregate_of_nothing_is_empty(self):
        assert aggregate([]) == {}

    def test_aggregate_counts_contributing_items_when_some_are_nan(self, untrained_model,
                                                                  tiny_cfg, tiny_recordings):
        """The standard requires the number of items behind each mean to be visible."""
        results = evaluate(untrained_model, tiny_cfg, list(tiny_recordings))
        results[0].metrics["der"] = float("nan")
        agg = aggregate(results)
        assert agg["n_contributing_der"] == len(results) - 1

    def test_per_recording_metric_is_a_vector(self, untrained_model, tiny_cfg,
                                              tiny_recordings):
        results = evaluate(untrained_model, tiny_cfg, list(tiny_recordings))
        vec = per_recording_metric(results, "der")
        assert vec.shape == (len(tiny_recordings),)

    def test_per_recording_metric_of_an_unknown_key_is_nan(self, untrained_model, tiny_cfg,
                                                           tiny_recordings):
        results = evaluate(untrained_model, tiny_cfg, list(tiny_recordings))
        assert np.isnan(per_recording_metric(results, "not_a_metric")).all()

    def test_pooled_calibration_concatenates(self, untrained_model, tiny_cfg,
                                             tiny_recordings):
        results = evaluate(untrained_model, tiny_cfg, list(tiny_recordings))
        conf, margin, correct, excluded = pooled_calibration(results)
        assert conf.size == margin.size == correct.size
        assert conf.size == sum(len(r.correct) for r in results)
        assert excluded == sum(r.n_excluded_decisions for r in results)

    def test_pooled_calibration_of_nothing(self):
        conf, margin, correct, excluded = pooled_calibration([])
        assert conf.size == 0 and excluded == 0

    def test_latency_budget_override_is_recorded_on_the_result(self, untrained_model,
                                                              tiny_cfg, tiny_recordings):
        results = evaluate(untrained_model, tiny_cfg, list(tiny_recordings),
                           latency_budget_ms=750.0)
        assert all(r.latency_budget_ms == 750.0 for r in results)


class TestEndToEndSmoke:
    """The build standard's explicit requirement: the real pipeline at tiny scale,
    asserting the output table is **populated** rather than merely that it ran."""

    @pytest.mark.slow
    def test_comparison_pipeline_produces_a_populated_table(self, tmp_path):
        from streamdiar.config import load_config
        from streamdiar.pipelines.comparison import (
            run_comparison,
            statistical_tests,
            summarise,
        )

        cfg = load_config("configs/smoke.yaml")
        variants = (
            ("online", "online", False, False),
            ("offline_ahc", "offline_ahc", False, False),
        )
        table = run_comparison(cfg, seeds=(0,), variants=variants,
                               out_dir=tmp_path, verbose=False)
        assert len(table) == 2 * cfg.data.n_test_recordings
        assert set(table["variant"]) == {"online", "offline_ahc"}
        assert table["der"].notna().all()
        assert (table["der"] >= 0).all()

        summary = summarise(table, out_dir=tmp_path)
        assert len(summary) == 2
        assert summary["der"].notna().all()

        tests = statistical_tests(table, summary, n_resamples=200, out_dir=tmp_path)
        assert len(tests) >= 1
        assert (tmp_path / "statistical_tests.csv").exists()
        assert "verdict" in tests.columns

    @pytest.mark.slow
    def test_latency_sweep_writes_incrementally(self, tmp_path):
        from streamdiar.config import load_config
        from streamdiar.pipelines.latency import already_done, run_latency_point

        cfg = load_config("configs/smoke.yaml")
        csv = tmp_path / "sweep.csv"
        assert not already_done(csv, 0.0, 0, "online")
        run_latency_point(cfg, 0, 0.0, csv, verbose=False)
        assert csv.exists()
        assert already_done(csv, 0.0, 0, "online")
        # A second call must skip rather than duplicate rows.
        import pandas as pd

        before = len(pd.read_csv(csv))
        run_latency_point(cfg, 0, 0.0, csv, verbose=False)
        assert len(pd.read_csv(csv)) == before

    @pytest.mark.slow
    def test_csv_output_uses_lf_endings(self, tmp_path):
        from streamdiar.config import load_config
        from streamdiar.pipelines.latency import run_latency_point

        cfg = load_config("configs/smoke.yaml")
        csv = tmp_path / "sweep.csv"
        run_latency_point(cfg, 0, 250.0, csv, verbose=False)
        assert b"\r\n" not in csv.read_bytes()
