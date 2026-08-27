"""The offline reference baselines: AHC, k-means, spectral clustering, eigengap.

The baselines have to be *strong* for the latency comparison to mean anything, so
these tests check they actually recover known cluster structure (purity 1.0 on
well-separated data) rather than merely running. They also pin the properties the
comparison relies on: determinism, and the fact that offline emission is the
recording's final frame.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from streamdiar.config import replace
from streamdiar.models.offline import (
    _densify,
    agglomerative_average_linkage,
    cosine_distance_matrix,
    diarize_offline,
    estimate_n_speakers_eigengap,
    kmeans,
    refine_affinity,
    spectral_cluster,
)


def purity(pred: np.ndarray, truth: np.ndarray) -> float:
    return sum(max(Counter(truth[pred == c]).values()) for c in set(pred.tolist())) / len(pred)


class TestCosineDistance:
    def test_diagonal_is_zero(self, three_clusters):
        x, _ = three_clusters
        assert np.abs(cosine_distance_matrix(x).diagonal()).max() < 1e-12

    def test_is_symmetric(self, three_clusters):
        x, _ = three_clusters
        d = cosine_distance_matrix(x)
        assert np.abs(d - d.T).max() == 0.0

    def test_is_non_negative(self, three_clusters):
        x, _ = three_clusters
        assert cosine_distance_matrix(x).min() >= 0.0

    def test_orthogonal_vectors_are_distance_one(self):
        x = np.eye(3)
        assert cosine_distance_matrix(x)[0, 1] == pytest.approx(1.0)

    def test_opposite_vectors_are_distance_two(self):
        x = np.array([[1.0, 0.0], [-1.0, 0.0]])
        assert cosine_distance_matrix(x)[0, 1] == pytest.approx(2.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match=r"\(n, d\)"):
            cosine_distance_matrix(np.zeros(5))


class TestAhc:
    def test_recovers_three_well_separated_clusters(self, three_clusters):
        x, truth = three_clusters
        labels = agglomerative_average_linkage(cosine_distance_matrix(x), 0.22)
        assert labels.max() + 1 == 3
        assert purity(labels, truth) == 1.0

    def test_threshold_zero_leaves_every_point_alone(self, three_clusters):
        x, _ = three_clusters
        labels = agglomerative_average_linkage(cosine_distance_matrix(x), -1.0)
        assert labels.max() + 1 == len(x)

    def test_large_threshold_merges_everything(self, three_clusters):
        x, _ = three_clusters
        labels = agglomerative_average_linkage(cosine_distance_matrix(x), 5.0)
        assert labels.max() + 1 == 1

    def test_lower_threshold_gives_at_least_as_many_clusters(self, three_clusters):
        x, _ = three_clusters
        d = cosine_distance_matrix(x)
        counts = [agglomerative_average_linkage(d, t).max() + 1
                  for t in [0.05, 0.15, 0.25, 0.4, 0.6]]
        assert counts == sorted(counts, reverse=True)

    def test_max_clusters_forces_merging_past_the_threshold(self, three_clusters):
        x, _ = three_clusters
        d = cosine_distance_matrix(x)
        assert agglomerative_average_linkage(d, 0.01, max_clusters=2).max() + 1 == 2

    def test_max_clusters_of_one(self, three_clusters):
        x, _ = three_clusters
        d = cosine_distance_matrix(x)
        assert agglomerative_average_linkage(d, 0.01, max_clusters=1).max() + 1 == 1

    def test_labels_are_dense_and_zero_based(self, three_clusters):
        x, _ = three_clusters
        labels = agglomerative_average_linkage(cosine_distance_matrix(x), 0.22)
        assert set(labels.tolist()) == set(range(labels.max() + 1))

    def test_empty_input(self):
        assert agglomerative_average_linkage(np.zeros((0, 0)), 0.5).shape == (0,)

    def test_single_point(self):
        assert agglomerative_average_linkage(np.zeros((1, 1)), 0.5).tolist() == [0]

    def test_two_identical_points_merge(self):
        assert agglomerative_average_linkage(np.zeros((2, 2)), 0.5).tolist() == [0, 0]

    def test_is_deterministic(self, three_clusters):
        x, _ = three_clusters
        d = cosine_distance_matrix(x)
        assert np.array_equal(
            agglomerative_average_linkage(d, 0.22), agglomerative_average_linkage(d, 0.22)
        )

    def test_does_not_mutate_its_input(self, three_clusters):
        x, _ = three_clusters
        d = cosine_distance_matrix(x)
        before = d.copy()
        agglomerative_average_linkage(d, 0.22)
        assert np.array_equal(d, before)


class TestDensify:
    def test_relabels_by_first_appearance(self):
        assert _densify(np.array([7, 7, 3, 3, 7])).tolist() == [0, 0, 1, 1, 0]

    def test_already_dense_is_unchanged(self):
        assert _densify(np.array([0, 1, 2])).tolist() == [0, 1, 2]

    def test_empty(self):
        assert _densify(np.array([], dtype=int)).shape == (0,)


class TestKmeans:
    def test_recovers_three_clusters(self, three_clusters):
        x, truth = three_clusters
        assert purity(kmeans(x, 3, seed=0), truth) == 1.0

    def test_k_is_capped_at_n(self):
        assert len(set(kmeans(np.random.rand(3, 4), 10, seed=0).tolist())) <= 3

    def test_k_of_one_puts_everything_together(self, three_clusters):
        x, _ = three_clusters
        assert set(kmeans(x, 1, seed=0).tolist()) == {0}

    def test_empty_input(self):
        assert kmeans(np.zeros((0, 3)), 2).shape == (0,)

    def test_single_point(self):
        assert kmeans(np.zeros((1, 3)), 3).tolist() == [0]

    def test_is_deterministic_given_the_seed(self, three_clusters):
        x, _ = three_clusters
        assert np.array_equal(kmeans(x, 3, seed=1), kmeans(x, 3, seed=1))

    def test_more_restarts_never_increase_inertia(self, three_clusters):
        """Best-of-N by inertia is monotone in N by construction."""
        x, _ = three_clusters

        def inertia(labels):
            total = 0.0
            for c in set(labels.tolist()):
                pts = x[labels == c]
                total += float(((pts - pts.mean(axis=0)) ** 2).sum())
            return total

        few = inertia(kmeans(x, 3, seed=0, n_restarts=1))
        many = inertia(kmeans(x, 3, seed=0, n_restarts=8))
        assert many <= few + 1e-9

    def test_all_clusters_are_populated(self, three_clusters):
        """An empty cluster silently reduces the effective k and makes the eigengap's
        speaker count a lie, so empties are re-seeded."""
        x, _ = three_clusters
        labels = kmeans(x, 5, seed=0)
        assert len(set(labels.tolist())) == 5


class TestAffinityRefinement:
    def test_diagonal_is_zeroed(self, three_clusters):
        x, _ = three_clusters
        assert refine_affinity(x, 0.9).diagonal().max() == 0.0

    def test_is_symmetric(self, three_clusters):
        x, _ = three_clusters
        a = refine_affinity(x, 0.9)
        assert np.abs(a - a.T).max() == 0.0

    def test_is_non_negative(self, three_clusters):
        x, _ = three_clusters
        assert refine_affinity(x, 0.9).min() >= 0.0

    def test_zero_percentile_keeps_the_dense_affinity(self, three_clusters):
        x, _ = three_clusters
        dense = refine_affinity(x, 0.0)
        sparse = refine_affinity(x, 0.9)
        assert (dense > 0).sum() > (sparse > 0).sum()

    def test_higher_percentile_is_sparser(self, three_clusters):
        x, _ = three_clusters
        counts = [(refine_affinity(x, p) > 0).sum() for p in [0.5, 0.7, 0.9, 0.95]]
        assert counts == sorted(counts, reverse=True)

    def test_tiny_input_skips_refinement(self):
        x = np.eye(2)
        assert refine_affinity(x, 0.9).shape == (2, 2)


class TestEigengap:
    def test_recovers_three_from_three_clusters(self, three_clusters):
        x, _ = three_clusters
        k, _ = estimate_n_speakers_eigengap(refine_affinity(x, 0.9), 10)
        assert k == 3

    def test_returns_ascending_eigenvalues(self, three_clusters):
        x, _ = three_clusters
        _, vals = estimate_n_speakers_eigengap(refine_affinity(x, 0.9), 10)
        assert np.all(np.diff(vals) >= -1e-9)

    def test_three_components_give_three_near_zero_eigenvalues(self, three_clusters):
        x, _ = three_clusters
        _, vals = estimate_n_speakers_eigengap(refine_affinity(x, 0.9), 10)
        assert np.abs(vals[:3]).max() < 1e-6
        assert vals[3] > 0.1

    def test_respects_max_k(self, three_clusters):
        x, _ = three_clusters
        k, _ = estimate_n_speakers_eigengap(refine_affinity(x, 0.9), 2)
        assert k <= 2

    def test_single_point(self):
        k, _ = estimate_n_speakers_eigengap(np.zeros((1, 1)), 5)
        assert k == 1

    def test_empty_affinity(self):
        k, _ = estimate_n_speakers_eigengap(np.zeros((0, 0)), 5)
        assert k == 1
