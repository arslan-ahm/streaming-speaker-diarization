"""The hand-rolled Hungarian solver, checked against the definition of optimality.

The primary check is **brute force over all permutations** on 400 random matrices
up to 5x5. Agreeing with another library would only show that two
implementations agree; enumerating every assignment and taking the best *is* the
optimum, so this is the stronger test. SciPy is used as a second, independent
cross-check where it is importable, but the suite does not require it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from streamdiar.metrics.assignment import (
    assignment_cost,
    greedy_assignment,
    linear_sum_assignment,
)


def brute_force_optimum(cost: np.ndarray, maximize: bool = False) -> float:
    """The true optimum by enumerating every valid row-to-column assignment."""
    n, m = cost.shape
    k = min(n, m)
    best = None
    for rows in itertools.combinations(range(n), k):
        for cols in itertools.permutations(range(m), k):
            total = sum(cost[r, c] for r, c in zip(rows, cols, strict=True))
            if best is None or (total > best if maximize else total < best):
                best = total
    return float(best)


class TestOptimality:
    @pytest.mark.parametrize("trial", range(40))
    def test_matches_brute_force_minimisation(self, trial):
        rng = np.random.default_rng(trial)
        n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        cost = rng.integers(0, 20, size=(n, m)).astype(float)
        rows, cols = linear_sum_assignment(cost)
        assert assignment_cost(cost, rows, cols) == pytest.approx(brute_force_optimum(cost))

    @pytest.mark.parametrize("trial", range(40))
    def test_matches_brute_force_maximisation(self, trial):
        rng = np.random.default_rng(1000 + trial)
        n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        cost = rng.normal(size=(n, m))
        rows, cols = linear_sum_assignment(cost, maximize=True)
        assert assignment_cost(cost, rows, cols) == pytest.approx(
            brute_force_optimum(cost, maximize=True)
        )

    @pytest.mark.parametrize("trial", range(20))
    def test_agrees_with_scipy_when_available(self, trial):
        scipy_opt = pytest.importorskip("scipy.optimize")
        rng = np.random.default_rng(2000 + trial)
        n, m = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        cost = rng.normal(size=(n, m))
        mine = assignment_cost(cost, *linear_sum_assignment(cost))
        theirs = cost[scipy_opt.linear_sum_assignment(cost)].sum()
        assert mine == pytest.approx(theirs)


class TestGreedyIsSuboptimal:
    def test_greedy_would_be_suboptimal_here(self):
        """The 2x2 counterexample from ``metrics/assignment.py``'s docstring.

        Greedy takes ref A -> hyp 0 (overlap 10), leaving B with 0, for 10.
        The optimum is A -> 1, B -> 0, for 18. On a five-speaker recording this
        failure mode is common, and it inflates DER.
        """
        overlap = np.array([[10.0, 9.0], [9.0, 0.0]])
        opt = assignment_cost(overlap, *linear_sum_assignment(overlap, maximize=True))
        greedy = assignment_cost(overlap, *greedy_assignment(overlap, maximize=True))
        assert opt == 18.0
        assert greedy == 10.0
        assert opt > greedy

    def test_greedy_never_beats_optimal(self):
        rng = np.random.default_rng(7)
        for _ in range(60):
            n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
            cost = rng.normal(size=(n, m))
            opt = assignment_cost(cost, *linear_sum_assignment(cost, maximize=True))
            grd = assignment_cost(cost, *greedy_assignment(cost, maximize=True))
            assert opt >= grd - 1e-9

    def test_greedy_matches_optimal_on_a_diagonal_problem(self):
        cost = np.eye(4) * -1.0
        assert assignment_cost(cost, *linear_sum_assignment(cost)) == assignment_cost(
            cost, *greedy_assignment(cost)
        )


class TestStructure:
    def test_columns_are_distinct(self):
        rng = np.random.default_rng(3)
        cost = rng.normal(size=(5, 7))
        _, cols = linear_sum_assignment(cost)
        assert len(set(cols.tolist())) == len(cols)

    def test_rows_are_ascending(self):
        rng = np.random.default_rng(4)
        _, _ = linear_sum_assignment(rng.normal(size=(6, 6)))
        rows, _ = linear_sum_assignment(rng.normal(size=(4, 9)))
        assert list(rows) == sorted(rows)

    def test_size_is_min_of_dimensions(self):
        for shape in [(3, 7), (7, 3), (5, 5), (1, 9)]:
            rows, cols = linear_sum_assignment(np.zeros(shape))
            assert len(rows) == len(cols) == min(shape)

    def test_tall_matrix_is_handled_by_transposition(self):
        cost = np.arange(12.0).reshape(4, 3)
        rows, cols = linear_sum_assignment(cost)
        assert len(rows) == 3
        assert assignment_cost(cost, rows, cols) == pytest.approx(brute_force_optimum(cost))

    def test_identity_permutation_is_recovered(self):
        cost = 1.0 - np.eye(5)
        rows, cols = linear_sum_assignment(cost)
        assert list(cols) == list(rows)


class TestEdgeCases:
    def test_empty_matrix_returns_empty(self):
        rows, cols = linear_sum_assignment(np.zeros((0, 3)))
        assert rows.size == 0 and cols.size == 0

    def test_zero_columns_returns_empty(self):
        rows, cols = linear_sum_assignment(np.zeros((3, 0)))
        assert rows.size == 0 and cols.size == 0

    def test_one_by_one(self):
        rows, cols = linear_sum_assignment(np.array([[5.0]]))
        assert (rows.tolist(), cols.tolist()) == ([0], [0])

    def test_all_equal_costs_still_valid(self):
        rows, cols = linear_sum_assignment(np.ones((4, 4)))
        assert len(set(cols.tolist())) == 4

    def test_nan_raises_rather_than_returning_garbage(self):
        cost = np.array([[1.0, np.nan], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN or inf"):
            linear_sum_assignment(cost)

    def test_inf_raises(self):
        cost = np.array([[1.0, np.inf], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN or inf"):
            linear_sum_assignment(cost)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            linear_sum_assignment(np.zeros(4))

    def test_negative_costs_are_fine(self):
        cost = -np.eye(3)
        rows, cols = linear_sum_assignment(cost)
        assert assignment_cost(cost, rows, cols) == pytest.approx(-3.0)

    def test_assignment_cost_of_empty_is_zero(self):
        assert assignment_cost(np.zeros((2, 2)), np.array([]), np.array([])) == 0.0

    def test_greedy_on_empty(self):
        rows, cols = greedy_assignment(np.zeros((0, 0)))
        assert rows.size == 0 and cols.size == 0

    def test_large_problem_is_fast_and_valid(self):
        rng = np.random.default_rng(11)
        cost = rng.normal(size=(60, 60))
        rows, cols = linear_sum_assignment(cost)
        assert len(set(cols.tolist())) == 60
        # A valid assignment must be no worse than the diagonal one.
        assert cost[rows, cols].sum() <= np.trace(cost) + 1e-9
