"""Hand-rolled optimal assignment (Jonker-Volgenant with potentials).

Why this is not a detail. DER requires a one-to-one mapping between reference and
hypothesis speaker labels, chosen to *maximise* agreement. The obvious
implementation — walk reference speakers in order, greedily take the unused
hypothesis label with the largest overlap — is wrong, and wrong in a direction
that inflates DER. Concretely, with

    overlap =  [[10,  9],      # ref A
                [ 9,  0]]      # ref B

greedy takes ``A->0`` (10), leaving B with the only remaining column, 0, for a
total of 10. The optimum is ``A->1, B->0`` for a total of 18. On a five-speaker
recording that failure mode is common, not pathological, and it makes an online
system look worse than it is exactly where it is most likely to swap two labels.
``tests/test_assignment.py::test_greedy_would_be_suboptimal_here`` pins this
example.

The algorithm is the shortest-augmenting-path form of Hungarian/Kuhn-Munkres
(Jonker & Volgenant, *Computing* 38, 1987), which is ``O(n^2 m)`` with dual
potentials ``u, v`` maintained so every augmentation is along a shortest path.
This repository has **no SciPy dependency** — the copy in this environment ships
without ``scipy.__config__`` and fails to import, which is exactly the
install-fragility the build standard warns about. So the cross-check in
``tests/test_assignment.py`` is against a brute-force search over all
permutations on 400 random matrices up to 5x5. That is a stronger check than
agreeing with another library: it is the definition of the optimum.

Conventions
-----------
* ``cost`` may be rectangular. Rows are matched to distinct columns; if there
  are more rows than columns the matrix is transposed internally, so the result
  always matches ``min(n, m)`` pairs.
* Minimisation by default. ``maximize=True`` negates.
* Non-finite entries are rejected. A ``NaN`` cost silently poisons the potentials
  and yields a valid-looking but arbitrary permutation, so it raises instead.
"""

from __future__ import annotations

import numpy as np


def linear_sum_assignment(
    cost: np.ndarray, maximize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one assignment of rows to columns.

    Args:
        cost: ``(n, m)`` cost matrix. Must be finite.
        maximize: Maximise the total instead of minimising it.

    Returns:
        ``(row_ind, col_ind)``, both of length ``min(n, m)``, with ``row_ind``
        ascending. ``cost[row_ind, col_ind].sum()`` is optimal.

    Raises:
        ValueError: If ``cost`` is not 2-D or contains a non-finite entry.
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {c.shape}")
    if c.size and not np.isfinite(c).all():
        raise ValueError("cost contains NaN or inf; assignment would be arbitrary")
    if c.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    if maximize:
        c = -c

    transposed = c.shape[0] > c.shape[1]
    if transposed:
        c = c.T

    rows, cols = _solve_min(c)
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows, kind="stable")
    return rows[order], cols[order]


def _solve_min(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Core JV loop for ``n <= m`` minimisation. Returns unsorted ``(rows, cols)``.

    Uses 1-based indexing internally with a virtual row 0, which is what makes
    the augmenting-path bookkeeping fit in a dozen lines. The inner scan over
    unmatched columns is vectorised over NumPy rather than looped, which is the
    difference between milliseconds and seconds once a sweep scores a few
    thousand recordings.
    """
    n, m = a.shape
    inf = np.inf
    # u[i]: potential of row i (1..n). v[j]: potential of column j (1..m).
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    # p[j]: row currently matched to column j, 0 = free. way[j]: predecessor column.
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = int(p[j0])
            free = ~used[1:]
            if not free.any():  # pragma: no cover - impossible for n <= m
                raise RuntimeError("assignment failed: no free column")
            cur = a[i0 - 1][free] - u[i0] - v[1:][free]
            improve = cur < minv[1:][free]
            # Scatter the improvements back through the free-column mask.
            idx = np.flatnonzero(free) + 1
            upd = idx[improve]
            minv[upd] = cur[improve]
            way[upd] = j0
            j1 = int(idx[np.argmin(minv[1:][free])])
            delta = float(minv[j1])

            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while j0:  # augment along the path recorded in `way`
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    matched = np.flatnonzero(p[1:] > 0) + 1
    return (p[matched] - 1).astype(np.int64), (matched - 1).astype(np.int64)


def greedy_assignment(cost: np.ndarray, maximize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Row-order greedy assignment. **Not** used for scoring; kept for the test that
    demonstrates it is suboptimal, and for the ``docs/METHOD.md`` figure that
    quantifies how much DER a greedy mapping invents.
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if maximize:
        c = -c
    taken: set[int] = set()
    rows: list[int] = []
    cols: list[int] = []
    for i in range(min(c.shape)):
        order = np.argsort(c[i], kind="stable")
        for j in order:
            if int(j) not in taken:
                taken.add(int(j))
                rows.append(i)
                cols.append(int(j))
                break
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def assignment_cost(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    """Total of the selected cells; ``0.0`` for an empty assignment."""
    if len(rows) == 0:
        return 0.0
    return float(np.asarray(cost, dtype=np.float64)[np.asarray(rows), np.asarray(cols)].sum())
