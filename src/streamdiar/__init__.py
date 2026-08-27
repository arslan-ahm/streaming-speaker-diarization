"""Online, bounded-latency, bounded-memory speaker diarization.

The reference approach this package argues against ingests a whole recording,
extracts speaker embeddings over sliding windows, and clusters them *once*, with
global spectral or agglomerative clustering. That design cannot emit a single
label until the recording ends, and its clustering step costs O(N^2) memory and
up to O(N^3) time in the number of windows.

This package emits a speaker label for every frame within a fixed latency budget
and never revisits more than a bounded window, so per-frame cost and resident
memory are O(1) in recording length. The same codebase also implements the
offline reference pipeline over identical features, so the accuracy that
streaming gives up can be measured rather than asserted.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
