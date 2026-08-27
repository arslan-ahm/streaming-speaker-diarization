"""Deterministic seeding.

Every experiment in this repository must be re-runnable to bit-identical
per-item scores (``docs/REPRODUCIBILITY.md`` shows the evidence). That needs
three separate generators pinned, not one: Python's ``random`` (used by the
conversation generator's turn sampling fallbacks), NumPy's global RNG, and
torch's CPU RNG.

The generator functions below return *explicit* ``np.random.Generator`` objects
rather than touching global state. Global seeding is a safety net for library
code we do not control; the project's own data generation threads a seed
through by hand so that generating utterance 7 of recording 3 does not depend
on how many random numbers recording 2 consumed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Pin every global RNG we can reach.

    ``deterministic`` additionally disables cuDNN autotuning. On this project's
    CPU-only target that is a no-op, but it is left in so the Colab notebook
    (``notebooks/05``) inherits the same guarantee on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU on the target machine
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng_for(seed: int, *stream: int | str) -> np.random.Generator:
    """A child generator addressed by a *name*, not by call order.

    ``rng_for(0, "speaker", 3)`` always yields the same stream regardless of how
    much randomness speakers 0-2 consumed. This is what makes it possible to
    change the number of utterances in a recording without perturbing the
    speaker timbres, which in turn is what makes the ablations comparable.
    """
    entropy: list[int] = [int(seed)]
    for part in stream:
        if isinstance(part, str):
            # Stable across processes, unlike hash() which is salted per run.
            entropy.append(int.from_bytes(part.encode("utf-8")[:8].ljust(8, b"\0"), "little"))
        else:
            entropy.append(int(part))
    return np.random.default_rng(np.random.SeedSequence(entropy))


def limit_threads(n: int = 2) -> None:
    """Cap CPU parallelism.

    Other agents share this machine; the build standard fixes the budget at two
    threads. Environment variables are set *and* ``torch.set_num_threads`` is
    called because the env vars only bind if they are read before OpenMP
    initialises, which has usually already happened by the time a module import
    reaches here.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    torch.set_num_threads(n)
