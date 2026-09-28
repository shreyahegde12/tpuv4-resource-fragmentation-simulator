"""
simulator.py

A simple Monte Carlo simulator modeling the resource-fragmentation /
availability problem described in:

    Zu et al., "Resiliency at Scale: Managing Google's TPUv4 Machine
    Learning Supercomputer," NSDI 2024.

GRANULARITY NOTE
----------------
The paper allocates and fails resources at cube granularity (a cube
= 64 TPU chips = 16 machines), not individual-chip granularity
(Section 2.2: "The 16-machine granularity for fault-tolerance was
chosen to balance convenience... while retaining a relatively small
blast radius"). A TPUv4 pod has 64 cubes = 4096 chips total.

This simulator therefore represents the pool as N=64 cube-sized
units. Job sizes and results are reported in equivalent chip counts
(chips = cubes * 64) so they can be compared directly against
Figure 1's x-axis, but the underlying allocation/failure/occupancy
logic operates on whole cubes -- matching how the real system
actually assigns and fails resources.

CONCEPT
-------
A supercomputer pod is modeled as a 1D pool of N interchangeable
resource units ("cubes"). For each trial, every unit is independently
placed into one of three states:

    HEALTHY_FREE - usable
    FAILED       - hardware fault (models machine/ICI/OCS outages)
    OCCUPIED     - already claimed by another running job (fragmentation)

A large distributed training job requests K units and needs all K to
be simultaneously usable (gang scheduling: partial allocation cannot
run). Three allocation policies decide whether a request of size K can
be satisfied given a particular pool state:

    1. STATIC              - requires one contiguous run of >= K
                              HEALTHY_FREE units (models TPUv3: fixed,
                              non-reconfigurable ICI mesh).
    2. RECONFIGURABLE       - requires any K HEALTHY_FREE units,
                              anywhere in the pool (models TPUv4: OCS
                              can cross-connect any healthy cube into
                              the requested topology).
    3. BOUNDED_RECONFIG(F)  - requires the HEALTHY_FREE units to be
                              assemble-able from at most F contiguous
                              fragments summing to >= K (a middle
                              ground: reconfigurability limited by a
                              finite number of OCS reconnections /
                              switch capacity -- a scenario the paper
                              does not explicitly evaluate).

For a given (N, K, failure_prob, occupied_prob, policy) configuration,
we run many independent trials and report the fraction of trials in
which the job could be scheduled. This fraction is our estimate of
job-level availability, directly analogous to the y-axis of Figure 1
in the paper.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import List

CHIPS_PER_CUBE = 64   # matches the paper's cube = 4x4x4 = 64 TPU chips
CUBES_PER_POD = 64    # matches the paper's 64-cube, 4096-chip pod


class State(Enum):
    HEALTHY_FREE = 0
    FAILED = 1
    OCCUPIED = 2


def generate_pool(n: int, failure_prob: float, occupied_prob: float,
                   rng: random.Random) -> List[State]:
    """Randomly generate one pool state of n units.

    Each unit is FAILED with probability `failure_prob`; otherwise it
    is OCCUPIED with probability `occupied_prob`; otherwise it is
    HEALTHY_FREE. Failure and occupation are treated as independent
    per-unit events, which is a simplification the report should
    flag (real deployments show some correlated failures).
    """
    pool = []
    for _ in range(n):
        r = rng.random()
        if r < failure_prob:
            pool.append(State.FAILED)
        elif r < failure_prob + occupied_prob:
            pool.append(State.OCCUPIED)
        else:
            pool.append(State.HEALTHY_FREE)
    return pool


def free_runs(pool: List[State]) -> List[int]:
    """Return the lengths of all maximal contiguous runs of
    HEALTHY_FREE units in the pool (pool is treated as linear, not
    wraparound -- a simplification vs. the paper's 3D torus)."""
    runs = []
    current = 0
    for unit in pool:
        if unit == State.HEALTHY_FREE:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


def can_allocate_static(pool: List[State], k: int) -> bool:
    """Static policy: need one contiguous run of >= k healthy-free units."""
    return any(run >= k for run in free_runs(pool))


def can_allocate_reconfigurable(pool: List[State], k: int) -> bool:
    """Full pooling policy: need any k healthy-free units anywhere."""
    total_free = sum(1 for u in pool if u == State.HEALTHY_FREE)
    return total_free >= k


def can_allocate_bounded(pool: List[State], k: int, max_fragments: int) -> bool:
    """Bounded-reconfigurable policy: assemble k units from at most
    `max_fragments` contiguous fragments, using the largest fragments
    first (greedy -- optimal for this simple "sum of chosen fragments"
    objective since fragments are otherwise interchangeable)."""
    runs = sorted(free_runs(pool), reverse=True)
    chosen = runs[:max_fragments]
    return sum(chosen) >= k


def run_trials(n: int, k: int, failure_prob: float, occupied_prob: float,
                policy: str, trials: int, seed: int = 0,
                max_fragments: int = 1) -> float:
    """Run `trials` independent Monte Carlo trials and return the
    fraction of trials in which the job of size k could be scheduled.

    policy: one of "static", "reconfigurable", "bounded"
    """
    rng = random.Random(seed)
    successes = 0
    for _ in range(trials):
        pool = generate_pool(n, failure_prob, occupied_prob, rng)
        if policy == "static":
            ok = can_allocate_static(pool, k)
        elif policy == "reconfigurable":
            ok = can_allocate_reconfigurable(pool, k)
        elif policy == "bounded":
            ok = can_allocate_bounded(pool, k, max_fragments)
        else:
            raise ValueError(f"unknown policy: {policy}")
        if ok:
            successes += 1
    return successes / trials


if __name__ == "__main__":
    # Quick sanity check / demo, at cube granularity.
    # k=16 cubes == 1024 chips, matching the paper's TPUv3 collapse point.
    avail_static = run_trials(n=CUBES_PER_POD, k=16, failure_prob=0.013,
                               occupied_prob=0.10, policy="static",
                               trials=2000)
    avail_reconf = run_trials(n=CUBES_PER_POD, k=16, failure_prob=0.013,
                               occupied_prob=0.10, policy="reconfigurable",
                               trials=2000)
    print(f"Static availability at 1024 chips:        {avail_static:.4f}")
    print(f"Reconfigurable availability at 1024 chips: {avail_reconf:.4f}")
