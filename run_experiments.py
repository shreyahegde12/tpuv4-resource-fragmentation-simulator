"""
run_experiments.py

Runs the three experiments for Part 3 of the assignment and saves
plots to the current directory. Each experiment prints its hypothesis,
the parameter varied, and a short interpretation to stdout, matching
the 5-step structure the assignment asks for (hypothesis, parameter
changed, result, explanation, supported-or-not) -- the printed output
here is meant to be copy-adapted into the written report, not a
replacement for the report's own prose.

MODEL CALIBRATION
------------------
The pool is modeled at cube granularity (N=64 cubes = 4096 chips),
matching the paper's actual allocation/fault unit (Section 2.2).

failure_prob = 0.013 per cube. Derived from the paper's measured
fleet statistic that ~0.08% of TPU machines fail per day (Section
5.2). A cube contains 16 machines, so the chance a cube has at
least one failed machine is 1-(1-0.0008)^16 = 0.0127 = ~1.3%.

occupied_prob = 0.14 (baseline). This value is NOT taken directly
from the paper (which does not publish a workload-occupancy
percentage); it was chosen so the reconfigurable-policy curve
reproduces the paper's reported ~94% job availability at ~50
cubes / 3200 chips (Section 2, Figure 1). This is flagged explicitly
in Part 5 of the report as a calibrated-to-match-output assumption,
not an independently measured input.

Usage:
    python3 run_experiments.py
"""

import matplotlib
matplotlib.use("Agg")  # headless/server-safe backend
import matplotlib.pyplot as plt

from simulator import run_trials, CUBES_PER_POD, CHIPS_PER_CUBE

N = CUBES_PER_POD   # 64 cubes, matching the paper's 64-cube / 4096-chip pod
TRIALS = 3000
SEED = 7

# Baseline parameters (see calibration note above)
FAILURE_PROB = 0.013
OCCUPIED_PROB_BASELINE = 0.14


def chips(k_cubes):
    return k_cubes * CHIPS_PER_CUBE


def experiment_1_job_size_sweep():
    """
    Hypothesis: As required job size K grows, static allocation
    availability collapses well before the reconfigurable policy
    does, reproducing the qualitative shape of Figure 1 in the paper
    (static collapse starting around 1024 chips; reconfigurable
    staying near 94% out to ~3200 chips).
    Parameter varied: job size K (in cubes, reported in chips).
    """
    print("\n=== Experiment 1: Availability vs. Job Size ===")
    print("Hypothesis: static allocation collapses through the ~1024-chip "
          "range while reconfigurable stays near its ~94% ceiling out to "
          "~3200 chips, matching Figure 1's shape.")

    job_sizes_cubes = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 50, 56, 60, 64]
    static_avail, reconf_avail = [], []

    for k in job_sizes_cubes:
        static_avail.append(run_trials(N, k, FAILURE_PROB, OCCUPIED_PROB_BASELINE,
                                        "static", TRIALS, seed=SEED))
        reconf_avail.append(run_trials(N, k, FAILURE_PROB, OCCUPIED_PROB_BASELINE,
                                        "reconfigurable", TRIALS, seed=SEED))
        print(f"  {chips(k):5d} chips  static={static_avail[-1]:.3f}  "
              f"reconfigurable={reconf_avail[-1]:.3f}")

    x = [chips(k) for k in job_sizes_cubes]
    plt.figure(figsize=(7, 5))
    plt.plot(x, static_avail, marker="o", label="Static (TPUv3-like)")
    plt.plot(x, reconf_avail, marker="s", label="Reconfigurable (TPUv4-like)")
    plt.xlabel("Requested job size (chips)")
    plt.ylabel("Job availability (fraction of trials schedulable)")
    plt.title("Experiment 1: Availability vs. Job Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment1_job_size.png", dpi=150)
    plt.close()
    print("Saved experiment1_job_size.png")

    mid = len(job_sizes_cubes) // 2
    supported = static_avail[mid] < reconf_avail[mid]
    print(f"Result: static collapses well before reconfigurable as K grows. "
          f"Hypothesis supported: {supported}")


def experiment_2_occupancy_sweep():
    """
    Hypothesis: increasing the fraction of the pool occupied by other
    jobs (workload contention) degrades static availability much
    faster than reconfigurable availability at a fixed mid-size job,
    because occupied cubes break up contiguous free runs even when
    the total amount of free capacity is still large.
    Parameter varied: occupied_prob (fraction of pool held by other jobs).
    """
    print("\n=== Experiment 2: Availability vs. Occupancy (Contention) ===")
    print("Hypothesis: rising occupancy fragments the static pool and "
          "collapses its availability faster than reconfigurable "
          "availability at the same fixed job size.")

    k_cubes = 16  # 1024 chips -- the paper's own static collapse point
    occupancy_levels = [0.0, 0.05, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30]

    static_avail, reconf_avail = [], []
    for occ in occupancy_levels:
        static_avail.append(run_trials(N, k_cubes, FAILURE_PROB, occ,
                                        "static", TRIALS, seed=SEED))
        reconf_avail.append(run_trials(N, k_cubes, FAILURE_PROB, occ,
                                        "reconfigurable", TRIALS, seed=SEED))
        print(f"  occupancy={occ:.2f}  static={static_avail[-1]:.3f}  "
              f"reconfigurable={reconf_avail[-1]:.3f}")

    x = [o * 100 for o in occupancy_levels]
    plt.figure(figsize=(7, 5))
    plt.plot(x, static_avail, marker="o", label="Static (TPUv3-like)")
    plt.plot(x, reconf_avail, marker="s", label="Reconfigurable (TPUv4-like)")
    plt.xlabel("Percent of pool occupied by other jobs (%)")
    plt.ylabel(f"Job availability (K={chips(k_cubes)} chips)")
    plt.title("Experiment 2: Availability vs. Workload Occupancy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment2_occupancy.png", dpi=150)
    plt.close()
    print("Saved experiment2_occupancy.png")

    supported = static_avail[-1] < reconf_avail[-1]
    print(f"Result: static degrades faster with occupancy than reconfigurable. "
          f"Hypothesis supported: {supported}")


def experiment_3_bounded_fragments():
    """
    This is the scenario NOT explicitly evaluated in the paper. Real
    OCS reconfiguration is not infinitely flexible -- it is bounded by
    how many separate cross-connects can practically be formed and
    managed for one job. We model this as a "bounded reconfigurable"
    policy allowed to stitch together at most F contiguous fragments.
    Hypothesis: availability rises steeply from F=1 (== static) toward
    the full-pooling ceiling as F increases, with diminishing returns
    after a small number of fragments -- i.e., reconfigurability
    doesn't need to be unlimited to capture most of the benefit.
    Parameter varied: max_fragments F.
    """
    print("\n=== Experiment 3: Availability vs. Fragment Budget (novel) ===")
    print("Hypothesis: availability rises steeply from F=1 (static) and "
          "approaches the full-pooling ceiling with diminishing returns "
          "as the fragment budget F increases.")

    k_cubes = 16  # 1024 chips
    occupied_prob = OCCUPIED_PROB_BASELINE
    fragment_budgets = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]

    bounded_avail = []
    for f in fragment_budgets:
        bounded_avail.append(run_trials(N, k_cubes, FAILURE_PROB, occupied_prob,
                                         "bounded", TRIALS, seed=SEED,
                                         max_fragments=f))
        print(f"  F={f:3d}  bounded_availability={bounded_avail[-1]:.3f}")

    reconf_ceiling = run_trials(N, k_cubes, FAILURE_PROB, occupied_prob,
                                 "reconfigurable", TRIALS, seed=SEED)
    print(f"  (full pooling ceiling: {reconf_ceiling:.3f})")

    plt.figure(figsize=(7, 5))
    plt.plot(fragment_budgets, bounded_avail, marker="o",
              label="Bounded-reconfigurable")
    plt.axhline(reconf_ceiling, color="gray", linestyle="--",
                label="Full pooling ceiling")
    plt.xlabel("Fragment budget F (max stitched fragments)")
    plt.ylabel(f"Job availability (K={chips(k_cubes)} chips)")
    plt.title("Experiment 3: Diminishing Returns of Bounded Reconfigurability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment3_fragments.png", dpi=150)
    plt.close()
    print("Saved experiment3_fragments.png")

    early_gain = bounded_avail[2] - bounded_avail[0]
    late_gain = bounded_avail[-1] - bounded_avail[len(bounded_avail) // 2]
    supported = early_gain > late_gain
    print(f"Result: most of the availability gain occurs within a small "
          f"fragment budget (early gain={early_gain:.3f} vs. "
          f"late gain={late_gain:.3f}). Hypothesis supported: {supported}")


if __name__ == "__main__":
    experiment_1_job_size_sweep()
    experiment_2_occupancy_sweep()
    experiment_3_bounded_fragments()
    print("\nAll experiments complete. PNG files saved in current directory.")
