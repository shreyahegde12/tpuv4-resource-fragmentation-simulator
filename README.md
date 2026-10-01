# TPUv4 Resource Fragmentation Simulator

A small Monte Carlo simulator of the resource-fragmentation /
availability problem in Zu et al., "Resiliency at Scale: Managing
Google's TPUv4 Machine Learning Supercomputer" (NSDI 2024).

A pod is modeled as 64 interchangeable cubes. Each cube is
**healthy-free**, **failed**, or **occupied** by another job. A large
distributed job needs enough usable cubes at the same time (gang
scheduling). Three allocation policies are compared:

- **Static**: needs one contiguous run of free cubes (TPUv3-like)
- **Reconfigurable**: can use any free cubes in the pool (TPUv4-like)
- **Bounded-reconfigurable**: can stitch a job together from at most F
  contiguous fragments (a middle ground the paper does not evaluate)

The output is job availability: the fraction of trials in which a job
of a given size can be scheduled.

## Files

- `simulator.py`: pool generation and the three allocation policies
- `run_experiments.py`: runs the four experiments used in the report and
  saves plots (`experiment1_job_size.png`, `experiment2_occupancy.png`,
  `experiment3_fragments.png`, `experiment4_pool_size.png`)

## Requirements

- Python 3.8+
- `simulator.py` uses the standard library only
- `run_experiments.py` also needs `matplotlib`

No cloud provider, GPU/TPU access, or ML framework is required.

## Setup (Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-matplotlib
```

## Running

Quick sanity check of the core model:

```bash
python3 simulator.py
```

This runs 2000 trials at 1024 chips (16 cubes) with cube failure
probability 0.013 and occupancy 0.10. Expected output (fixed seed):

```
Static availability at 1024 chips:        0.7360
Reconfigurable availability at 1024 chips: 1.0000
```

Run all four experiments (generates the plots used in the report):

```bash
python3 run_experiments.py
```

This prints each experiment's hypothesis, the parameter swept, raw
results, and whether the hypothesis was supported, then saves four PNG
plots to the current directory. It runs in well under a minute on a
normal laptop.

## Model notes

- Allocation and failures are modeled at **cube granularity**
  (64 cubes = 4096 chips), matching Section 2.2 of the paper.
- `failure_prob = 0.013` per cube is derived from the paper's reported
  ~0.08%/day machine failure rate (Section 5.2), compounded over the
  16 machines in a cube.
- `occupied_prob = 0.14` (experiment baseline) is *not* taken from the
  paper. It is calibrated so the reconfigurable curve reproduces the
  paper's ~94% availability at ~50 cubes / 3200 chips (Figure 1).
- The pool is a 1D line of cubes (not the paper's 3D torus).
- Each cube is assigned one of three mutually exclusive states using the configured failure and occupancy probabilities.
