# TPUv4 Resource Fragmentation Simulator

A small Monte Carlo simulator of the resource-fragmentation /
availability problem in Zu et al., "Resiliency at Scale: Managing
Google's TPUv4 Machine Learning Supercomputer" (NSDI 2024).

A pod is modeled as 64 interchangeable cubes. Each cube is
**healthy-free**, **failed**, or **occupied** by another job. A large
distributed job needs enough usable cubes at the same time (gang
scheduling). The demo compares:

- **Static** — needs one contiguous run of free cubes (TPUv3-like)
- **Reconfigurable** — can use any free cubes in the pool (TPUv4-like)

It reports the fraction of trials in which a 1024-chip job can be
scheduled. Static availability is lower because contiguous allocation
is more sensitive to holes from failures and other jobs.

## Files

- `simulator.py` — pool generation and allocation policies

## Requirements

- Python 3.8+ (standard library only)

No cloud provider, GPU/TPU access, or ML framework is required.

## Setup (Ubuntu)

```bash
sudo apt update
sudo apt install -y python3
```

## Running

```bash
python3 simulator.py
```

This runs 2000 trials at 1024 chips (16 cubes) with cube failure
probability 0.013 and occupancy 0.10. It prints static vs.
reconfigurable availability. Expected output (fixed seed):

```
Static availability at 1024 chips:        0.7360
Reconfigurable availability at 1024 chips: 1.0000
```

Static should be clearly lower than reconfigurable.

## Model notes

- Allocation and failures are modeled at **cube granularity**
  (64 cubes = 4096 chips), matching Section 2.2 of the paper.
- The pool is a 1D line of cubes (not the paper's 3D torus).
- Failures and occupancy are independent per cube.
