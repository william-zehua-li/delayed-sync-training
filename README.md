# Virtual Local SGD on CIFAR-10
## A Small Prototype for Studying Synchronization Frequency in Local-Update Training

---

# 1. Overview

This project implements a small experimental prototype for studying how synchronization frequency influences training behavior in a simplified local-update training setting.

In communication-efficient distributed training, workers may perform several **local SGD updates** before synchronizing model parameters. Reducing synchronization frequency can lower communication cost, but it may also introduce **model drift**, which can negatively affect training stability and final model accuracy.

To examine this behavior in a controlled setting, this repository simulates multiple **virtual workers** on a single GPU. Each worker performs several local updates before model parameters are averaged. Although this implementation is a **serial simulation rather than a true distributed system**, it provides a simple environment for observing how reduced synchronization affects training dynamics and final test accuracy.

---

# 2. Research Question

This project focuses on the following questions:

1. How does increasing the number of local updates between parameter averaging operations affect training stability and final test accuracy?

2. Can synchronization frequency be reduced while maintaining comparable model quality?

---

# 3. Method

## Core Idea

The training procedure follows a simplified **virtual local SGD** approach:

1. Maintain a global model.
2. Clone the model into several virtual worker models.
3. Each worker performs several local SGD updates.
4. Worker parameters are averaged.
5. The averaged parameters replace the global model.
6. The process repeats.

Each parameter averaging step is counted as an **averaging round**, which is used as a simple proxy for synchronization frequency.

---

## Difference from Gradient Accumulation

This method should not be confused with gradient accumulation.

| Method | Core idea | Multiple drifting workers? | Parameter averaging? | Distributed communication? |
|---|---|---|---|---|
| Standard SGD | Update parameters after every batch | No | No | No |
| Gradient Accumulation | Accumulate gradients before optimizer step | No | No | No |
| Virtual Local SGD (this project) | Workers perform local updates before averaging | Yes | Yes | No |

Gradient accumulation modifies the update frequency of **one model**, while local SGD studies the behavior of **multiple models drifting apart before synchronization**.

---

# 4. Experiment Design

## Dataset
CIFAR-10

## Model
ResNet-18 adapted for CIFAR-10:

- 3×3 first convolution
- max-pool layer removed

## Hardware
Single GPU (RTX 4060 Laptop GPU in this experiment)

## Training Setup

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 512 |
| Learning rate | 0.1 |
| Optimizer | simplified SGD update |

## Virtual Worker Configuration

| Parameter | Values |
|---|---|
| Virtual workers | 4 |
| Local steps | {1, 2, 4, 8} |

Interpretation:

- **h = 1** → frequent synchronization
- **h = 2** → moderate local computation
- **h = 4** → reduced synchronization
- **h = 8** → aggressive local updates

As the number of local steps increases, synchronization becomes less frequent.

These configurations simulate different synchronization intervals in a simplified local-SGD style training process.

---

# 5. Repository Structure

```text
project/
├── train.py
├── plot.py
├── results/
├── figures/
└── README.md
```

- **train.py** – virtual local SGD training script
- **plot.py** – generates experiment plots
- **results/** – CSV logs for each experiment
- **figures/** – generated figures

---

# 6. Running Experiments

Run the training experiments:

```bash
python train.py --epochs 10 --v_worker 4 --local_steps 1 --seed 42
python train.py --epochs 10 --v_worker 4 --local_steps 2 --seed 42
python train.py --epochs 10 --v_worker 4 --local_steps 4 --seed 42
python train.py --epochs 10 --v_worker 4 --local_steps 8 --seed 42
```

Generate plots:

```bash
python plot.py --v_worker 4 --seed 42 --local_steps_list 1 2 4 8
```

---

# 7. Output Files

Training logs are saved as:

```text
results/w4_h1_seed42.csv
results/w4_h2_seed42.csv
results/w4_h4_seed42.csv
results/w4_h8_seed42.csv
```

Naming convention:

- **w** – number of virtual workers
- **h** – number of local steps

Generated figures:

```text
figures/test_acc_vs_epoch.png
figures/test_acc_vs_averaging_rounds.png
```

---

# 8. Experimental Results

Experiments were conducted with four virtual workers and varying numbers of local updates between parameter averaging steps.

Final test accuracy after 10 epochs:

| Configuration | Final Accuracy |
|---|---|
| w=1, h=1 (reference) | 62.45% |
| w=4, h=1 | 59.7% |
| w=4, h=2 | 55.9% |
| w=4, h=4 | 58.8% |
| w=4, h=8 | 56.0% |

The results illustrate how synchronization frequency influences final model accuracy under this simplified training setup.

Frequent synchronization (h=1) achieves the highest accuracy among the multi-worker configurations. Increasing the number of local updates reduces the number of averaging rounds, thereby lowering synchronization frequency.

However, larger local update intervals introduce stronger model drift between workers and can slightly degrade final test accuracy. For example, h=8 significantly reduces averaging rounds but results in lower final test accuracy.

The configuration **h=4 maintains accuracy close to h=1 while requiring substantially fewer averaging rounds**, suggesting that moderate local computation may reduce synchronization frequency while preserving most of the final model accuracy.

The **w=1, h=1** configuration is included as a reference corresponding to standard single-worker training without parameter averaging.

Training loss curves plotted against averaging rounds provide an approximate view of optimization progress under different synchronization intervals. Since each averaging round corresponds to a parameter synchronization step, the horizontal axis can be interpreted as a proxy for communication budget.

---

# 9. Limitations

This repository is a **small experimental prototype**, not a full distributed training system.

Key limitations include:

- training is performed on a **single GPU**
- no real distributed framework (DDP, NCCL, etc.) is used
- communication is not measured directly
- averaging rounds are only a proxy for synchronization cost

The results should therefore be interpreted as empirical observations about training behavior under different synchronization intervals rather than theoretical claims about convergence.

---

# 10. Why This Project Matters

Despite its simplicity, this prototype demonstrates several important ideas in communication-efficient training:

- synchronization frequency affects training dynamics
- fewer synchronization steps can reduce communication cost
- excessive local updates may introduce model drift
- moderate local computation may preserve accuracy while reducing synchronization

The repository therefore provides a small testbed for exploring communication-efficient optimization strategies.

---

# 11. Future Work

Possible extensions include:

- running multiple seeds and reporting mean ± standard deviation
- comparing virtual local SGD with gradient accumulation
- tuning learning rate for larger local steps
- adding momentum or alternative optimizers
- implementing real multi-GPU distributed training
- comparing with other communication-efficient training methods

---

# 12. Summary

This repository presents a small experimental implementation of **virtual local SGD** for studying how synchronization intervals influence training behavior and final model accuracy on CIFAR-10.

The goal is not to build a production-level distributed training system, but to provide a simple experimental setup for examining:

- local computation
- periodic parameter averaging
- model drift
- communication-efficient optimization

While the current experiments focus on final test accuracy for simplicity, future work may include analyzing training loss trajectories and other optimization-side indicators.