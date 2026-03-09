Delayed Synchronization in Mini-batch Training
A Communication–Accuracy Trade-off Study on CIFAR-10
============================================================


1. Overview
------------------------------------------------------------

This project investigates the trade-off between communication
frequency and model convergence during training.

In many distributed machine learning systems, frequent
synchronization between workers introduces significant
communication overhead.

One possible strategy is to reduce synchronization frequency
so that workers perform more local computation before sharing
updates.

To study this effect, we simulate delayed synchronization
during training by updating the optimizer only every k
mini-batches.

This reduces the effective number of communication rounds
while keeping the training pipeline simple.


2. Experiment Design
------------------------------------------------------------

Dataset
CIFAR-10

Model
ResNet-18 (adapted for CIFAR-10)

Hardware
Single GPU (RTX 4060 Laptop GPU)

Training configuration

epochs = 10
batch_size = 512
optimizer = SGD
learning_rate = 0.1

Synchronization interval

k ∈ {1, 2, 4, 8}

Interpretation of k

k = 1
update parameters after every mini-batch (baseline)

k = 2
update every two mini-batches

k = 4
update every four mini-batches

k = 8
update every eight mini-batches

Increasing k reduces the number of synchronization rounds.


3. Implementation Idea
------------------------------------------------------------

Standard mini-batch training performs

forward → backward → optimizer step

for every batch.

In this experiment, gradients are accumulated across k batches,
and optimizer updates are executed only after every k batches.

Pseudo logic:

for each batch:
    compute loss
    loss.backward()

    if batch % k == 0:
        optimizer.step()
        optimizer.zero_grad()

This simulates reduced synchronization frequency.


4. Results
------------------------------------------------------------

Final test accuracy

k = 1
communication rounds = 980
test accuracy = 73.92%

k = 2
communication rounds = 490
test accuracy = 68.38%

k = 4
communication rounds = 250
test accuracy = 65.23%

k = 8
communication rounds = 130
test accuracy = 53.70%


5. Observations
------------------------------------------------------------

1. Reducing synchronization frequency dramatically reduces
   communication rounds.

2. A moderate delay (k=2) preserves most of the baseline
   accuracy while reducing communication cost by 50%.

3. Larger delays (k=4 and k=8) noticeably slow convergence
   and lead to lower final accuracy.

4. These results illustrate the fundamental
   computation–communication trade-off in distributed training.


6. Repository Structure
------------------------------------------------------------

project/

train.py
training script

plot.py
generates accuracy plots

results/
experiment logs

figures/
generated plots

README


7. Reproducing Experiments
------------------------------------------------------------

Run experiments

python train.py --epochs 10 --k 1 --seed 42
python train.py --epochs 10 --k 2 --seed 42
python train.py --epochs 10 --k 4 --seed 42
python train.py --epochs 10 --k 8 --seed 42

Generate plots

python plot.py


8. Possible Extensions
------------------------------------------------------------

future directions include

multiple random seeds
larger datasets
true multi-GPU distributed training
comparison with Local SGD algorithms
gradient compression techniques
communication-efficient optimizers