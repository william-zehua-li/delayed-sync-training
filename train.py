import os
import csv
import time
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    model.train()
    return avg_loss, acc


def clone_model(global_model, device):
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    local_model.train()
    return local_model


@torch.no_grad()
def average_local_models(global_model, local_model):
    global_state = global_model.state_dict()
    local_state = [m.state_dict() for m in local_model]

    for key in global_state:
        tensors = [state[key] for state in local_state]

        if tensors[0].is_floating_point():
            stacked = torch.stack([t.detach() for t in tensors], dim=0)
            global_state[key] = stacked.mean(dim=0)
        else:
            global_state[key] = tensors[0]

    global_model.load_state_dict(global_state)


def local_sgd_update(model, x, y, loss_fn, lr):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--v_worker", type=int, default=4)
    parser.add_argument("--local_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    global_model = models.resnet18(num_classes=10)
    global_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    global_model.maxpool = nn.Identity()
    global_model = global_model.to(device)
    train_loss_fn = nn.CrossEntropyLoss()

    os.makedirs("results", exist_ok=True)
    csv_path = (
        f"results/w{args.v_worker}_h{args.local_steps}_seed{args.seed}.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "v_worker",
            "local_steps",
            "seed",
            "train_loss",
            "test_loss",
            "test_acc",
            "epoch_time_sec",
            "samples_seen",
            "batches_seen",
            "averaging_rounds",
        ])

    total_batches = 0
    total_samples = 0
    rounds = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        global_model.train()

        train_iter = iter(train_loader)
        b_in_epoch = len(train_loader)

        train_loss_sum = 0.0
        batch_count = 0

        while batch_count < b_in_epoch:
            local_models = [
                clone_model(global_model, device)
                for _ in range(args.v_worker)
            ]

            any_update = False

            for worker_id in range(args.v_worker):
                for _ in range(args.local_steps):
                    if batch_count >= b_in_epoch:
                        break

                    try:
                        x, y = next(train_iter)
                    except StopIteration:
                        break

                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    loss_value = local_sgd_update(
                        local_models[worker_id],
                        x,
                        y,
                        train_loss_fn,
                        args.lr,
                    )

                    train_loss_sum += loss_value
                    batch_count += 1
                    total_batches += 1
                    total_samples += x.size(0)
                    any_update = True

            if any_update:
                average_local_models(global_model, local_models)
                rounds += 1

        avg_train_loss = train_loss_sum / max(batch_count, 1)
        test_loss, test_acc = evaluate(global_model, test_loader, device)
        dt = time.time() - t0

        print(
            f"epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}  "
            f"test_loss={test_loss:.4f}  "
            f"test_acc={test_acc * 100:.2f}%  "
            f"time={dt:.2f}s  "
            f"workers={args.v_worker}  "
            f"local_steps={args.local_steps}  "
            f"batches_seen={total_batches}  "
            f"averaging_rounds={rounds}"
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                args.v_worker,
                args.local_steps,
                args.seed,
                avg_train_loss,
                test_loss,
                test_acc,
                dt,
                total_samples,
                total_batches,
                rounds,
            ])

    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()