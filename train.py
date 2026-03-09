import os
import csv
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--k", type=int, default=1)  # update every k batches
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    # speed settings
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

    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    opt = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    train_loss_fn = nn.CrossEntropyLoss()
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # results file
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/k{args.k}_seed{args.seed}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "k",
            "seed",
            "train_loss",
            "test_loss",
            "test_acc",
            "epoch_time_sec",
            "total_batches",
            "optimizer_steps_this_epoch",
            "comm_rounds",
        ])

    total_batches = 0
    comm_rounds = 0

    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        total_train_loss = 0.0
        optimizer_steps_this_epoch = 0

        opt.zero_grad(set_to_none=True)

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = train_loss_fn(logits, y)
                loss = loss / args.k  # gradient accumulation normalization

            scaler.scale(loss).backward()

            total_train_loss += loss.item() * args.k
            total_batches += 1

            if total_batches % args.k == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                comm_rounds += 1
                optimizer_steps_this_epoch += 1

        # flush leftover gradients at epoch end
        if total_batches % args.k != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            comm_rounds += 1
            optimizer_steps_this_epoch += 1

        avg_train_loss = total_train_loss / len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        print(
            f"epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}  "
            f"test_loss={test_loss:.4f}  "
            f"test_acc={test_acc*100:.2f}%  "
            f"time={dt:.2f}s  "
            f"total_batches={total_batches}  "
            f"k={args.k}  "
            f"optimizer_steps_this_epoch={optimizer_steps_this_epoch}  "
            f"comm_rounds={comm_rounds}"
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                args.k,
                args.seed,
                avg_train_loss,
                test_loss,
                test_acc,
                dt,
                total_batches,
                optimizer_steps_this_epoch,
                comm_rounds,
            ])

    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()