import os
import csv
import argparse
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "num_virtual_workers": int(row["num_virtual_workers"]),
                "local_steps": int(row["local_steps"]),
                "test_acc": float(row["test_acc"]) * 100.0,
                "averaging_rounds": int(row["averaging_rounds"]),
                "train_loss": float(row["train_loss"]),
                "test_loss": float(row["test_loss"]),
                "epoch_time_sec": float(row["epoch_time_sec"]),
                "samples_seen": int(row["samples_seen"]),
                "batches_seen": int(row["batches_seen"]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local_steps_list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    args = parser.parse_args()

    files = [
        (
            f"w={args.v_workers}, h={h}",
            f"results/w{args.v_workers}_h{h}_seed{args.seed}.csv",
        )
        for h in args.local_steps_list
    ]

    data = {}

    for label, path in files:
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            return
        data[label] = read_csv(path)

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(8, 5))
    for label, rows in data.items():
        xs = [r["epoch"] for r in rows]
        ys = [r["test_acc"] for r in rows]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Virtual Local SGD: Test Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/test_acc_vs_epoch.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, rows in data.items():
        xs = [r["averaging_rounds"] for r in rows]
        ys = [r["test_acc"] for r in rows]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Averaging Rounds (Analytical Communication Proxy)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Virtual Local SGD: Test Accuracy vs Averaging Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/test_acc_vs_averaging_rounds.png", dpi=200)
    plt.close()

    print("Saved figures:")
    print(" - figures/test_acc_vs_epoch.png")
    print(" - figures/test_acc_vs_averaging_rounds.png")

    print("\nFinal epoch summary:")
    for label, rows in data.items():
        last = rows[-1]
        print(
            f"{label}: "
            f"test_acc={last['test_acc']:.2f}% | "
            f"averaging_rounds={last['averaging_rounds']} | "
            f"train_loss={last['train_loss']:.4f} | "
            f"test_loss={last['test_loss']:.4f}"
        )


if __name__ == "__main__":
    main()