import os
import csv
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "k": int(row["k"]),
                "test_acc": float(row["test_acc"]) * 100.0,  # convert to %
                "comm_rounds": int(row["comm_rounds"]),
                "train_loss": float(row["train_loss"]),
                "test_loss": float(row["test_loss"]),
                "epoch_time_sec": float(row["epoch_time_sec"]),
            })
    return rows


def main():
    files = [
        ("k=1", "results/k1_seed42.csv"),
        ("k=2", "results/k2_seed42.csv"),
        ("k=4", "results/k4_seed42.csv"),
        ("k=8", "results/k8_seed42.csv"),
    ]

    data = {}

    for label, path in files:
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            return
        data[label] = read_csv(path)

    os.makedirs("figures", exist_ok=True)

    # 1) test accuracy vs epoch
    plt.figure(figsize=(8, 5))
    for label, rows in data.items():
        xs = [r["epoch"] for r in rows]
        ys = [r["test_acc"] for r in rows]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/test_acc_vs_epoch.png", dpi=200)
    plt.close()

    # 2) test accuracy vs communication rounds
    plt.figure(figsize=(8, 5))
    for label, rows in data.items():
        xs = [r["comm_rounds"] for r in rows]
        ys = [r["test_acc"] for r in rows]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs Communication Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/test_acc_vs_comm_rounds.png", dpi=200)
    plt.close()

    print("Saved figures:")
    print(" - figures/test_acc_vs_epoch.png")
    print(" - figures/test_acc_vs_comm_rounds.png")

    # print final summary
    print("\nFinal epoch summary:")
    for label, rows in data.items():
        last = rows[-1]
        print(
            f"{label}: "
            f"test_acc={last['test_acc']:.2f}% | "
            f"comm_rounds={last['comm_rounds']} | "
            f"train_loss={last['train_loss']:.4f} | "
            f"test_loss={last['test_loss']:.4f}"
        )


if __name__ == "__main__":
    main()