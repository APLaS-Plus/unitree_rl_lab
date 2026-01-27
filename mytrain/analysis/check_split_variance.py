import os
import glob
import random
import torch
import numpy as np
import argparse


def compute_stats(files, dataset_name="Dataset"):
    print(f"\n[INFO] Computing stats for {dataset_name} ({len(files)} files)...")
    if not files:
        print("  No files found.")
        return None, None, None, None

    sum_z = 0
    sum_sq_z = 0
    sum_p = 0
    sum_sq_p = 0
    total_samples = 0

    device = "cpu"

    for i, fpath in enumerate(files):
        try:
            data = torch.load(fpath, map_location=device)
            z_t = data["teacher_latent"]
            phys = data["phys_params"]

            sum_z += torch.sum(z_t, dim=0)
            sum_sq_z += torch.sum(z_t**2, dim=0)

            sum_p += torch.sum(phys, dim=0)
            sum_sq_p += torch.sum(phys**2, dim=0)

            total_samples += z_t.shape[0]

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(files)} files...")
        except Exception as e:
            print(f"  [ERROR] Failed to load {fpath}: {e}")

    if total_samples == 0:
        return None, None, None, None

    mean_z = sum_z / total_samples
    var_z = (sum_sq_z / total_samples) - (mean_z**2)
    std_z = torch.sqrt(torch.clamp(var_z, min=1e-6))

    mean_p = sum_p / total_samples
    var_p = (sum_sq_p / total_samples) - (mean_p**2)
    std_p = torch.sqrt(torch.clamp(var_p, min=1e-6))

    return mean_z, std_z, mean_p, std_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/teacher_latent_10M.pt",
        help="Path pattern",
    )
    args = parser.parse_args()

    # Replicate Split Logic from train_adaptation.py
    # ---------------------------------------------------------
    pattern = args.data_path
    if "*" not in pattern and "_chunk" not in pattern:
        search_pattern = pattern.replace(".pt", "_chunk*.pt")
    else:
        search_pattern = pattern

    print(f"[INFO] Searching for files: {search_pattern}")
    all_files = sorted(glob.glob(search_pattern))

    if not all_files:
        if os.path.exists(pattern):
            all_files = [pattern]
        else:
            print(f"[ERROR] No files found.")
            return

    # Random shuffle using SAME seed as training
    random.seed(42)
    random.shuffle(all_files)

    val_ratio = 0.05
    n_val = max(1, int(len(all_files) * val_ratio))
    n_train = len(all_files) - n_val

    train_files = all_files[:n_train]
    val_files = all_files[n_train:]
    # ---------------------------------------------------------

    print(f"[INFO] Total Files: {len(all_files)}")
    print(f"[INFO] Train Files: {len(train_files)}")
    print(f"[INFO] Val Files:   {len(val_files)}")

    if len(val_files) == 0:
        print("[WARN] Validation set is empty! (Maybe only 1 file exists?)")
        # Proceed with just Train stats
        t_mean_z, t_std_z, t_mean_p, t_std_p = compute_stats(train_files, "TRAIN")
        return

    # Compute Stats
    t_mean_z, t_std_z, t_mean_p, t_std_p = compute_stats(train_files, "TRAIN")
    v_mean_z, v_std_z, v_mean_p, v_std_p = compute_stats(val_files, "VAL")

    # Compare
    print("\n" + "=" * 50)
    print("STATISTICS COMPARISON")
    print("=" * 50)

    # Latent
    print("\n--- Latent Vector (Teacher Z) ---")
    print(
        f"{'Dim':<5} {'Train Mean':<12} {'Val Mean':<12} {'Diff':<12} | {'Train Std':<12} {'Val Std':<12} {'Diff':<12}"
    )

    # Detailed check for first 5 dims
    for d in range(min(5, len(t_mean_z))):
        tm, vm = t_mean_z[d].item(), v_mean_z[d].item()
        ts, vs = t_std_z[d].item(), v_std_z[d].item()
        print(
            f"{d:<5} {tm:<12.4f} {vm:<12.4f} {abs(tm-vm):<12.4f} | {ts:<12.4f} {vs:<12.4f} {abs(ts-vs):<12.4f}"
        )

    print(f"\nAvg Diff across all {len(t_mean_z)} dims:")
    print(f"Mean Abs Diff: {torch.mean(torch.abs(t_mean_z - v_mean_z)).item():.6f}")
    print(f"Std Abs Diff:  {torch.mean(torch.abs(t_std_z - v_std_z)).item():.6f}")

    # Phys
    print("\n--- Physical Params ---")
    print(
        f"{'Dim':<5} {'Train Mean':<12} {'Val Mean':<12} {'Diff':<12} | {'Train Std':<12} {'Val Std':<12} {'Diff':<12}"
    )

    for d in range(min(5, len(t_mean_p))):
        tm, vm = t_mean_p[d].item(), v_mean_p[d].item()
        ts, vs = t_std_p[d].item(), v_std_p[d].item()
        print(
            f"{d:<5} {tm:<12.4f} {vm:<12.4f} {abs(tm-vm):<12.4f} | {ts:<12.4f} {vs:<12.4f} {abs(ts-vs):<12.4f}"
        )

    print(f"\nAvg Diff across all {len(t_mean_p)} dims:")
    print(f"Mean Abs Diff: {torch.mean(torch.abs(t_mean_p - v_mean_p)).item():.6f}")
    print(f"Std Abs Diff:  {torch.mean(torch.abs(t_std_p - v_std_p)).item():.6f}")

    print("\n[CONCLUSION]")
    diff_thresh = (
        0.5  # A bit loose since randomly collected interaction data can be noisy
    )
    if (
        torch.mean(torch.abs(t_std_z - v_std_z)) < diff_thresh
        and torch.mean(torch.abs(t_std_p - v_std_p)) < diff_thresh
    ):
        print(">> Distributions are largely consistent. Split is safe.")
    else:
        print(
            ">> CAUTION: Distribution shift detected. Data might be insufficient or non-i.i.d."
        )


if __name__ == "__main__":
    main()
