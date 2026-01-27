"""Analysis script for Teacher Latent Space.

Performs:
1. PCA Analysis on z_t.
2. Visualization of z_t clustering by terrain (2D Projection).
3. Calculation of Global Mean and Std for z_t.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import argparse


import glob


def compute_stats_incremental(file_paths, device="cpu"):
    """Compute mean and std incrementally to save memory."""
    count = 0
    mean = None
    m2 = None  # Sum of squares of differences from the current mean

    print(f"[INFO] Computing stats over {len(file_paths)} chunks...")

    for fpath in file_paths:
        data = torch.load(fpath)
        z_t = data["teacher_latent"].to(device)
        n = z_t.shape[0]

        if count == 0:
            mean = torch.zeros_like(z_t[0])
            m2 = torch.zeros_like(z_t[0])

        # Welford's online algorithm (vectorized for batch)
        # However, for batch updates, standard sum accumulation is numerically stable enough for this scale
        # Let's use standard sum accumulation for simplicity and speed
        pass

    # Re-loop for simple sum/sq_sum (Welford is tricky for batches without caveats)
    # Let's do: Mean = Sum / N, Std = sqrt(SumSq/N - Mean^2)
    sum_z = 0
    sum_sq_z = 0
    total_samples = 0

    for fpath in file_paths:
        data = torch.load(fpath)
        z_t = data["teacher_latent"].to(device)  # (N, dim)

        sum_z += torch.sum(z_t, dim=0)
        sum_sq_z += torch.sum(z_t**2, dim=0)
        total_samples += z_t.shape[0]

    mean = sum_z / total_samples
    variance = (sum_sq_z / total_samples) - (mean**2)
    std = torch.sqrt(torch.clamp(variance, min=1e-6))

    return mean, std, total_samples


def analyze(data_path_pattern, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Find files
    # pattern can be "dataset/teacher_latent_10M_chunk*.pt" or just the base path
    if "*" not in data_path_pattern and "_chunk" not in data_path_pattern:
        # Assume it's a prefix or exact file. If exact file exists, use it.
        if os.path.exists(data_path_pattern):
            files = [data_path_pattern]
        else:
            # Try adding wildcard
            files = sorted(glob.glob(data_path_pattern.replace(".pt", "_chunk*.pt")))
    else:
        files = sorted(glob.glob(data_path_pattern))

    if not files:
        print(f"[ERROR] No files found matching {data_path_pattern}")
        return

    print(f"[INFO] Found {len(files)} data files.")

    # 2. Statistics (Incremental)
    device = "cpu"  # sufficient for stats
    mean, std, total_samples = compute_stats_incremental(files, device)

    print("\n--- Latent Statistics ---")
    print(f"Total Samples: {total_samples}")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")

    # Save stats
    stats_path = os.path.join(output_dir, "latent_stats.pt")
    torch.save({"mean": mean, "std": std}, stats_path)
    print(f"[INFO] Saved stats to {stats_path}")

    # 3. PCA Analysis (Sampled)
    # Load a subset for PCA (e.g., first 100k samples or random chunks)
    # Loading first 2 chunks is usually diverse enough if shuffled, but
    # to be safe, let's just load the first chunk or up to 50k samples
    print("\n--- PCA Analysis (Sampled) ---")

    z_t_sampled = []
    samples_needed = 50000
    samples_collected = 0

    for fpath in files:
        data = torch.load(fpath)
        z_t = data["teacher_latent"]
        z_t_sampled.append(z_t)
        samples_collected += z_t.shape[0]
        if samples_collected >= samples_needed:
            break

    z_t_all = torch.cat(z_t_sampled, dim=0)[:samples_needed]

    # Centralize
    X = z_t_all.cpu().numpy()
    X_centered = X - np.mean(X, axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Explained Variance
    variance = (S**2) / (X.shape[0] - 1)
    total_variance = np.sum(variance)
    explained_variance_ratio = variance / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"Explained Variance Ratio: {explained_variance_ratio}")
    print(f"Cumulative Variance: {cumulative_variance}")

    # Projection to 2D
    components = Vt[:2, :]
    projected = X_centered @ components.T

    # 4. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Cumulative Variance
    latent_dim = X.shape[1]
    axes[0].plot(
        range(1, latent_dim + 1), cumulative_variance, marker="o", linestyle="-"
    )
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Explained Variance")
    axes[0].set_title("PCA Cumulative Variance")
    axes[0].grid(True)
    axes[0].axhline(y=0.95, color="r", linestyle="--", label="95% Explained")
    axes[0].legend()

    # Plot 2: 2D Projection
    axes[1].scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=1)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title(f"Latent Space 2D Projection (First {samples_needed} samples)")
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pca_analysis.png")
    plt.savefig(plot_path)
    print(f"[INFO] Saved PCA plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/teacher_latent_stats.pt",
        help="Path pattern (e.g. dataset/dataset_chunk*.pt)",
    )
    parser.add_argument("--output_dir", type=str, default="dataset/analysis")
    args = parser.parse_args()

    analyze(args.data_path, args.output_dir)
