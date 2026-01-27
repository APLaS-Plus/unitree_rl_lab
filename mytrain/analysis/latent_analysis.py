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


def analyze(data_path, output_dir):
    print(f"[INFO] Loading data from {data_path}")
    data = torch.load(data_path)

    # Extract data
    # student_obs = data["student_obs"]
    z_t = data["teacher_latent"]  # (N, latent_dim)
    # phys_params = data["phys_params"]

    latent_dim = z_t.shape[1]
    print(f"[INFO] Latent Shape: {z_t.shape}")

    # 1. Statistics
    mean = torch.mean(z_t, dim=0)
    std = torch.std(z_t, dim=0)

    print("\n--- Latent Statistics ---")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")

    # Save stats
    stats_path = os.path.join(output_dir, "latent_stats.pt")
    torch.save({"mean": mean, "std": std}, stats_path)
    print(f"[INFO] Saved stats to {stats_path}")

    # 2. PCA Analysis
    print("\n--- PCA Analysis ---")
    # Centralize
    X = z_t.cpu().numpy()
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
    # PCA components are rows of Vt
    components = Vt[:2, :]
    projected = X_centered @ components.T

    # 3. Visualization
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Cumulative Variance
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
    # Since we don't strictly have terrain labels per sample in the simple collection (improving this would require storing terrain ID),
    # we will just plot the density or simple scatter.
    # If the user implemented terrain ID logging, we would color by it.
    # For now, we assume "terrain diversity" implies effective clustering if the teacher is robust.
    axes[1].scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=1)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Latent Space 2D Projection (PCA)")
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pca_analysis.png")
    plt.savefig(plot_path)
    print(f"[INFO] Saved PCA plot to {plot_path}")

    # plt.show() # Disabled for headless environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="dataset/teacher_latent_stats.pt"
    )
    parser.add_argument("--output_dir", type=str, default="dataset/analysis")
    args = parser.parse_args()

    analyze(args.data_path, args.output_dir)
