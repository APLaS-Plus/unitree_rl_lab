# Copyright (c) 2024-2025, Adaptation Module Authors
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for Multi-Head Adaptation Network.

This script trains the adaptation network using collected data.
The data should contain:
    - student_obs: History of proprioception (batch, obs_dim * history_len)
    - teacher_latent: Teacher's first layer output (batch, latent_dim)
    - phys_params: Ground truth physical parameters (batch, phys_dim)
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import MultiHeadAdaptationNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Head Adaptation Network.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the collected dataset (.pt file).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lambda_aux",
        type=float,
        default=0.5,
        help="Weight for auxiliary physics loss.",
    )
    parser.add_argument(
        "--conv_channels",
        type=str,
        default="32,64,128",
        help="Comma-separated output channels for Conv1d layers.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="Kernel size for Conv1d layers.",
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=50,
        help="Number of history steps (must match data).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/adaptation",
        help="Directory to save logs and checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training.",
    )
    return parser.parse_args()


def load_dataset(data_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load collected dataset.

    Expected format: dict with keys 'student_obs', 'teacher_latent', 'phys_params'.
    """
    print(f"[INFO] Loading dataset from: {data_path}")
    data = torch.load(data_path, weights_only=False)

    student_obs = data["student_obs"]  # (N, obs_dim * history_len)
    teacher_latent = data["teacher_latent"]  # (N, latent_dim)
    phys_params = data["phys_params"]  # (N, phys_dim)

    print(f"[INFO] Dataset loaded:")
    print(f"  - student_obs shape: {student_obs.shape}")
    print(f"  - teacher_latent shape: {teacher_latent.shape}")
    print(f"  - phys_params shape: {phys_params.shape}")

    return student_obs, teacher_latent, phys_params


def train(args):
    """Main training loop."""
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Load data
    student_obs, teacher_latent, phys_params = load_dataset(args.data_path)

    # Get dimensions
    input_dim = student_obs.shape[1]
    latent_dim = teacher_latent.shape[1]
    phys_dim = phys_params.shape[1]

    # Parse conv channels
    conv_channels = [int(x) for x in args.conv_channels.split(",")]

    # Calculate obs_dim from input_dim and history_length
    obs_dim = input_dim // args.history_length
    assert obs_dim * args.history_length == input_dim, (
        f"input_dim ({input_dim}) must be divisible by history_length ({args.history_length}). "
        f"Got obs_dim={obs_dim}, remainder={input_dim % args.history_length}"
    )

    print(f"[INFO] Model config:")
    print(f"  - obs_dim: {obs_dim}")
    print(f"  - history_length: {args.history_length}")
    print(f"  - input_dim (obs_dim * history): {input_dim}")
    print(f"  - latent_dim: {latent_dim}")
    print(f"  - phys_dim: {phys_dim}")
    print(f"  - conv_channels: {conv_channels}")
    print(f"  - kernel_size: {args.kernel_size}")

    # Create model
    model = MultiHeadAdaptationNet(
        obs_dim=obs_dim,
        history_length=args.history_length,
        latent_dim=latent_dim,
        phys_dim=phys_dim,
        conv_channels=conv_channels,
        kernel_size=args.kernel_size,
        activation="elu",
    ).to(args.device)

    # Create dataloader
    dataset = TensorDataset(
        student_obs.to(args.device),
        teacher_latent.to(args.device),
        phys_params.to(args.device),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    print(f"[INFO] Starting training for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_loss_latent = 0.0
        total_loss_phys = 0.0

        start_time = time.time()

        for batch_idx, (obs, latent_target, phys_target) in enumerate(dataloader):
            # Forward
            latent_pred, phys_pred = model(obs)

            # Compute losses
            loss_latent = criterion(latent_pred, latent_target)
            loss_phys = criterion(phys_pred, phys_target)
            loss = loss_latent + args.lambda_aux * loss_phys

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_latent += loss_latent.item()
            total_loss_phys += loss_phys.item()

        # Epoch stats
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_loss_latent = total_loss_latent / num_batches
        avg_loss_phys = total_loss_phys / num_batches
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1:4d}/{args.epochs} | "
            f"Loss: {avg_loss:.6f} (Latent: {avg_loss_latent:.6f}, Phys: {avg_loss_phys:.6f}) | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(log_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "obs_dim": obs_dim,
                        "history_length": args.history_length,
                        "latent_dim": latent_dim,
                        "phys_dim": phys_dim,
                        "conv_channels": conv_channels,
                        "kernel_size": args.kernel_size,
                    },
                    "epoch": epoch,
                    "loss": best_loss,
                },
                save_path,
            )
            print(f"  -> Saved best model to {save_path}")

    # Save final model
    final_path = os.path.join(log_dir, "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "obs_dim": obs_dim,
                "history_length": args.history_length,
                "latent_dim": latent_dim,
                "phys_dim": phys_dim,
                "conv_channels": conv_channels,
                "kernel_size": args.kernel_size,
            },
            "epoch": args.epochs,
            "loss": avg_loss,
        },
        final_path,
    )
    print(f"[INFO] Final model saved to {final_path}")
    print(f"[INFO] Training complete! Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
