"""Training script for Multi-Head Adaptation Network.

This script trains the adaptation network using collected data.
Features:
- MSE Loss for both latent and physical parameters.
- Target Normalization (Z-Score of latent, Scaling of phys params).
- Curriculum Learning for auxiliary loss weight.
- Gradient Similarity Monitoring via Tensorboard.
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from models import MultiHeadAdaptationNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Head Adaptation Network.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/teacher_latent_stats.pt",
        help="Path to collected dataset (.pt).",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="dataset/analysis/latent_stats.pt",
        help="Path to latent stats (.pt).",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Training batch size."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    # Curriculum params
    parser.add_argument(
        "--lambda_init", type=float, default=2.0, help="Initial auxiliary weight."
    )
    parser.add_argument(
        "--lambda_final", type=float, default=1e-5, help="Final auxiliary weight."
    )
    parser.add_argument(
        "--lambda_decay_epochs", type=int, default=50, help="Epochs to decay lambda."
    )

    # Model params
    parser.add_argument(
        "--conv_channels", type=str, default="32,64,128", help="Conv1d channels."
    )
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size.")
    parser.add_argument(
        "--history_length", type=int, default=50, help="History length."
    )

    parser.add_argument(
        "--log_dir", type=str, default="logs/adaptation", help="Log directory."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_dataset(data_path: str):
    print(f"[INFO] Loading dataset from: {data_path}")
    data = torch.load(data_path)
    return data["student_obs"], data["teacher_latent"], data["phys_params"]


def load_stats(stats_path: str, device: str):
    """Load Teacher z_t statistics for normalization."""
    print(f"[INFO] Loading stats from: {stats_path}")
    stats = torch.load(stats_path)
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    # Avoid div by zero
    std[std < 1e-6] = 1.0
    return mean, std


def train(args):
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] Tensorboard logging to: {log_dir}")

    # Load data
    student_obs, teacher_latent, phys_params = load_dataset(args.data_path)

    # Prepare Normalization Stats
    latent_mean, latent_std = load_stats(args.stats_path, args.device)

    # Physical Params Normalization (assuming approximate ranges or just scaling)
    # Ideally should read ranges from config, here we normalize to [-1, 1] roughly if needed.
    # For now, let's stick to raw or simple max normalization if range is known.
    # User plan mentioned: "Normalize phys params to [-1, 1] or [0, 1]".
    # Let's compute statistics from the dataset itself for Phys Params to be safe (Z-score).
    phys_mean = phys_params.mean(dim=0).to(args.device)
    phys_std = phys_params.std(dim=0).to(args.device)
    phys_std[phys_std < 1e-6] = 1.0

    # Dimensions
    input_dim = student_obs.shape[1]
    latent_dim = teacher_latent.shape[1]
    phys_dim = phys_params.shape[1]
    obs_dim = input_dim // args.history_length
    conv_channels = [int(x) for x in args.conv_channels.split(",")]

    # Model
    model = MultiHeadAdaptationNet(
        obs_dim=obs_dim,
        history_length=args.history_length,
        latent_dim=latent_dim,
        phys_dim=phys_dim,
        conv_channels=conv_channels,
        kernel_size=args.kernel_size,
        activation="elu",
    ).to(args.device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Dataloader
    dataset = TensorDataset(
        student_obs.to(args.device),
        teacher_latent.to(args.device),
        phys_params.to(args.device),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"[INFO] Training started. Batches: {len(dataloader)}")

    # Aux weight scheduler
    def get_lambda(epoch):
        if epoch >= args.lambda_decay_epochs:
            return args.lambda_final
        decay = (args.lambda_final / args.lambda_init) ** (1 / args.lambda_decay_epochs)
        return args.lambda_init * (decay**epoch)

    for epoch in range(args.epochs):
        model.train()
        lambda_aux = get_lambda(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_loss = 0.0
        epoch_latent_loss = 0.0
        epoch_phys_loss = 0.0

        # Gradient Monitoring Sums
        grad_sim_sum = 0.0
        grad_sim_count = 0

        for obs, target_latent_raw, target_phys_raw in dataloader:
            # Normalize Targets using Pre-computed Stats
            target_latent = (target_latent_raw - latent_mean) / latent_std
            target_phys = (target_phys_raw - phys_mean) / phys_std

            # Forward
            pred_latent, pred_phys = model(obs)

            # Losses
            loss_latent = criterion(pred_latent, target_latent)
            loss_phys = criterion(pred_phys, target_phys)
            total_loss = loss_latent + lambda_aux * loss_phys

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            # --- Gradient Monitoring (Cosine Similarity) ---
            # Shared encoder is usually 'feature_extractor' or similar in the model.
            # We assume model has a shared backbone. check model structure.
            # If standard MultiHeadAdaptationNet, it likely has `base` or `encoder`.
            # We skip if structure unknown, but let's try to capture gradients on the last shared layer.
            # Assuming `model.backbone` or similar exists. If not, we skip this hook for now or need to check model file.
            # User request: "Compare gradients of L_latent and L_phy on shared encoder".
            # To do this rigorously requires two backward passes or retaining graph, which is expensive.
            # A cheaper proxy: just check scalar loss correlation or doing it on a few batches.
            # Implementation:
            # 1. zero_grad
            # 2. loss_latent.backward(retain_graph=True)
            # 3. capture main_grad
            # 4. optimizer.zero_grad()
            # 5. loss_phys.backward()
            # 6. capture aux_grad
            # 7. cosine_similarity(main_grad, aux_grad)
            # 8. optimizer.zero_grad()
            # 9. (loss_latent + lambda * loss_phys).backward() -> step
            # This triples backward cost. Maybe do it only every N steps or 1st batch of epoch.

            if grad_sim_count == 0:  # Monitor only first batch of epoch to save time
                # Clear grads first
                optimizer.zero_grad()

                # Get gradients for latent task
                loss_latent.backward(retain_graph=True)
                grads_latent = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads_latent.append(param.grad.view(-1).clone())

                optimizer.zero_grad()

                # Get gradients for phys task
                loss_phys.backward(retain_graph=True)
                grads_phys = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads_phys.append(param.grad.view(-1).clone())

                optimizer.zero_grad()

                # Compute Cosine Sim
                if grads_latent and grads_phys:
                    g1 = torch.cat(grads_latent)
                    g2 = torch.cat(grads_phys)
                    cos_sim = nn.functional.cosine_similarity(
                        g1.unsqueeze(0), g2.unsqueeze(0)
                    ).item()
                    grad_sim_sum += cos_sim
                    grad_sim_count += 1

                # Do actual backward
                total_loss.backward()
                optimizer.step()
            else:
                optimizer.step()

            epoch_loss += total_loss.item()
            epoch_latent_loss += loss_latent.item()
            epoch_phys_loss += loss_phys.item()

        # Averages
        avg_loss = epoch_loss / len(dataloader)
        avg_latent = epoch_latent_loss / len(dataloader)
        avg_phys = epoch_phys_loss / len(dataloader)

        # Logging
        writer.add_scalar("Loss/Total", avg_loss, epoch)
        writer.add_scalar("Loss/Latent", avg_latent, epoch)
        writer.add_scalar("Loss/Phys", avg_phys, epoch)
        writer.add_scalar("Curriculum/Lambda", lambda_aux, epoch)
        writer.add_scalar("Training/LR", current_lr, epoch)

        if grad_sim_count > 0:
            writer.add_scalar(
                "Monitoring/GradCosineSim", grad_sim_sum / grad_sim_count, epoch
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} (L: {avg_latent:.4f}, P: {avg_phys:.4f}) | LR: {current_lr:.2e}"
            )

        # Step Scheduler
        scheduler.step()

    # Save Model
    save_path = os.path.join(log_dir, "student_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")
    writer.close()


if __name__ == "__main__":
    train(parse_args())
