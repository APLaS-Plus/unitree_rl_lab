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
import json

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


def train(args):
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] Tensorboard logging to: {log_dir}")


import glob
import random
import math


class ChunkedDataset(torch.utils.data.IterableDataset):
    """Iterable Dataset that loads data in chunks from disk."""

    def __init__(self, file_pattern=None, file_list=None, device="cpu", shuffle_chunks=True):
        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted(glob.glob(file_pattern))
            if not self.files:
                # Fallback for single file or mismatch
                if os.path.exists(file_pattern):
                    self.files = [file_pattern]
                else:
                    # Try adding chunk wildcard
                    self.files = sorted(
                        glob.glob(file_pattern.replace(".pt", "_chunk*.pt"))
                    )

        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {file_pattern}")

        self.device = device  # Note: When using num_workers>0, device should usually be 'cpu' here, and to(device) called in loop
        self.shuffle_chunks = shuffle_chunks

        print(f"[INFO] ChunkedDataset found {len(self.files)} files.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Single-process data loading
            file_order = list(self.files)
        else:  # Multi-process data loading, split workload
            per_worker = int(
                math.ceil(len(self.files) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            file_order = list(self.files[iter_start:iter_end])

        if self.shuffle_chunks:
            random.shuffle(file_order)

        for fpath in file_order:
            try:
                # Load to CPU first if using workers, or directly to device if 0 workers
                # Ideally always load to CPU in dataset for separate workers
                data = torch.load(fpath, map_location="cpu")

                # Assume data structure match
                obs = data["student_obs"]
                latent = data["teacher_latent"]
                phys = data["phys_params"]

                # Shuffle within chunk
                if self.shuffle_chunks:
                    perm = torch.randperm(obs.shape[0])  # CPU shuffle
                    obs = obs[perm]
                    latent = latent[perm]
                    phys = phys[perm]

                # Yield items
                # Note: Yielding item by item is slow in Python.
                # It is better to rely on DataLoader's batching if we yield items.
                # However, default DataLoader with IterableDataset expects the iterator to yield samples.
                # To optimize, we could yield batches, but let's stick to standard first.
                for i in range(obs.shape[0]):
                    yield obs[i], latent[i], phys[i]

            except Exception as e:
                print(f"[ERROR] Failed to load {fpath}: {e}")
                continue

    def __len__(self):
        # Approximate length (sum of file sizes / sample size) or just arbitrary
        # We can try to peek at first file and multiply
        # But IterableDataset doesn't enforce __len__.
        return 10000000  # Dummy, doesn't affect training loop much except progress bar


def load_stats(stats_path, device):
    """Load pre-computed stats."""
    print(f"[INFO] Loading stats from: {stats_path}")
    stats = torch.load(stats_path, map_location=device)

    latent_mean = stats["mean"].to(device)
    latent_std = stats["std"].to(device)
    latent_std[latent_std < 1e-6] = 1.0

    # Check for Phys Stats (added in new analysis version)
    if "phys_mean" in stats:
        phys_mean = stats["phys_mean"].to(device)
        phys_std = stats["phys_std"].to(device)
        phys_std[phys_std < 1e-6] = 1.0
        print("[INFO] Found physical parameter statistics.")
    else:
        print("[WARNING] No physical parameter statistics found. Using identity normalization.")
        phys_mean = torch.zeros(1, device=device)
        phys_std = torch.ones(1, device=device)

    return latent_mean, latent_std, phys_mean, phys_std


def train(args):
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Save Hyperparameters
    with open(os.path.join(log_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] Tensorboard logging to: {log_dir}")

    # Check data files
    # We delay loading to dataset init

    # Prepare Normalization Stats
    latent_mean, latent_std, phys_mean, phys_std = load_stats(args.stats_path, args.device)

    # Dimensions - Peek at first file
    first_file = sorted(glob.glob(args.data_path.replace(".pt", "_chunk*.pt")))[0]
    data_peek = torch.load(first_file)
    input_dim = data_peek["student_obs"].shape[1]
    latent_dim = data_peek["teacher_latent"].shape[1]
    phys_dim = data_peek["phys_params"].shape[1]
    obs_dim = input_dim // args.history_length
    conv_channels = [int(x) for x in args.conv_channels.split(",")]

    del data_peek

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

    # --- Data Preparation (Split Train/Val) ---
    all_files = sorted(glob.glob(args.data_path.replace(".pt", "_chunk*.pt")))
    if not all_files:
        # Fallback if specific file passed
        if os.path.exists(args.data_path):
            all_files = [args.data_path]
        else:
            raise FileNotFoundError(f"No files found for {args.data_path}")

    # Random shuffle before split
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(all_files)

    val_ratio = 0.05  # 5% for validation
    n_val = max(1, int(len(all_files) * val_ratio))
    n_train = len(all_files) - n_val

    train_files = all_files[:n_train]
    val_files = all_files[n_train:]

    print(f"[INFO] Data Split: {n_train} Train files, {n_val} Val files.")

    # Dataloaders
    # Use CPU for dataset loading so workers can function
    train_dataset = ChunkedDataset(file_list=train_files, device="cpu")
    val_dataset = ChunkedDataset(
        file_list=val_files, device="cpu", shuffle_chunks=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2,  # Less workers for val
        pin_memory=True,
    )

    print(f"[INFO] Training started. (Streaming data, Workers: 4)")

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

        for obs, target_latent_raw, target_phys_raw in train_dataloader:
            obs = obs.to(args.device)
            target_latent_raw = target_latent_raw.to(args.device)
            target_phys_raw = target_phys_raw.to(args.device)
            # Normalize Targets using Pre-computed Stats
            target_latent = (target_latent_raw - latent_mean) / latent_std
            target_phys = (target_phys_raw - phys_mean) / phys_std

            # Forward
            pred_latent, pred_phys = model(obs)

            # Losses
            loss_latent = criterion(pred_latent, target_latent)
            loss_phys = criterion(pred_phys, target_phys)
            total_loss = loss_latent + lambda_aux * loss_phys

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

            if grad_sim_count == 0:  # Monitor only first batch of epoch
                # 1. Latent Gradients
                optimizer.zero_grad()
                loss_latent.backward(retain_graph=True)

                grads_latent = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads_latent.append(param.grad.view(-1).clone())
                    else:
                        grads_latent.append(torch.zeros(param.numel(), device=args.device))

                # 2. Phys Gradients
                optimizer.zero_grad()
                loss_phys.backward()  # Graph freed here

                grads_phys = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads_phys.append(param.grad.view(-1).clone())
                    else:
                        grads_phys.append(torch.zeros(param.numel(), device=args.device))

                # 3. Compute Cosine Sim
                g1_vec = torch.cat(grads_latent)
                g2_vec = torch.cat(grads_phys)

                # Avoid zero vectors
                if g1_vec.norm() > 1e-6 and g2_vec.norm() > 1e-6:
                    cos_sim = nn.functional.cosine_similarity(
                        g1_vec.unsqueeze(0), g2_vec.unsqueeze(0)
                    ).item()
                    grad_sim_sum += cos_sim
                    grad_sim_count += 1

                # 4. Combine Gradients for Update: total_grad = grad_latent + lambda * grad_phys
                # Currently p.grad contains grad_phys
                with torch.no_grad():
                    idx = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            # param.grad = param.grad * lambda + grad_latent
                            param.grad.mul_(lambda_aux).add_(grads_latent[idx].view_as(param.grad))
                        idx += 1

                optimizer.step()
            else:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            epoch_loss += total_loss.item()
            epoch_latent_loss += loss_latent.item()
            epoch_phys_loss += loss_phys.item()

        # Averages Train
        avg_loss = epoch_loss / max(1, len(train_dataloader))
        avg_latent = epoch_latent_loss / max(1, len(train_dataloader))
        avg_phys = epoch_phys_loss / max(1, len(train_dataloader))
        avg_grad_sim = grad_sim_sum / max(1, grad_sim_count)

        # Logging Train
        writer.add_scalar("Loss/Total", avg_loss, epoch)
        writer.add_scalar("Loss/Latent", avg_latent, epoch)
        writer.add_scalar("Loss/Phys", avg_phys, epoch)
        writer.add_scalar("Monitoring/LambdaAux", lambda_aux, epoch)
        writer.add_scalar("Monitoring/LR", current_lr, epoch)
        if grad_sim_count > 0:
            writer.add_scalar("Monitoring/GradCosineSim", avg_grad_sim, epoch)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_latent = 0.0
        val_phys = 0.0

        with torch.no_grad():
            for obs, target_latent_raw, target_phys_raw in val_dataloader:
                obs = obs.to(args.device)
                target_latent_raw = target_latent_raw.to(args.device)
                target_phys_raw = target_phys_raw.to(args.device)

                target_latent = (target_latent_raw - latent_mean) / latent_std
                target_phys = (target_phys_raw - phys_mean) / phys_std

                pred_latent, pred_phys = model(obs)

                l_latent = criterion(pred_latent, target_latent)
                l_phys = criterion(pred_phys, target_phys)
                t_loss = l_latent + lambda_aux * l_phys

                val_loss += t_loss.item()
                val_latent += l_latent.item()
                val_phys += l_phys.item()

        avg_val_loss = val_loss / max(1, len(val_dataloader))
        avg_val_latent_loss = val_latent / max(1, len(val_dataloader))
        avg_val_phys_loss = val_phys / max(1, len(val_dataloader))

        writer.add_scalar("Val/Loss_Total", avg_val_loss, epoch)
        writer.add_scalar("Val/Loss_Latent", avg_val_latent_loss, epoch)
        writer.add_scalar("Val/Loss_Phys", avg_val_phys_loss, epoch)

        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f} (L: {avg_val_latent_loss:.4f}) | LR: {current_lr:.2e}"
            )

        if (epoch + 1) % 10 == 0:
            # Save Checkpoint
            ckpt_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

        # Step Scheduler
        scheduler.step()

    # Save Model
    save_path = os.path.join(log_dir, "student_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")
    writer.close()


if __name__ == "__main__":
    train(parse_args())
