# Copyright (c) 2024-2025, Adaptation Module Authors
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-Head Adaptation Network for RMA-style distillation.

This module implements the adaptation network that predicts:
1. Teacher's first layer latent features (Main Head)
2. Actual physical parameters (Aux Head for regularization)
"""

import torch
import torch.nn as nn


class MultiHeadAdaptationNet(nn.Module):
    """Multi-Head Adaptation Network with 1D CNN Encoder (TCN-style).

    Architecture:
        Input (History Obs) -> 1D Conv Encoder -> [Latent Head, Phys Head]

    The 1D CNN encoder is better suited for processing temporal observation
    history compared to MLP, as it can capture local temporal patterns.

    Args:
        obs_dim: Dimension of a single observation (without history).
        history_length: Number of history steps (e.g., 50).
        latent_dim: Dimension of Teacher's first layer output (e.g., 512).
        phys_dim: Dimension of privileged/physical parameters (e.g., 17).
        conv_channels: List of output channels for each Conv1d layer.
        kernel_size: Kernel size for Conv1d layers.
        activation: Activation function name ('elu', 'relu', 'tanh').
    """

    def __init__(
        self,
        obs_dim: int,
        history_length: int,
        latent_dim: int,
        phys_dim: int,
        conv_channels: list[int] = [32, 64, 128],
        kernel_size: int = 5,
        activation: str = "elu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.phys_dim = phys_dim

        # For backward compatibility
        self.input_dim = obs_dim * history_length

        # Activation function
        activations = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        act_cls = activations.get(activation, nn.ELU)

        # Build 1D CNN Encoder
        # Input shape: (batch, obs_dim, history_length)
        # We treat obs_dim as "channels" and history_length as "sequence length"
        conv_layers = []
        in_channels = obs_dim
        current_length = history_length

        for out_channels in conv_channels:
            # Padding to maintain sequence length (same padding)
            padding = (kernel_size - 1) // 2
            conv_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    act_cls(),
                ]
            )
            in_channels = out_channels

        self.conv_encoder = nn.Sequential(*conv_layers)

        # Global Average Pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final feature dimension after pooling
        self.feature_dim = conv_channels[-1]

        # MLP projection after conv
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            act_cls(),
            nn.Linear(256, 128),
            act_cls(),
        )

        # Main Head: predicts Teacher's latent features
        self.latent_head = nn.Linear(128, latent_dim)

        # Aux Head: predicts physical parameters (for regularization)
        self.phys_head = nn.Linear(128, phys_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, obs_dim * history_length).
               Will be reshaped to (batch_size, obs_dim, history_length) for Conv1d.

        Returns:
            Tuple of (latent_pred, phys_pred):
                - latent_pred: (batch_size, latent_dim)
                - phys_pred: (batch_size, phys_dim)
        """
        batch_size = x.shape[0]

        # Reshape: (batch, obs_dim * history) -> (batch, obs_dim, history)
        x = x.view(batch_size, self.obs_dim, self.history_length)

        # 1D Conv encoding
        conv_feat = self.conv_encoder(x)  # (batch, channels, seq_len)

        # Global pooling
        pooled = self.global_pool(conv_feat)  # (batch, channels, 1)
        pooled = pooled.squeeze(-1)  # (batch, channels)

        # MLP projection
        feat = self.projection(pooled)

        # Heads
        latent_pred = self.latent_head(feat)
        phys_pred = self.phys_head(feat)

        return latent_pred, phys_pred

    def predict_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode: only predict latent features (skip Aux Head).

        Args:
            x: Input tensor of shape (batch_size, obs_dim * history_length).

        Returns:
            latent_pred: (batch_size, latent_dim)
        """
        with torch.no_grad():
            latent, _ = self.forward(x)
            return latent


class StudentTeacherPolicy(nn.Module):
    """Combined Student-Teacher Policy for RMA-style deployment.

    RMA架构: 学生encoder预测隐向量，与当前本体感知拼接后输入教师actor。

    推理流程:
        1. student_obs (历史本体感知) -> Student Encoder -> predicted_latent
        2. current_proprioception + predicted_latent -> Teacher Actor -> actions

    Args:
        student_net: Trained MultiHeadAdaptationNet instance (encoder only, phys_head will be ignored).
        teacher_actor: Teacher's actor network.
        proprioception_dim: Dimension of current proprioception (single step).
    """

    def __init__(
        self,
        student_net: MultiHeadAdaptationNet,
        teacher_actor: nn.Module,
        proprioception_dim: int,
    ):
        super().__init__()
        self.student = student_net
        self.teacher_actor = teacher_actor
        self.proprioception_dim = proprioception_dim

        # Freeze all parameters for inference
        for param in self.student.parameters():
            param.requires_grad = False
        for param in self.teacher_actor.parameters():
            param.requires_grad = False

    def forward(
        self, student_obs: torch.Tensor, current_proprioception: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for deployment.

        Args:
            student_obs: Student observation (flattened history), shape (batch, obs_dim * history_len).
            current_proprioception: Current step proprioception, shape (batch, proprioception_dim).

        Returns:
            actions: Action output from teacher's policy.
        """
        # Student predicts latent (only use latent head, ignore phys_head)
        latent, _ = self.student(student_obs)

        # Concatenate: [proprioception, predicted_latent] -> teacher actor input
        # 这与RMA论文一致: actor输入是本体感知+隐向量
        actor_input = torch.cat([current_proprioception, latent], dim=-1)

        # Pass through teacher actor
        actions = self.teacher_actor(actor_input)
        return actions

    def forward_with_latent_only(self, student_obs: torch.Tensor) -> torch.Tensor:
        """Alternative forward: only use student encoder output directly.

        用于兼容旧版推理方式，student直接替换teacher第一层。

        Args:
            student_obs: Student observation (flattened history).

        Returns:
            actions: Action output.
        """
        latent, _ = self.student(student_obs)

        # Skip first layer of teacher, feed latent to remaining layers
        layers = list(self.teacher_actor.children())
        x = latent
        for layer in layers[1:]:  # Skip first layer
            x = layer(x)
        return x


class RMAStudentEncoder(nn.Module):
    """Standalone RMA Student Encoder for deployment.

    仅用于推理阶段，去掉了phys_head，只保留latent预测功能。
    可以直接从MultiHeadAdaptationNet加载权重。
    """

    def __init__(
        self,
        obs_dim: int,
        history_length: int,
        latent_dim: int,
        conv_channels: list[int] = [32, 64, 128],
        kernel_size: int = 5,
        activation: str = "elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.latent_dim = latent_dim

        # Activation function
        activations = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        act_cls = activations.get(activation, nn.ELU)

        # Build 1D CNN Encoder (same as MultiHeadAdaptationNet)
        conv_layers = []
        in_channels = obs_dim

        for out_channels in conv_channels:
            padding = (kernel_size - 1) // 2
            conv_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    act_cls(),
                ]
            )
            in_channels = out_channels

        self.conv_encoder = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = conv_channels[-1]

        # MLP projection
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            act_cls(),
            nn.Linear(256, 128),
            act_cls(),
        )

        # Only latent head (no phys_head)
        self.latent_head = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, obs_dim * history_length).

        Returns:
            latent: Predicted latent vector, shape (batch_size, latent_dim).
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, self.obs_dim, self.history_length)

        conv_feat = self.conv_encoder(x)
        pooled = self.global_pool(conv_feat).squeeze(-1)
        feat = self.projection(pooled)
        latent = self.latent_head(feat)

        return latent

    @classmethod
    def from_multi_head_net(
        cls, multi_head_net: MultiHeadAdaptationNet
    ) -> "RMAStudentEncoder":
        """从训练好的MultiHeadAdaptationNet创建encoder (去掉phys_head).

        Args:
            multi_head_net: Trained MultiHeadAdaptationNet.

        Returns:
            RMAStudentEncoder: Encoder-only version for deployment.
        """
        encoder = cls(
            obs_dim=multi_head_net.obs_dim,
            history_length=multi_head_net.history_length,
            latent_dim=multi_head_net.latent_dim,
            # Note: we use the same architecture but don't copy conv_channels/kernel_size
            # since they're already baked into the trained model
        )

        # Copy relevant parameters
        encoder.conv_encoder.load_state_dict(multi_head_net.conv_encoder.state_dict())
        encoder.projection.load_state_dict(multi_head_net.projection.state_dict())
        encoder.latent_head.load_state_dict(multi_head_net.latent_head.state_dict())

        return encoder
