# Copyright (c) 2024-2025, RMA Implementation
# SPDX-License-Identifier: BSD-3-Clause

"""RMA-style ActorCritic with Privileged Information Encoder.

This module implements the teacher policy architecture from the RMA paper:
- Privileged info → Encoder → Latent vector (z_t)
- [Proprioception, z_t] → Actor MLP → Actions
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn


class MLP(nn.Module):
    """简单的多层感知机"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
    ):
        super().__init__()

        activations = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        act_cls = activations.get(activation, nn.ELU)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RMAActorCritic(nn.Module):
    """RMA-style ActorCritic with Privileged Information Encoder.

    教师模型架构:
        1. Privileged Info → Encoder (MLP) → Latent (z_t)
        2. [Proprioception, z_t] → Actor MLP → Actions

    Critic直接接收所有信息（本体感知 + 特权信息）进行价值估计。

    Args:
        obs: TensorDict containing observation groups.
        obs_groups: Mapping from policy/critic to observation group names.
        num_actions: Number of action dimensions.
        latent_dim: Dimension of the latent vector from encoder (default: 8, as in RMA paper).
        encoder_hidden_dims: Hidden dimensions for privileged encoder.
        actor_hidden_dims: Hidden dimensions for actor MLP.
        critic_hidden_dims: Hidden dimensions for critic MLP.
        activation: Activation function name.
        init_noise_std: Initial action noise standard deviation.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        latent_dim: int = 8,
        encoder_hidden_dims: list[int] = [256, 128],
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(f"RMAActorCritic got unexpected arguments: {list(kwargs.keys())}")
        super().__init__()

        self.obs_groups = obs_groups
        self.latent_dim = latent_dim

        # 获取各观测组的维度
        # 假设 obs_groups = {"policy": ["policy", "privileged"], "critic": ["policy", "privileged"]}
        # policy组 = 本体感知, privileged组 = 特权信息

        self.proprioception_dim = 0
        self.privileged_dim = 0

        for obs_group in obs_groups.get("policy", []):
            dim = obs[obs_group].shape[-1]
            if obs_group == "policy":
                self.proprioception_dim = dim
            elif obs_group == "privileged":
                self.privileged_dim = dim

        print(f"[RMAActorCritic] Proprioception dim: {self.proprioception_dim}")
        print(f"[RMAActorCritic] Privileged dim: {self.privileged_dim}")
        print(f"[RMAActorCritic] Latent dim: {self.latent_dim}")

        # === Privileged Encoder ===
        # 特权信息 → 隐向量
        self.encoder = MLP(
            input_dim=self.privileged_dim,
            output_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
        )
        print(
            f"[RMAActorCritic] Encoder: {self.privileged_dim} -> {encoder_hidden_dims} -> {latent_dim}"
        )

        # === Actor ===
        # [本体感知, 隐向量] → 动作
        actor_input_dim = self.proprioception_dim + latent_dim
        self.actor = MLP(
            input_dim=actor_input_dim,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )
        print(
            f"[RMAActorCritic] Actor: {actor_input_dim} -> {actor_hidden_dims} -> {num_actions}"
        )

        # === Critic ===
        # Critic可以直接看所有原始信息
        critic_input_dim = self.proprioception_dim + self.privileged_dim
        self.critic = MLP(
            input_dim=critic_input_dim,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        print(
            f"[RMAActorCritic] Critic: {critic_input_dim} -> {critic_hidden_dims} -> 1"
        )

        # === Action Noise ===
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _get_obs_components(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """从观测中提取本体感知和特权信息"""
        proprioception = obs.get("policy", None)
        privileged = obs.get("privileged", None)

        if proprioception is None:
            raise ValueError("Missing 'policy' observation group")
        if privileged is None:
            raise ValueError("Missing 'privileged' observation group")

        return proprioception, privileged

    def _encode_privileged(self, privileged: torch.Tensor) -> torch.Tensor:
        """将特权信息编码为隐向量"""
        return self.encoder(privileged)

    def _update_distribution(self, actor_input: torch.Tensor) -> None:
        """更新动作分布"""
        mean = self.actor(actor_input)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """采样动作（训练时使用）"""
        proprioception, privileged = self._get_obs_components(obs)

        # 编码特权信息
        latent = self._encode_privileged(privileged)

        # 拼接 [本体感知, 隐向量]
        actor_input = torch.cat([proprioception, latent], dim=-1)

        # 更新分布并采样
        self._update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """确定性动作（推理时使用）"""
        proprioception, privileged = self._get_obs_components(obs)

        # 编码特权信息
        latent = self._encode_privileged(privileged)

        # 拼接 [本体感知, 隐向量]
        actor_input = torch.cat([proprioception, latent], dim=-1)

        return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """计算状态价值（Critic）"""
        proprioception, privileged = self._get_obs_components(obs)

        # Critic直接使用原始信息
        critic_input = torch.cat([proprioception, privileged], dim=-1)

        return self.critic(critic_input)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算动作的log概率"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_encoder_output(self, obs: TensorDict) -> torch.Tensor:
        """获取encoder输出的隐向量（用于学生模型训练）"""
        _, privileged = self._get_obs_components(obs)
        return self._encode_privileged(privileged)

    def update_normalization(self, obs: TensorDict) -> None:
        """更新归一化统计（如果需要）"""
        pass  # 当前不使用归一化
