# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from unitree_rl_lab.tasks.locomotion import mdp


@configclass
class RslRlRMAActorCriticCfg:
    """RMA-style ActorCritic配置 (带特权信息Encoder).

    架构:
        特权信息 → Encoder → 隐向量 (latent_dim)
        [本体感知, 隐向量] → Actor → 动作
    """

    class_name: str = "RMAActorCritic"  # 自定义类名

    # Encoder配置 (特权信息 → 隐向量)
    latent_dim: int = 8  # RMA论文默认值
    encoder_hidden_dims: list[int] = [256, 128]

    # Actor配置
    actor_hidden_dims: list[int] = [512, 512, 256]

    # Critic配置
    critic_hidden_dims: list[int] = [512, 512, 256]

    # 通用配置
    activation: str = "elu"
    init_noise_std: float = 1.0


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500_00
    save_interval = 2000
    experiment_name = ""  # same as task name
    empirical_normalization = False

    # RMA Teacher: obs_groups配置
    # policy和critic都需要访问policy和privileged观测组
    obs_groups = {
        "policy": ["policy", "privileged"],  # Actor需要两者来构建输入
        "critic": ["policy", "privileged"],  # Critic直接使用两者
    }

    # 使用RMA风格的ActorCritic (带Encoder)
    policy = RslRlRMAActorCriticCfg(
        latent_dim=8,
        encoder_hidden_dims=[256, 128],
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
        init_noise_std=1.0,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# 备选配置: 使用标准ActorCritic (不带Encoder，直接拼接)
@configclass
class StandardPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """标准PPO配置 (不使用RMA Encoder，直接拼接特权信息)"""

    num_steps_per_env = 24
    max_iterations = 500_00
    save_interval = 2000
    experiment_name = ""
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
