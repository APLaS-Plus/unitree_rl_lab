from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """返回当前步态相位的 sin/cos 编码"""
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def friction_coefficients(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """返回机器人脚底的静摩擦和动摩擦系数 (特权信息)

    输出: [num_envs, 2] -> [static_friction, dynamic_friction]
    """
    asset = env.scene[asset_cfg.name]
    # 返回形状: [num_envs, num_bodies, 3] -> [static_friction, dynamic_friction, restitution]
    material_props = asset.root_physx_view.get_material_properties()
    # 取第一个 body (base) 的摩擦系数
    friction = material_props[:, 0, :2]  # [num_envs, 2]
    return friction.to(env.device)


def base_mass(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回机器人基座的质量 (特权信息)

    输出: [num_envs, 1] -> [mass]
    """
    asset = env.scene[asset_cfg.name]
    # 获取所有 body 的质量
    # 返回形状: [num_envs, num_bodies]
    body_masses = asset.root_physx_view.get_masses()
    # 取第一个 body (base) 的质量
    base_mass_val = body_masses[:, 0:1]  # [num_envs, 1]
    # normalized = (base_mass_val - 3) / 2.0

    return base_mass_val.to(env.device)
