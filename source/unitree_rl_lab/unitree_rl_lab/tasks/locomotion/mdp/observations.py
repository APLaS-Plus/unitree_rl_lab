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


def feet_heights_relative(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """返回四只脚相对地面的高度 (特权信息)

    使用body位置的z坐标减去地形高度（如果有terrain）

    输出: [num_envs, 4] -> [FL_foot, FR_foot, RL_foot, RR_foot]
    """
    asset = env.scene[asset_cfg.name]

    # 获取脚部body的索引和世界坐标
    # Go2的脚部body名称: FL_foot, FR_foot, RL_foot, RR_foot
    foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_indices = asset.find_bodies(foot_body_names)[0]

    # body_pos_w shape: [num_envs, num_bodies, 3]
    foot_positions = asset.data.body_pos_w[:, foot_indices, :]  # [N, 4, 3]

    # 获取地形高度
    if hasattr(env.scene, "terrain") and env.scene.terrain is not None:
        # 使用terrain的height_lookup获取每个脚下方的地形高度
        # foot_positions[..., :2] 是 xy 坐标
        terrain = env.scene.terrain
        if hasattr(terrain, "flat_terrain_origins"):
            # 计算脚的绝对xy位置
            foot_xy = foot_positions[:, :, :2]  # [N, 4, 2]
            # 获取地形高度 - 简化处理，使用env地形高度
            # 这里使用机器人根部下方的地形高度作为近似
            terrain_heights = env.scene.terrain.data.root_terrain_heights.unsqueeze(
                -1
            )  # [N, 1]
            terrain_heights = terrain_heights.expand(-1, 4)  # [N, 4]
        else:
            terrain_heights = torch.zeros(env.num_envs, 4, device=env.device)
    else:
        terrain_heights = torch.zeros(env.num_envs, 4, device=env.device)

    # 相对高度 = 脚部z坐标 - 地形高度
    relative_heights = foot_positions[:, :, 2] - terrain_heights  # [N, 4]

    return relative_heights.to(env.device)
