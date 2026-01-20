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

    使用 height_scanner (RayCaster) 数据来估算每只脚下的地形高度。
    采用距离加权平均的方法：取脚周围一定半径内的扫描点高度平均值。

    输出: [num_envs, 4] -> [FL_foot, FR_foot, RL_foot, RR_foot]
    """
    asset = env.scene[asset_cfg.name]

    # 1. 获取脚部世界坐标
    # Go2的脚部body名称: FL_foot, FR_foot, RL_foot, RR_foot
    foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_indices = asset.find_bodies(foot_body_names)[0]
    foot_pos_w = asset.data.body_pos_w[:, foot_indices, :]  # [N, 4, 3]

    # 2. 获取地形高度扫描数据
    if "height_scanner" in env.scene.sensors:
        scanner = env.scene.sensors["height_scanner"]
        # scanner.data.ray_hits_w shape: [N, num_rays, 3]
        scan_pos_w = scanner.data.ray_hits_w

        # 3. 计算距离矩阵 (XY平面)
        # scan_pos_w: [N, K, 3]
        # foot_pos_w: [N, 4, 3]
        # expanded: [N, 4, K, 2]
        diff_xy = foot_pos_w[:, :, None, :2] - scan_pos_w[:, None, :, :2]
        dists_sq = torch.sum(diff_xy.square(), dim=-1)  # [N, 4, K]

        # 4. 筛选范围内的点并求平均
        radius = 0.15
        radius_sq = radius**2
        mask = dists_sq < radius_sq  # [N, 4, K]

        # 获取扫描点高度并应用掩码
        scan_heights = scan_pos_w[..., 2]  # [N, K]
        # 扩展 scan_heights 以匹配 mask: [N, 1, K] -> [N, 4, K]
        scan_heights_exp = scan_heights.unsqueeze(1).expand(-1, 4, -1)

        # 计算加权和
        # 注意: 如果某个脚周围没有扫描点，mask全为False
        weights = mask.float()
        total_weight = weights.sum(dim=-1)  # [N, 4]

        # 计算平均高度 (避免除零)
        avg_heights = (scan_heights_exp * weights).sum(dim=-1) / (total_weight + 1e-6)

        # Fallback: 如果没有点在范围内，取最近的一个点的高度
        # min_indices: [N, 4] - 每只脚对应的最近扫描点索引
        min_dists, min_indices = torch.min(dists_sq, dim=-1)
        # 从 scan_heights_exp [N, 4, K] 中按 min_indices [N, 4] 索引
        fallback_heights = torch.gather(
            scan_heights_exp, 2, min_indices.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [N, 4]

        has_neighbors = total_weight > 0.5
        terrain_heights = torch.where(
            has_neighbors, avg_heights, fallback_heights
        )  # [N, 4]

    else:
        # Fallback if sensor missing (should not happen in config)
        terrain_heights = torch.zeros(env.num_envs, 4, device=env.device)
        if hasattr(env.scene, "terrain") and env.scene.terrain is not None:
            if hasattr(env.scene.terrain, "data"):
                # Fallback to root height if totally failed
                terrain_heights = env.scene.terrain.data.root_terrain_heights.unsqueeze(
                    -1
                ).expand(-1, 4)

    # 5. 计算相对高度
    relative_heights = foot_pos_w[:, :, 2] - terrain_heights  # [N, 4]

    return relative_heights.to(env.device)
