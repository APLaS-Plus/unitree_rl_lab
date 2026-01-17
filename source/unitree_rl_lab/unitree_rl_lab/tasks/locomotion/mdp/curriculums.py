from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def reward_weight_decay(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    initial_weight: float,
    target_weight: float,
) -> torch.Tensor:
    """Decay reward weight based on terrain levels.

    This function modifies the weight of a reward term linearly based on the average terrain level.
    Args:
        env: The environment.
        env_ids: The environment IDs (not used, as this is a global modification).
        reward_term_name: The name of the reward term to modify.
        initial_weight: The weight at terrain level 0.
        target_weight: The weight at the maximum terrain level (assumed to be 9).
    """
    try:
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    except Exception:
        return torch.tensor(0.0, device=env.device)

    # Get average terrain level
    if hasattr(env.scene.terrain, "terrain_levels"):
        avg_level = torch.mean(env.scene.terrain.terrain_levels.float()).item()
    else:
        avg_level = 0.0

    # Assume max level is 9 (standard for Isaac Lab terrains usually 0-9)
    max_level = 9.0
    alpha = min(avg_level / max_level, 1.0)
    alpha = max(0.0, alpha)

    # Linear interpolation
    new_weight = initial_weight + (target_weight - initial_weight) * alpha

    # Update weight
    term_cfg.weight = new_weight

    return torch.tensor(new_weight, device=env.device)
