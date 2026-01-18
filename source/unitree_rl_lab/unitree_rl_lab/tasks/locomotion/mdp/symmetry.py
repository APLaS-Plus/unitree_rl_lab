# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry data augmentation functions for quadruped robot locomotion training.

This module provides functions to generate symmetric observations and actions
for quadruped robots (e.g., Unitree Go2) to improve sample efficiency via
morphological symmetry exploitation.

Go2 Joint Configuration (SDK order):
    Index 0-2:  FR_hip, FR_thigh, FR_calf  (Front Right)
    Index 3-5:  FL_hip, FL_thigh, FL_calf  (Front Left)
    Index 6-8:  RR_hip, RR_thigh, RR_calf  (Rear Right)
    Index 9-11: RL_hip, RL_thigh, RL_calf  (Rear Left)

Symmetry Swap: FR <-> FL, RR <-> RL
"""

from __future__ import annotations

import torch


# ============================================================================
# Permutation Indices for Joint Symmetry
# ============================================================================

# Joint index mapping for left-right swap (12 joints)
# FR (0,1,2) <-> FL (3,4,5), RR (6,7,8) <-> RL (9,10,11)
JOINT_SWAP_INDICES = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Indices of hip joints that need sign flip (hip abduction/adduction)
HIP_JOINT_INDICES = [0, 3, 6, 9]


def _swap_and_flip_joints(tensor: torch.Tensor, dim_offset: int = 0) -> torch.Tensor:
    """Swap left-right joints and flip hip joint signs for a 12-dim joint tensor.

    Args:
        tensor: Tensor of shape [batch, ..., 12, ...]
        dim_offset: The dimension offset where the 12-joint data starts in the last axis

    Returns:
        Symmetrized tensor with swapped and flipped joints
    """
    result = tensor.clone()
    # Extract the 12-joint slice
    joint_slice = result[:, dim_offset : dim_offset + 12]
    # Swap left-right
    swapped = joint_slice[:, JOINT_SWAP_INDICES]
    # Flip hip joints signs
    swapped[:, HIP_JOINT_INDICES] = -swapped[:, HIP_JOINT_INDICES]
    result[:, dim_offset : dim_offset + 12] = swapped
    return result


def compute_symmetric_states(
    env,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute symmetric observations and actions for quadruped robot.

    This function exploits the left-right morphological symmetry of quadruped robots
    to generate augmented training data. For a robot that is symmetric about the
    sagittal plane (front-back axis), the mirrored observation should produce
    mirrored actions.

    The function augments the batch by concatenating the original data with the
    symmetric data. Returns [original; symmetric].

    Observation Structure (Go2 PolicyCfg):
        [0:3]   base_ang_vel       - flip X (Roll), Z (Yaw) components. Keep Y (Pitch).
        [3:6]   projected_gravity  - flip Y component
        [6:9]   velocity_commands  - flip Y, Z components (vy, wz)
        [9:21]  joint_pos_rel      - swap L/R legs, flip hip signs
        [21:33] joint_vel_rel      - swap L/R legs, flip hip signs
        [33:45] last_action        - swap L/R legs, flip hip signs
        --- Privileged Info (if present) ---
        [45:57] joint_effort       - swap L/R legs, flip hip signs
        [57:60] base_lin_vel       - flip Y component
        [60:62] friction           - no change (scalar property)
        [62:63] base_mass          - no change (scalar property)

    Args:
        env: The environment object (passed by rsl_rl, can be used to access env properties).
        obs: Observations tensor of shape [batch, obs_dim] or TensorDict.
        actions: Actions tensor of shape [batch, action_dim].

    Returns:
        Tuple of (augmented_obs, augmented_actions).
        The returned tensors have size [2 * batch, ...].
    """
    # Handle None obs (rsl_rl may call with None obs for action-only transformation)
    if obs is None:
        # Only transform actions
        # Augment batch size by 2: [actions; sym_actions]
        sym_actions = actions.clone()

        # Handle 1D actions
        if actions.dim() == 1:
            sym_actions = sym_actions.unsqueeze(0)
            actions_in = actions.unsqueeze(0)
            squeeze_output = True
        else:
            actions_in = actions
            squeeze_output = False

        actions_swapped = sym_actions[:, JOINT_SWAP_INDICES]
        actions_swapped[:, HIP_JOINT_INDICES] = -actions_swapped[:, HIP_JOINT_INDICES]

        # Concatenate: [original; symmetric]
        augmented_actions = torch.cat([actions_in, actions_swapped], dim=0)

        if squeeze_output:
            augmented_actions = augmented_actions.squeeze(0)

        return None, augmented_actions

    # Handle TensorDict input (rsl_rl may pass obs as TensorDict)
    if hasattr(obs, "keys"):
        # obs is a TensorDict, extract the actual observation tensor
        if "obs" in obs.keys():
            obs_tensor = obs["obs"]
        elif "policy" in obs.keys():
            obs_tensor = obs["policy"]
        else:
            # Try to get the first tensor
            obs_tensor = next(iter(obs.values()))
        is_tensordict = True
        original_obs = obs
    else:
        obs_tensor = obs
        is_tensordict = False

    # Ensure 2D shape [batch, obs_dim]
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
        actions = actions.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = obs_tensor.shape[0]
    obs_dim = obs_tensor.shape[1]
    sym_obs = obs_tensor.clone()
    sym_actions = actions.clone()

    # ========================================================================
    # Transform Observations
    # ========================================================================

    # [0:3] base_ang_vel: [Roll, Pitch, Yaw]
    # Flip Roll (0) and Yaw (2). Keep Pitch (1) unchanged.
    sym_obs[:, 0] = -sym_obs[:, 0]  # Roll rate (X) -> Flip
    # sym_obs[:, 1] = sym_obs[:, 1] # Pitch rate (Y) -> Keep
    sym_obs[:, 2] = -sym_obs[:, 2]  # Yaw rate (Z) -> Flip

    # [3:6] projected_gravity: flip Y component
    sym_obs[:, 4] = -sym_obs[:, 4]

    # [6:9] velocity_commands: flip Y (vy) and Z (wz)
    sym_obs[:, 7] = -sym_obs[:, 7]  # vy
    sym_obs[:, 8] = -sym_obs[:, 8]  # wz

    # [9:21] joint_pos_rel: swap L/R, flip hip signs
    joint_pos = sym_obs[:, 9:21]
    joint_pos_swapped = joint_pos[:, JOINT_SWAP_INDICES]
    joint_pos_swapped[:, HIP_JOINT_INDICES] = -joint_pos_swapped[:, HIP_JOINT_INDICES]
    sym_obs[:, 9:21] = joint_pos_swapped

    # [21:33] joint_vel_rel: swap L/R, flip hip signs
    joint_vel = sym_obs[:, 21:33]
    joint_vel_swapped = joint_vel[:, JOINT_SWAP_INDICES]
    joint_vel_swapped[:, HIP_JOINT_INDICES] = -joint_vel_swapped[:, HIP_JOINT_INDICES]
    sym_obs[:, 21:33] = joint_vel_swapped

    # [33:45] last_action: swap L/R, flip hip signs
    last_action = sym_obs[:, 33:45]
    last_action_swapped = last_action[:, JOINT_SWAP_INDICES]
    last_action_swapped[:, HIP_JOINT_INDICES] = -last_action_swapped[
        :, HIP_JOINT_INDICES
    ]
    sym_obs[:, 33:45] = last_action_swapped

    # ========================================================================
    # Transform Privileged Observations (if present)
    # ========================================================================

    if obs_dim > 45:
        # [45:57] joint_effort: swap L/R, flip hip signs
        if obs_dim >= 57:
            joint_effort = sym_obs[:, 45:57]
            joint_effort_swapped = joint_effort[:, JOINT_SWAP_INDICES]
            joint_effort_swapped[:, HIP_JOINT_INDICES] = -joint_effort_swapped[
                :, HIP_JOINT_INDICES
            ]
            sym_obs[:, 45:57] = joint_effort_swapped

        # [57:60] base_lin_vel: flip Y component
        if obs_dim >= 60:
            sym_obs[:, 58] = -sym_obs[:, 58]

        # [60:62] friction & [62:63] base_mass: no change needed

    # ========================================================================
    # Transform Actions
    # ========================================================================

    # Actions are joint position commands (12 dims): swap L/R, flip hip signs
    actions_swapped = sym_actions[:, JOINT_SWAP_INDICES]
    actions_swapped[:, HIP_JOINT_INDICES] = -actions_swapped[:, HIP_JOINT_INDICES]
    sym_actions = actions_swapped

    # ========================================================================
    # Concatenate and Return
    # ========================================================================

    # Concatenate original and symmetric data: [batch*2, dim]
    augmented_obs_tensor = torch.cat([obs_tensor, sym_obs], dim=0)
    augmented_actions = torch.cat([actions, sym_actions], dim=0)

    # Handle squeeze if input was 1D
    if squeeze_output:
        augmented_obs_tensor = augmented_obs_tensor.squeeze(0)
        augmented_actions = augmented_actions.squeeze(0)

    # Always return plain tensors (matching anymal.py reference implementation)
    return augmented_obs_tensor, augmented_actions
