"""Script to collect data for student policy training.

Runs the teacher policy in a custom 'Audit' environment with diverse terrains
and strict forward walking commands. Collects:
- Student observations (proprioception history)
- Teacher latent vectors (z_t)
- Physical parameters (privileged info)
- Terrain metadata (optional)
"""

import argparse
import os
import time
import sys

# Import AppLauncher first
from isaaclab.app import AppLauncher
import cli_args  # isort: skip


def main():
    parser = argparse.ArgumentParser(description="Collect Teacher Latent Data.")
    parser.add_argument(
        "--task", type=str, default="Unitree-Go2-Velocity", help="Name of the task."
    )
    parser.add_argument(
        "--num_envs", type=int, default=100, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="Total steps to run."
    )
    parser.add_argument(
        "--data_save_path",
        type=str,
        default="dataset/teacher_latent_stats.pt",
        help="Output path.",
    )

    # RSL-RL and AppLauncher args
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch App
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # ---------------------------------------------------------
    # Perform other imports AFTER SimulationApp is instantiated
    # ---------------------------------------------------------
    import torch
    import gymnasium as gym
    from importlib.metadata import version
    from tensordict import TensorDict

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.dict import print_dict
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from rsl_rl.runners import OnPolicyRunner

    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    # Import custom audit config
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
        RobotDataAuditEnvCfg,
    )

    # --- Environment Setup ---
    # Load default cfg
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )

    # PATCH: Swap to our Audit Config
    print("[INFO] Applying Audit Environment Configuration...")
    audit_cfg = RobotDataAuditEnvCfg()
    audit_cfg.scene.num_envs = args_cli.num_envs

    # Replace scene terrain and commands
    env_cfg.scene.terrain.terrain_generator = audit_cfg.scene.terrain.terrain_generator
    env_cfg.commands = audit_cfg.commands

    # RSL-RL config
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    # Load Checkpoint
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # Create Env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load Policy
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Get Policy Module (for Encoder)
    try:
        policy_module = runner.alg.policy
    except AttributeError:
        policy_module = runner.alg.actor_critic

    has_encoder = hasattr(policy_module, "get_encoder_output")
    if not has_encoder:
        print(
            "[WARNING] Policy does not have 'get_encoder_output'. Using hook on first layer."
        )
        teacher_actor = policy_module.actor
        first_layer_output = {}

        def hook_fn(module, input, output):
            first_layer_output["latent"] = output.detach()

        hook_handle = list(teacher_actor.children())[0].register_forward_hook(hook_fn)

    # --- Data Collection Loop ---
    obs, _ = env.get_observations()

    student_obs_list = []
    teacher_latent_list = []
    phys_params_list = []

    print(f"[INFO] Collecting data for {args_cli.max_steps} steps...")

    with torch.inference_mode():
        for step in range(args_cli.max_steps):
            # Act
            actions = policy(obs)

            # Record Data
            raw_obs = env.unwrapped.observation_manager.compute()

            # 1. Student Obs
            if "student_policy" in raw_obs:
                student_obs_list.append(raw_obs["student_policy"].clone().cpu())

            # 2. Teacher Latent
            if has_encoder:
                obs_td = TensorDict(raw_obs, batch_size=[raw_obs["policy"].shape[0]])
                latent = policy_module.get_encoder_output(obs_td)
                teacher_latent_list.append(latent.clone().cpu())
            else:
                teacher_latent_list.append(first_layer_output["latent"].clone().cpu())

            # 3. Phys Params (Privileged)
            if "privileged" in raw_obs:
                phys_params_list.append(raw_obs["privileged"].clone().cpu())

            # Step Env
            obs, _, _, _ = env.step(actions)

            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{args_cli.max_steps}")

    # --- Save Data ---
    os.makedirs(os.path.dirname(args_cli.data_save_path), exist_ok=True)

    data = {
        "student_obs": torch.cat(student_obs_list, dim=0),
        "teacher_latent": torch.cat(teacher_latent_list, dim=0),
        "phys_params": torch.cat(phys_params_list, dim=0),
        # Save config info if useful
        "velocity_range": [0.5, 1.0],
        "terrains": "audit_mix",
    }

    torch.save(data, args_cli.data_save_path)
    print(
        f"[INFO] Saved {len(data['student_obs'])} samples to {args_cli.data_save_path}"
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
