# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a custom terrain curriculum using the Terrain Generator.
It generates a terrain with multiple sub-terrains:
- Rough Terrain (Random Uniform)
- Stairs (Pyramid Stairs)
- Slopes (Pyramid Sloped)
- Discrete Stepping Stones (Discrete Obstacles)
"""

import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Generate complex terrain with IsaacLab.")
# append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg

# Import specific terrain configurations
from isaaclab.terrains.height_field import (
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
)
from isaaclab.terrains.trimesh import MeshPyramidStairsTerrainCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the custom terrain scene."""

    # 2. Add the terrain to the scene
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            # General settings
            size=(8.0, 8.0),  # Size of each terrain patch (length, width)
            border_width=20.0,
            num_rows=4,  # Number of rows in the grid
            num_cols=4,  # Number of columns in the grid
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,  # Disable cache to ensure regeneration during testing
            # Sub-terrains definition
            sub_terrains={
                "rough_terrain": HfRandomUniformTerrainCfg(
                    proportion=0.25,
                    noise_range=(0.02, 0.10),
                    noise_step=0.02,
                    border_width=0.25,
                ),
                "stairs": MeshPyramidStairsTerrainCfg(
                    proportion=0.25,
                    step_height_range=(0.05, 0.20),
                    step_width=0.3,
                    border_width=0.25,
                ),
                "slopes": HfPyramidSlopedTerrainCfg(
                    proportion=0.25,
                    slope_range=(0.05, 0.4),  # Slope in radians
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
                    proportion=0.25,
                    num_obstacles=20,
                    obstacle_height_mode="choice",
                    obstacle_height_range=(0.1, 0.2),  # Height of obstacles
                    obstacle_width_range=(0.2, 0.8),  # Width of obstacles
                    platform_width=2.0,
                    border_width=0.25,
                ),
            },
        ),
        max_init_terrain_level=9,  # Initialize at max difficulty for visualization
        collision_group=-1,  # Default collision group
    )

    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


def main():
    """Main function to generate and visualize the terrain."""

    # Initialize the scene configuration
    scene_cfg = MySceneCfg(
        num_envs=1,
        env_spacing=2.0,
        lazy_sensor_update=True,
    )

    # 3. Setup the Simulation Environment
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set the main camera view
    sim.set_camera_view(eye=[15.0, 15.0, 15.0], target=[0.0, 0.0, 0.0])

    # 4. Create the Scene
    scene = InteractiveScene(scene_cfg)

    # 5. Play the Simulation
    sim.reset()
    print("[INFO]: Setup complete. Playing simulation...")

    while simulation_app.is_running():
        # Step the simulation
        sim.step()
        # Update the scene (physics, sensors, etc.)
        scene.update(dt=sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()

