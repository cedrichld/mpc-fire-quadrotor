#!/usr/bin/env python3
"""
main.py
A simple entry point to import all modules in this project, ensuring pydeps
can detect all dependencies from one file.
"""

# flight_controller
import flight_controller.flight_controller
import flight_controller.full_dynamics
import flight_controller.init_quad_constants
import flight_controller.LPV
import flight_controller.MPC_controller
import flight_controller.position_controller
import flight_controller.trajectory_reference

# forest
import forest.firezone
import forest.forest_generation
import forest.plot_forest

# path_planning
import path_planning.obstacles
import path_planning.path_points_generation
import path_planning.rrt
import path_planning.rrt_star
import path_planning.smooth_path

# scripts
import scripts.create_animation
import scripts.rrt_animation
import scripts.visualize_fire_quadrotor

# tests
import tests.create_animation_old

# utils (if you have any .py files or just __init__, this import ensures detection)
import utils


def main():
    """
    Main function: runs nothing special, just ensures that all modules are imported.
    """
    print("All modules imported successfully for pydeps analysis.")


if __name__ == "__main__":
    main()

