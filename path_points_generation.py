# path_points_generation.py

import numpy as np
from obstacles import is_point_in_collision
from scipy.interpolate import splprep, splev

def adjust_goal_position_smoothly(goal_pos, obstacles, inflation, dim=2, step_size=0.25, max_attempts=50):
    """
    Adjust the goal position to ensure it is not in collision.

    Args:
        goal_pos (tuple): Initial goal position (x, y).
        obstacles (list): List of CylinderObstacle objects.
        inflation (float): Amount to inflate the obstacle radii for safety.
        step (float): Step size for searching nearby valid positions.
        max_attempts (int): Maximum number of attempts to adjust the position.

    Returns:
        tuple: Adjusted goal position (x, y).
    """
    for _ in range(max_attempts):
        if not is_point_in_collision(goal_pos, obstacles, inflation):
            return goal_pos  # Valid position found
        # Move goal slightly in random direction
        direction = np.random.uniform(-1, 1, size=dim)
        goal_pos = np.array(goal_pos) + step_size * direction / np.linalg.norm(direction)
    raise RuntimeError("Failed to adjust goal position within constraints.")

def generate_extinguishing_path(fire_zone, step_size=5.0, inward_translation=3.0):
    """
    Generate a spiral-like extinguishing path by following the boundary of the fire zone
    and progressively shrinking inward.

    Args:
        fire_zone (tuple): The fire zone spline (x, y).
        step_size (float): The resolution of the points on the path.
        inward_translation (float): Amount to translate inward after each loop.

    Returns:
        list: The extinguishing path as a list of points (x, y).
    """
    x_spline, y_spline = fire_zone
    extinguishing_path = []
    current_zone = np.column_stack((x_spline, y_spline))
    
    for _ in range(10):
        print(f"len: {len(current_zone)}")
        # Add current zone boundary to path
        extinguishing_path.extend(current_zone)

        # Move inward by translating each point toward the centroid
        centroid = np.mean(current_zone, axis=0)
        current_zone = current_zone + inward_translation * (centroid - current_zone) / np.linalg.norm(
            centroid - current_zone, axis=1, keepdims=True
        )

        # Ensure points stay dense along the boundary
        tck, _ = splprep(current_zone.T, s=0.5, per=True)
        u = np.linspace(0, 1, int(len(current_zone) / step_size))
        current_zone = np.column_stack(splev(u, tck))

    return extinguishing_path
