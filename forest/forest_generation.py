# forrest_generation.py

import numpy as np
from path_planning.obstacles import CylinderObstacle
from .firezone import is_point_in_fire_zone

def generate_random_forest_with_grid(
    grid_size, radius_range, height_range, space_dim, fire_zone, start_pos, min_start_distance
):
    """
    Generate a forest of cylindrical obstacles (Trees) using a grid-based approach.

    Args:
        grid_size (int): Number of grid cells along one axis (total grid cells = grid_size^2).
        radius_range (tuple): Min and max radius for tree cylinders.
        height_range (tuple): Min and max height for tree cylinders.
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        zone_center (tuple): Center of the fire zone (x, y).
        zone_radius (float): Approximate radius for the fire zone boundary.
        start_pos (tuple): Start position (x, y, z).
        min_start_distance (float): Minimum allowed distance between trees and the start position.

    Returns:
        tuple: (list of CylinderObstacle objects, list of CylinderObstacle objects in fire zone)
    """
    trees_outside = []
    trees_inside = []

    # Calculate the size of each grid cell
    grid_cell_size_x = space_dim[0] / grid_size
    grid_cell_size_y = space_dim[1] / grid_size

    # Iterate over grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            # Compute the center of the grid cell
            cell_min_x = i * grid_cell_size_x
            cell_max_x = (i + 1) * grid_cell_size_x
            cell_min_y = j * grid_cell_size_y
            cell_max_y = (j + 1) * grid_cell_size_y

            # Randomly place a tree within the grid cell
            radius = np.random.uniform(*radius_range)
            height = np.random.uniform(*height_range)
            x = np.random.uniform(cell_min_x + radius, cell_max_x - radius)
            y = np.random.uniform(cell_min_y + radius, cell_max_y - radius)

            center = np.array([x, y, 0])

            # Check if the tree is too close to the start position
            if np.linalg.norm(center[:2] - np.array(start_pos[:2])) < (radius + min_start_distance):
                continue
            if is_point_in_fire_zone(fire_zone, center[:2]):
                trees_inside.append(CylinderObstacle(center=center, height=height, radius=radius))
            else:
                trees_outside.append(CylinderObstacle(center=center, height=height, radius=radius))

    return trees_outside, trees_inside

