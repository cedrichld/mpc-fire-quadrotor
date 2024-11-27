# visualize_forest.py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from forrest_generation import generate_random_forest_with_grid
from firezone import *
from astar import a_star_2d
from rrt_star import rrt_star, plot_rrt_attempts
from smooth_path import *
from path_points_generation import *

def visualize_forest(space_dim, obstacles, fire_zone, start_pos, goal_pos, fire_zone_trees=None):
    """
    Visualize the forest environment with cylindrical obstacles, fire zone, and start/goal points.

    Args:
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        obstacles (list): List of CylinderObstacle objects outside the fire zone.
        fire_zone (tuple): Coordinates of the fire zone (x, y).
        start_pos (tuple): Start position of the drone.
        goal_pos (tuple): Goal position of the drone.
        fire_zone_trees (list): Trees inside the fire zone, displayed in orange. (Optional)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_dim[0])
    ax.set_ylim(0, space_dim[1])
    ax.set_zlim(0, space_dim[2])

    # Plot trees outside fire zone
    for obstacle in obstacles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, 1, 10)
        x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.center[0]
        y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.center[1]
        z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.center[2]
        ax.plot_surface(x, y, z, color='green', alpha=0.5)

    # Plot trees inside fire zone (if provided)
    if fire_zone_trees:
        for obstacle in fire_zone_trees:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, 1, 10)
            x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.center[0]
            y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.center[1]
            z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.center[2]
            ax.plot_surface(x, y, z, color='orange', alpha=0.7)

    # Plot fire zone as filled red area
    fire_x, fire_y = fire_zone
    fire_z = np.zeros_like(fire_x)
    ax.plot(fire_x, fire_y, fire_z, color='red', linewidth=2, label='Fire Zone')

    # Plot start and goal positions
    ax.scatter(*start_pos, color='blue', s=100, label='Start')
    ax.scatter(*goal_pos, color='yellow', s=100, label='Goal')

    plt.legend()
    plt.show()

def visualize_forest_2d(space_dim, obstacles, fire_zone, start_pos, goal_pos, fire_zone_trees=None):
    """
    Visualize the forest environment in 2D with cylindrical obstacles, fire zone, and start/goal points.

    Args:
        space_dim (tuple): Dimensions of the 2D space (x, y).
        obstacles (list): List of CylinderObstacle objects outside the fire zone.
        fire_zone (tuple): Coordinates of the fire zone (x, y).
        start_pos (tuple): Start position of the drone.
        goal_pos (tuple): Goal position of the drone.
        fire_zone_trees (list): Trees inside the fire zone, displayed in orange. (Optional)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, space_dim[0])
    ax.set_ylim(0, space_dim[1])

    # Plot trees outside fire zone in green
    for obstacle in obstacles:
        circle = plt.Circle(
            obstacle.center[:2], obstacle.radius, color='green', alpha=0.5
        )
        ax.add_patch(circle)

    # Plot trees inside fire zone in orange
    if fire_zone_trees:
        for obstacle in fire_zone_trees:
            circle = plt.Circle(
                obstacle.center[:2], obstacle.radius, color='orange', alpha=0.7
            )
            ax.add_patch(circle)

    # Plot fire zone boundary
    fire_x, fire_y = fire_zone
    ax.plot(fire_x, fire_y, color='red', linewidth=2, label='Fire Zone')

    # Plot start and goal positions
    ax.scatter(start_pos[0], start_pos[1], color='blue', s=100, label='Start')
    ax.scatter(goal_pos[0], goal_pos[1], color='yellow', s=100, label='Goal')
    
    return fig, ax
    
# Implementation
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize forest environment for path planning.")
    parser.add_argument("mode", choices=["2d", "3d", "a_star_2d", "rrt_star_2d"], 
        help="Visualization mode: '2d' for 2D visualization, '3d' for 3D visualization, 'a_star' for pathfinding.")
    args = parser.parse_args()

    # Define 3D space dimensions and parameters
    space_dim = (50, 50, 20)
    grid_size = 25  # Define a grid
    radius_range = (0.1, 0.25)
    height_range = (5.0, 15.0)
    zone_center = (25, 25)
    zone_radius = 10
    
    # Generate fire zone
    fire_zone = generate_fire_zone(center=zone_center, size=zone_radius)
    
    # Define start and goal positions
    start_pos = (5, 5, 0)
    closest_fire_point = closest_point_on_spline(fire_zone, start_pos[:2])
    goal_pos = (*closest_fire_point, 0)

    min_start_distance = 1.0  # Minimum distance from start position to any tree

    # Generate forest using grid-based approach
    trees_outside, trees_inside = generate_random_forest_with_grid(
        grid_size, radius_range, height_range, space_dim, fire_zone, start_pos, min_start_distance
    )

    # Handle different modes
    if args.mode == "3d":
        visualize_forest(space_dim, trees_outside, fire_zone, start_pos, goal_pos, trees_inside)
    elif args.mode == "2d":
        fig, ax = visualize_forest_2d(space_dim[:2], trees_outside, fire_zone, start_pos, goal_pos, trees_inside)

        # Test points from (0, 0) to (50, 50)
        test_points_x, test_points_y = np.meshgrid(np.linspace(0, 50, 200), np.linspace(0, 50, 200))
        test_points = np.column_stack((test_points_x.ravel(), test_points_y.ravel()))

        # Check if points are in the fire zone
        inside_points = []
        outside_points = []
        for point in test_points:
            if is_point_in_fire_zone(fire_zone, point):
                inside_points.append(point)
            else:
                outside_points.append(point)

        # Separate inside and outside points for visualization
        inside_points = np.array(inside_points)
        outside_points = np.array(outside_points)

        # Plot inside and outside points
        if inside_points.size > 0:
            ax.scatter(inside_points[:, 0], inside_points[:, 1], color='red', s=0.02, label="Inside Fire Zone")
        if outside_points.size > 0:
            ax.scatter(outside_points[:, 0], outside_points[:, 1], color='blue', s=0.02, label="Outside Fire Zone")

        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    ## A*
    elif args.mode == "a_star_2d":
        # Combine trees for obstacle input
        obstacles = trees_outside + trees_inside
        
        # Run A* Pathfinding
        print("\n=== A* Pathfinding ===")

        # Adjust goal position if it is in collision
        adjusted_goal_pos = adjust_goal_position_smoothly(
            goal_pos[:2], obstacles, inflation=0.05, step_size=0.5, max_attempts=50
        )
        
        if adjusted_goal_pos is None:
            print("No valid goal position could be determined.")
            exit()
        goal_pos = (*adjusted_goal_pos, 0)
        
        visualize_forest_2d(space_dim[:2], trees_outside, fire_zone, start_pos, goal_pos, trees_inside)
        
        start_time = time.perf_counter()
        path, visited_positions = a_star_2d(start_pos[:2], goal_pos[:2], obstacles, space_dim[:2], eps=0.5)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000

        if path:
            path_cost = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
            print(f"A* Path found in {elapsed_time:.2f} ms with cost {path_cost:.2f}")

            # Smooth the path
            smoothed_path = smooth_path_with_collision_avoidance(path, obstacles)
            smoothed_path_x, smoothed_path_y = smoothed_path[:2]

            # Visualize in 2D
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(0, space_dim[0])
            ax.set_ylim(0, space_dim[1])

            
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, color='purple', linewidth=2, label='A* Path')

            # Smoothed path
            ax.plot(smoothed_path_x, smoothed_path_y, color='cyan', linestyle='--', linewidth=2, label='Smoothed Path')
        else:
            print(f"A* failed to find a path in {elapsed_time:.2f} ms.")
    
    ## RRT*
    elif args.mode == "rrt_star_2d":
        # Combine trees for obstacle input
        obstacles = trees_outside + trees_inside

        # Run RRT* Pathfinding
        print("\n=== RRT* Pathfinding ===")

        # Adjust goal position if it is in collision
        adjusted_goal_pos = adjust_goal_position_smoothly(
            goal_pos[:2], obstacles, inflation=0.05, step_size=0.5, max_attempts=50
        )
        if adjusted_goal_pos is None:
            print("No valid goal position could be determined.")
            exit()
        goal_pos = (*adjusted_goal_pos, 0)

        fig, ax = visualize_forest_2d(space_dim[:2], trees_outside, fire_zone, start_pos, goal_pos, trees_inside)
        
        start_time = time.perf_counter()
        path, tree = rrt_star(
            start_pos[:2], goal_pos[:2], obstacles, space_dim, dim=2
        )

        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000

        if path:
            path_cost = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
            print(f"RRT* Path found in {elapsed_time:.2f} ms with cost {path_cost:.2f}")

            # Smooth the path
            smoothed_path = smooth_path_with_collision_avoidance(path, obstacles)
            
            smooth_path_disc_test = False
            if (smooth_path_disc_test):            
                # Get points at 1/5, 2/5, and 3/5 of the total trajectory time
                points_to_plot = []  
                velocity_vectors = []
                tf = smooth_path_discretized(smoothed_path)
                print(f"Total time to path: {tf}s at 2.0 m/s")
                
                t_array = np.linspace(0.1, tf - 0.1, 10)
                for t in t_array:
                    point = smooth_path_pos_t(smoothed_path, t, tf)
                    velocity = smooth_path_vel_t(smoothed_path, t, tf)
                    
                    points_to_plot.append(point)
                    velocity_vectors.append(velocity)
                
                # Plot specific points
                for i, point in enumerate(points_to_plot, start=1):
                    ax.scatter(point[0], point[1], color='red', label=f'Point {i} at {i}/5 tf')
                    ax.annotate(f"P{i}", (point[0], point[1]))
                
                points_to_plot = np.array(points_to_plot)
                velocity_vectors = np.array(velocity_vectors)

                # Plot velocity vectors as arrows
                plt.quiver(
                    points_to_plot[:, 0], points_to_plot[:, 1],  # Starting points of arrows
                    velocity_vectors[:, 0], velocity_vectors[:, 1],  # Components of velocity vectors
                    angles='xy', scale_units='xy', scale=1, color='red', label='Velocity Vectors'
                )
            
            
            smoothed_path_x, smoothed_path_y = smoothed_path[:2]
            
            # Generate extinguishing path points
            extinguishing_path = []
            for i in range(len(smoothed_path_x)):
                point = (smoothed_path_x[i], smoothed_path_y[i])
                if is_point_in_fire_zone(fire_zone, point):
                    extinguishing_path.append(point)
                    
            # Visualize in 2D
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, color='purple', linewidth=2, label='RRT* Path')
            ax.plot(smoothed_path_x, smoothed_path_y, color='cyan', linestyle='--', linewidth=2, label='Smoothed Path')
            
            # Generate and visualize extinguishing path
            extinguishing_path = generate_extinguishing_path(fire_zone, step_size=0.5, inward_translation=0.5)
            if extinguishing_path:
                ext_x, ext_y = zip(*extinguishing_path)
                ax.plot(ext_x, ext_y, color='orange', linewidth=1, linestyle='-', label='Extinguishing Path')
                
        else:
            print(f"RRT* failed to find a path in {elapsed_time:.2f} ms.")
        
        # Plot RRT attempts
        # plot_rrt_attempts(ax, tree, dim=2)
            
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
