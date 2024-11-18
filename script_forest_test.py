# script_forest_test.py
from visualize_forest import *

from tqdm import tqdm
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev, Rbf
from obstacles import CylinderObstacle, is_point_in_collision
from astar import a_star_2d, a_star_3d
from rrt_star import rrt_star, plot_rrt_attempts
from smooth_path import smooth_path_with_collision_avoidance

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize forest environment for path planning.")
    parser.add_argument("mode", choices=["2d", "3d", "a_star_2d", "rrt_star_2d_testing"], 
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
        grid_size, radius_range, height_range, space_dim, zone_center, zone_radius, start_pos, min_start_distance
    )

    if args.mode == "rrt_star_2d_testing":
        # Combine trees for obstacle input
        obstacles = trees_outside + trees_inside

        # Parameter ranges to test
        retries_range = [5, 7, 10, 15, 20]#np.linspace(1, 30, 15, dtype=int)  # Vary the number of retries
        max_iter_range = [50, 100, 150, 200, 250, 300] # np.linspace(50, 500, 10, dtype=int)  # Vary the maximum number of iterations
        test_points = 1  # Number of test runs per configuration
        results = []  # To store (retries, max_iter, avg_elapsed_time, avg_path_cost)
        
        # List to store failed configurations
        failed_cases = []
        # List to store the number of iterations for successful configurations
        iteration_counts = []

        # Iterate through retries and max iterations
        for retries in tqdm(retries_range, desc="Testing RRT*", unit="Config"):
            for max_iter in max_iter_range:
                elapsed_times = []
                path_costs = []

                for test_point in range(test_points):
                    # Dynamic print to avoid clutter
                    tqdm.write(f"\r=== Test #{test_point + 1} === Testing RRT* with retries={retries}, max_iter={max_iter} ===", end="")

                    # Adjust goal position if it is in collision
                    adjusted_goal_pos = adjust_goal_position_smoothly(
                        goal_pos[:2], obstacles, inflation=0.05, step_size=0.5, max_attempts=50
                    )
                    if adjusted_goal_pos is None:
                        tqdm.write(f"\nTest failed: retries={retries}, max_iter={max_iter}, test_point={test_point}.")
                        failed_cases.append((retries, test_point, max_iter, "Goal adjustment failed"))
                        continue
                    goal_pos = (*adjusted_goal_pos, 0)

                    # Measure time and attempt RRT*
                    start_time = time.perf_counter()
                    try:
                        path, tree = rrt_star(
                            start_pos[:2], goal_pos[:2], obstacles, space_dim, max_iter=max_iter,
                            step_size=1.5, base_radius=0.1, retries=retries, dim=2, goal_bias=0.5, inflation=0.65
                        )
                    except Exception as e:
                        tqdm.write(f"\nTest failed: retries={retries}, max_iter={max_iter}, test_point={test_point}. Error: {str(e)}")
                        failed_cases.append((retries, test_point, max_iter, f"RRT* failed: {str(e)}"))
                        continue
                    end_time = time.perf_counter()

                    elapsed_time = (end_time - start_time) * 1000  # Convert to ms

                    # Process results
                    if path:
                        path_cost = sum(
                            np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
                            for i in range(len(path) - 1)
                        )
                        tqdm.write(f"\nRRT* Path found in {elapsed_time:.2f} ms with cost {path_cost:.2f}")
                        elapsed_times.append(elapsed_time)
                        path_costs.append(path_cost)
                        iteration_counts.append(max_iter)  # Log the max_iter for successful paths
                    else:
                        tqdm.write(f"\nTest failed: retries={retries}, max_iter={max_iter}, test_point={test_point}. No path found.")
                        failed_cases.append((retries, test_point, max_iter, "No path found"))
                        elapsed_times.append(elapsed_time)
                        path_costs.append(float('inf'))

                # Average results over test points
                avg_elapsed_time = np.mean(elapsed_times) if elapsed_times else float('inf')
                avg_path_cost = (
                    np.mean([cost for cost in path_costs if cost != float('inf')])
                    if any(cost != float('inf') for cost in path_costs) else float('inf')
                )

                # Store valid results
                if avg_path_cost != float('inf'):
                    results.append((retries, max_iter, avg_elapsed_time, avg_path_cost))
                else:
                    failed_cases.append((retries, test_point, max_iter, "All test points failed"))

        # Output failed cases in the terminal
        if failed_cases:
            print("\n=== Failed Cases ===")
            for case in failed_cases:
                print(f"Retries: {case[0]}, Test point: {case[1]}, Max Iterations: {case[2]}, Reason: {case[3]}")

        # Process and graph valid results
        if results:
            results = np.array(results)
            retries_vals = results[:, 0]
            max_iter_vals = results[:, 1]
            avg_elapsed_times = results[:, 2]
            avg_path_costs = results[:, 3]

            # Plot Probability Distribution for Iterations
            plt.figure(figsize=(10, 6))
            plt.hist(iteration_counts, bins=10, color='blue', alpha=0.7, density=True)
            plt.xlabel("Number of Iterations")
            plt.ylabel("Probability Density")
            plt.title("Probability Distribution of Iterations for Successful Paths")
            plt.grid()
            plt.show()

            # RBF Interpolation
            rbf_path_cost = Rbf(retries_vals, max_iter_vals, avg_path_costs, function='multiquadric')
            rbf_elapsed_time = Rbf(retries_vals, max_iter_vals, avg_elapsed_times, function='multiquadric')

            # Dense grid for plotting
            retries_dense, max_iter_dense = np.meshgrid(
                np.linspace(min(retries_range), max(retries_range), 50),
                np.linspace(min(max_iter_range), max(max_iter_range), 50)
            )
            path_cost_dense = rbf_path_cost(retries_dense, max_iter_dense)
            elapsed_time_dense = rbf_elapsed_time(retries_dense, max_iter_dense)

            # Plot Path Cost Surface
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            norm = plt.Normalize(vmin=elapsed_time_dense.min(), vmax=elapsed_time_dense.max())
            colors = plt.cm.plasma(norm(elapsed_time_dense))
            surf = ax.plot_surface(
                retries_dense, max_iter_dense, path_cost_dense, facecolors=colors,
                rstride=1, cstride=1, linewidth=0, alpha=0.9, edgecolor='k'
            )
            ax.set_xlabel("Retries")
            ax.set_ylabel("Max Iterations")
            ax.set_zlabel("Average Path Cost")
            ax.set_title("Path Cost Surface")
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='plasma', norm=norm), ax=ax, shrink=0.5, aspect=10)
            cbar.set_label("Elapsed Time (ms)")
            plt.show()
        else:
            print("\nNo valid configurations found for graphing.")
