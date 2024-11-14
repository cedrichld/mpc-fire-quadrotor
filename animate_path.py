# animate_path.py
import numpy as np
import matplotlib.pyplot as plt
from create_animation import create_animation
from obstacles import SphereObstacle
from astar import a_star_3d
from smooth_path import smooth_path_with_collision_avoidance
from visualize import generate_random_spheres_along_line
from matplotlib.animation import PillowWriter

if __name__ == "__main__":
    # Define 3D space dimensions and parameters
    space_dim = (20, 20, 20)
    num_spheres = 10
    radius_range = (1.5, 2.5)

    # Define start and goal positions
    start_pos = np.array([0, 0, 0])
    goal_pos = np.array([15, 15, 15])

    # Generate obstacles close to the line between start and goal
    obstacles = generate_random_spheres_along_line(num_spheres, radius_range, start_pos, goal_pos, spread=1.5, min_distance=1)

    # Find path using A*
    path, visited_positions = a_star_3d(start_pos, goal_pos, obstacles, space_dim)

    # Check if a path was found
    if path is None or len(visited_positions) == 0:
        print("No path found or no positions visited.")
    else:
        # Smooth the path with collision avoidance
        smoothed_path = smooth_path_with_collision_avoidance(path, obstacles)

        # Animate the pathfinding process
        animation, fig = create_animation(space_dim, start_pos, goal_pos, obstacles, visited_positions, smoothed_path)
        plt.close()  # Close the static plot to focus on the animation

        # Save or display the animation
        try:
            animation.save("pathfinding_animation.mp4", fps=30, writer='ffmpeg')
            print("Animation saved as 'pathfinding_animation.mp4'.")
        except RuntimeError:
            print("ffmpeg not available, saving as GIF using Pillow.")
            animation.save("pathfinding_animation.gif", writer=PillowWriter(fps=30))
            print("Animation saved as 'pathfinding_animation.gif'.")
