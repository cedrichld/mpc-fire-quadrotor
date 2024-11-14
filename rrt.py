# rrt.py
import numpy as np
from tqdm import tqdm
from obstacles import SphereObstacle

class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_position_in_collision(position, obstacles, inflation=1.0):
    for obstacle in obstacles:
        if obstacle.is_inside(position, inflation):
            return True
    return False

def rrt(start_pos, goal_pos, obstacles, space_dim, max_iter=1000, step_size=1.0):
    """
    Rapidly-exploring Random Tree (RRT) algorithm with inflation and progress tracking.

    Args:
        start_pos (tuple): Starting position in 3D space.
        goal_pos (tuple): Goal position in 3D space.
        obstacles (list): List of SphereObstacle objects.
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        max_iter (int): Maximum number of iterations.
        step_size (float): Maximum step size for tree expansion.

    Returns:
        tuple: (final path, list of all nodes).
    """
    start_node = Node(start_pos)
    goal_node = Node(goal_pos)
    tree = [start_node]

    with tqdm(total=max_iter, desc="RRT Progress", unit="steps") as pbar:
        for iteration in range(max_iter):
            # Sample a random point in the space
            random_point = np.random.uniform([0, 0, 0], space_dim)

            # Find the nearest node in the tree
            nearest_node = min(tree, key=lambda node: distance(node.position, random_point))

            # Generate a new point in the direction of the random point
            direction = random_point - nearest_node.position
            direction = direction / np.linalg.norm(direction)  # Normalize direction
            new_position = nearest_node.position + direction * step_size

            # Check if the new position is in collision
            if is_position_in_collision(new_position, obstacles, inflation=1.0):
                pbar.update(1)
                continue

            # Add the new node to the tree
            new_node = Node(new_position, parent=nearest_node)
            tree.append(new_node)

            # Check if we reached the goal
            if distance(new_position, goal_pos) < step_size:
                path = []
                current = new_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                pbar.close()
                return path[::-1], tree  # Return reversed path and the tree

            # Update progress bar
            pbar.update(1)

    print("RRT terminated: Goal not reached within max iterations.")
    return None, tree

# Adjust visualization to use gradient coloring for branches
def plot_rrt_attempts(space_dim, start_pos, goal_pos, tree, obstacles):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_dim[0])
    ax.set_ylim(0, space_dim[1])
    ax.set_zlim(0, space_dim[2])

    # Plot obstacles
    for obstacle in obstacles:
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = obstacle.radius * np.cos(u) * np.sin(v) + obstacle.center[0]
        y = obstacle.radius * np.sin(u) * np.sin(v) + obstacle.center[1]
        z = obstacle.radius * np.cos(v) + obstacle.center[2]
        ax.plot_surface(x, y, z, color="black", alpha=0.3)

    # Plot start and goal
    ax.scatter(*start_pos, color='blue', s=50, label='Start')
    ax.scatter(*goal_pos, color='red', s=50, label='Goal')

    # Plot the tree with gradient coloring
    num_nodes = len(tree)
    for i, node in enumerate(tree):
        if node.parent is not None:
            color_intensity = i / num_nodes
            ax.plot(
                [node.position[0], node.parent.position[0]],
                [node.position[1], node.parent.position[1]],
                [node.position[2], node.parent.position[2]],
                color=(color_intensity, 1 - color_intensity, 0), alpha=0.7
            )

    plt.legend()
    plt.show()

# Update SphereObstacle class in obstacles.py
def sphere_obstacle_is_inside(position, inflation=0.0):
    def is_inside(self, position, inflation=0.0):
        """Check if a position is inside the sphere with optional inflation."""
        return np.linalg.norm(np.array(position) - self.center) <= self.radius + inflation
