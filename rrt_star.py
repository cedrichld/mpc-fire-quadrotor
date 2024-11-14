import numpy as np
from tqdm import tqdm
from obstacles import is_point_in_collision
# from scipy.spatial import KDTree

class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.cost = 0  # Cost to reach this node

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_position_in_collision(position, obstacles, inflation=1.0):
    for obstacle in obstacles:
        if obstacle.is_inside(position, inflation):
            return True
    return False

def get_nearby_nodes(tree, new_node, radius):
    """
    Get all nodes in the tree within a certain radius of the new node.
    """
    nearby_nodes = []
    for node in tree:
        if distance(node.position, new_node.position) <= radius:
            nearby_nodes.append(node)
    return nearby_nodes

def rewire_tree(tree, new_node, nearby_nodes):
    """
    Rewire nearby nodes if connecting them through the new node reduces their cost.
    """
    for nearby_node in nearby_nodes:
        new_cost = new_node.cost + distance(new_node.position, nearby_node.position)
        if new_cost < nearby_node.cost:
            nearby_node.parent = new_node
            nearby_node.cost = new_cost

def rrt_star(
    start_pos, goal_pos, obstacles, space_dim, max_iter=1500, step_size=2.0,
    base_radius=0.1, retries=10, dim=3, goal_bias=0.2
):
    """
    Simplified and optimized RRT* algorithm with goal bias for both 2D and 3D spaces.

    Args:
        start_pos (tuple): Starting position in space (2D or 3D).
        goal_pos (tuple): Goal position in space (2D or 3D).
        obstacles (list): List of obstacles (SphereObstacle or CylinderObstacle).
        space_dim (tuple): Dimensions of the space (x, y[, z]).
        max_iter (int): Maximum number of iterations for each retry.
        step_size (float): Maximum step size for tree expansion.
        base_radius (float): Initial radius for rewiring.
        retries (int): Number of independent retries to find the best path.
        dim (int): Dimensionality of the space (2 for 2D, 3 for 3D).
        goal_bias (float): Probability of sampling near the goal.

    Returns:
        tuple: (final optimized path, list of all nodes from the best run).
    """
    best_path = None
    best_cost = float("inf")

    def calculate_path_cost(path):
        return sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    print(f"Starting RRT* with {retries} retries in {dim}D space...")

    for retry in range(retries):
        start_node = Node(start_pos)
        tree = [start_node]
        found_path = None

        for iteration in tqdm(range(max_iter), desc=f"RRT* Retry {retry + 1}", unit="steps", leave=False):
            # Adaptive radius based on the number of nodes
            radius = min(base_radius * (np.log(len(tree)) / len(tree)) ** (1 / dim), step_size)

            # Goal bias sampling: prioritize samples closer to the goal
            if np.random.rand() < goal_bias:
                random_point = tuple(np.array(goal_pos) + np.random.uniform(-step_size, step_size, size=dim))
                random_point = np.clip(random_point, [0] * dim, space_dim[:dim])  # Ensure within bounds
            else:
                random_point = tuple(np.random.uniform(low=[0] * dim, high=space_dim[:dim]))

            # Find the nearest node in the tree
            nearest_node = min(tree, key=lambda node: distance(node.position, random_point))

            # Generate a new point in the direction of the random point
            direction = np.array(random_point) - np.array(nearest_node.position)
            direction = direction / np.linalg.norm(direction)  # Normalize direction
            new_position = tuple(np.array(nearest_node.position) + direction * step_size)

            # Check if the new position is in collision
            if is_point_in_collision(new_position, obstacles, inflation=1.0):
                continue

            # Add the new node to the tree
            new_node = Node(new_position, parent=nearest_node)
            new_node.cost = nearest_node.cost + distance(nearest_node.position, new_position)
            tree.append(new_node)

            # Get nearby nodes for rewiring
            tree_positions = np.array([node.position for node in tree])
            distances = np.linalg.norm(tree_positions - np.array(new_node.position), axis=1)
            nearby_indices = np.where(distances <= radius)[0]
            nearby_nodes = [tree[i] for i in nearby_indices]

            # Rewire nearby nodes
            for nearby_node in nearby_nodes:
                new_cost = new_node.cost + distance(new_node.position, nearby_node.position)
                if new_cost < nearby_node.cost:
                    nearby_node.parent = new_node
                    nearby_node.cost = new_cost

            # Check if we reached the goal
            if distance(new_position, goal_pos) < step_size:
                path = []
                current = new_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                path_cost = calculate_path_cost(path[::-1])
                if path_cost < best_cost:
                    best_path = path[::-1]
                    best_cost = path_cost
                    print(f"\rRRT* Retry {retry + 1}: Path found with cost {path_cost:.2f} on iteration {iteration + 1}", end="")
                found_path = True
                break

        if not found_path:
            print(f"\rRRT* Retry {retry + 1}: No path found within max iterations.", end="")

    print(f"Final best path cost: {best_cost:.2f}" if best_path else "\nNo path found.")
    return best_path, tree  # if best_path else None



def plot_rrt_attempts(ax, tree, dim=3):
    """
    Plot the sampled points and tree for RRT attempts with gradient-colored branches.

    Args:
        space_dim (tuple): Dimensions of the space (x, y[, z]).
        start_pos (tuple): Starting position.
        goal_pos (tuple): Goal position.
        tree (list): List of Node objects representing the RRT tree.
        obstacles (list): List of obstacles (SphereObstacle or CylinderObstacle).
        dim (int): Dimensionality of the space (2 for 2D, 3 for 3D).
    """
    # import matplotlib.pyplot as plt

    # # Plot start and goal positions
    # if dim == 3:
    #     from mpl_toolkits.mplot3d import Axes3D  # Import only when needed

    # Plot the RRT tree with gradient coloring
    num_nodes = len(tree)
    for i, node in enumerate(tree):
        if node.parent is not None:
            color_intensity = i / num_nodes  # Scale intensity by node index
            if dim == 3:
                ax.plot(
                    [node.position[0], node.parent.position[0]],
                    [node.position[1], node.parent.position[1]],
                    [node.position[2], node.parent.position[2]],
                    color=(1 - color_intensity, 1 - color_intensity, color_intensity),  # Gradient color
                    alpha=0.8,
                )
            elif dim == 2:
                ax.plot(
                    [node.position[0], node.parent.position[0]],
                    [node.position[1], node.parent.position[1]],
                    color=(1 - color_intensity, 1 - color_intensity, color_intensity),  # Gradient color
                    alpha=0.8,
                )
            else:
                import sys
                raise ValueError("Dimension neither 2 or 3.")