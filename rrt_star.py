import numpy as np
from tqdm import tqdm
from obstacles import SphereObstacle
from scipy.spatial import KDTree

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

def rrt_star(start_pos, goal_pos, obstacles, space_dim, max_iter=350, step_size=1.0, base_radius=2.0, retries=15):
    """
    Simplified and optimized RRT* algorithm with reduced verbosity for terminal output.

    Args:
        start_pos (tuple): Starting position in 3D space.
        goal_pos (tuple): Goal position in 3D space.
        obstacles (list): List of SphereObstacle objects.
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        max_iter (int): Maximum number of iterations for each retry.
        step_size (float): Maximum step size for tree expansion.
        base_radius (float): Initial radius for rewiring.
        retries (int): Number of independent retries to find the best path.

    Returns:
        tuple: (final optimized path, list of all nodes from the best run).
    """
    best_path = None
    best_cost = float("inf")

    def calculate_path_cost(path):
        return sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    print(f"Starting RRT* with {retries} retries...")

    for retry in range(retries):
        start_node = Node(start_pos)
        tree = [start_node]
        found_path = None

        for iteration in tqdm(range(max_iter), desc=f"RRT* Retry {retry + 1}", unit="steps", leave=False):
            # Adaptive radius based on the number of nodes
            radius = min(base_radius * (np.log(len(tree)) / len(tree)) ** (1 / 3), step_size)

            # Goal bias sampling
            random_point = goal_pos if np.random.rand() < 0.1 else np.random.uniform([0, 0, 0], space_dim)

            # Find the nearest node in the tree
            nearest_node = min(tree, key=lambda node: distance(node.position, random_point))

            # Generate a new point in the direction of the random point
            direction = random_point - nearest_node.position
            direction = direction / np.linalg.norm(direction)  # Normalize direction
            new_position = nearest_node.position + direction * step_size

            # Check if the new position is in collision
            if is_position_in_collision(new_position, obstacles, inflation=1.0):
                continue

            # Add the new node to the tree
            new_node = Node(new_position, parent=nearest_node)
            new_node.cost = nearest_node.cost + distance(nearest_node.position, new_position)
            tree.append(new_node)

            # Get nearby nodes for rewiring
            tree_positions = np.array([node.position for node in tree])
            distances = np.linalg.norm(tree_positions - new_node.position, axis=1)
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
    return best_path, tree if best_path else None