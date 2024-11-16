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
    p1, p2 = np.array(p1), np.array(p2)
    if p1.shape != p2.shape:
        raise ValueError(f"Dimension mismatch: {p1.shape} vs {p2.shape}")
    return np.linalg.norm(p1 - p2)

def is_position_in_collision(position, obstacles, inflation):
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

def refine_path_line_segments(path, obstacles, inflation, dim, n_points=6):
    """
    Refine the path by dividing each line segment into n points. If a point is too close to an obstacle,
    move it along the line from the point to the center of the obstacle to the inflation distance.

    Args:
        path (list): Initial path from RRT* as a list of waypoints [(x1, y1, ...), (x2, y2, ...)].
        obstacles (list): List of obstacles.
        inflation (float): Inflation radius for obstacles.
        n_points (int): Number of points to divide each line segment into.

    Returns:
        list: Refined path with adjusted points.
    """
    refined_path = [path[0]]  # Start with the first point

    def adjust_point(point, obstacles, inflation):
        """Move the point along the line away from the closest obstacle at the inflation boundary."""
        closest_obstacle = None
        min_dist = float('inf')
        
        for obstacle in obstacles:
            obstacle_center = obstacle.center[:dim]
            dist = np.linalg.norm(point - obstacle_center)
            if dist < min_dist:
                min_dist = dist
                closest_obstacle = obstacle_center

        # If the point is within the inflation radius, adjust it
        if min_dist < inflation and closest_obstacle is not None:
            direction = (point - closest_obstacle) / np.linalg.norm(point - closest_obstacle)  # Normalize
            adjusted_point = closest_obstacle + direction * inflation
            return adjusted_point
        return point

    for i in range(len(path) - 1):
        start, end = np.array(path[i]), np.array(path[i + 1])
        direction = (end - start) / np.linalg.norm(end - start)  # Direction vector

        # Generate n_points along the line segment
        for j in range(1, n_points + 1):
            point = start + direction * (j / n_points) * np.linalg.norm(end - start)

            # Adjust the point if it is too close to any obstacle
            adjusted_point = adjust_point(point, obstacles, inflation)
            refined_path.append(adjusted_point)

    refined_path.append(path[-1])  # Add the final endpoint
    return refined_path

def rrt_star(
    start_pos, goal_pos, obstacles, space_dim, max_iter=1000, step_size=1.5,
    base_radius=0.1, retries=10, dim=2, goal_bias=0.5, inflation=0.65
):
    best_path = None
    best_cost = float("inf")
    tree = []  # Ensure tree is initialized

    def calculate_path_cost(path):
        return sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    print(f"Starting RRT* with {retries} retries in {dim}D space...")

    for retry in range(retries):
        start_node = Node(start_pos)
        tree = [start_node]
        found_path = None

        for iteration in tqdm(range(max_iter), desc=f"RRT* Retry {retry + 1}", unit="steps", leave=False):
            # Adaptive radius
            radius = min(base_radius * (np.log(len(tree)) / len(tree)) ** (1 / dim), step_size)

            # Goal bias sampling
            if np.random.rand() < goal_bias:
                random_point = tuple(np.array(goal_pos[:dim]) + np.random.uniform(-step_size, step_size, size=dim))
                random_point = np.clip(random_point, [0] * dim, space_dim[:dim])
            else:
                random_point = tuple(np.random.uniform(low=[0] * dim, high=space_dim[:dim]))

            # Find nearest node
            nearest_node = min(tree, key=lambda node: distance(node.position, random_point))

            # Generate a new point
            direction = np.array(random_point) - np.array(nearest_node.position)
            direction = direction / np.linalg.norm(direction)
            new_position = tuple(np.array(nearest_node.position) + direction * step_size)

            # Check for collision
            if is_point_in_collision(new_position, obstacles, inflation):
                continue

            # Add the new node
            new_node = Node(new_position, parent=nearest_node)
            new_node.cost = nearest_node.cost + distance(nearest_node.position, new_position)
            tree.append(new_node)

            # Rewire
            tree_positions = np.array([node.position for node in tree])
            distances = np.linalg.norm(tree_positions - np.array(new_node.position), axis=1)
            nearby_indices = np.where(distances <= radius)[0]
            nearby_nodes = [tree[i] for i in nearby_indices]

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

    # Refine the best path if found
    if best_path:
        best_path = refine_path_line_segments(best_path, obstacles, inflation, dim)
    else:
        sub_trees = []

    print(f"Final best path cost: {best_cost:.2f}" if best_path else "\nNo path found.")
    return best_path, tree


def plot_rrt_attempts(ax, tree, dim=2):
    """
    Plot the sampled points and tree for RRT* with gradient-colored branches.

    Args:
        ax (matplotlib.axes): Matplotlib axis for plotting.
        tree (list): List of Node objects representing the RRT tree.
        dim (int): Dimensionality of the space (2 or 3).
    """
    # Plot the main tree in blue
    for node in tree:
        if node.parent is not None:
            if dim == 3:
                ax.plot(
                    [node.position[0], node.parent.position[0]],
                    [node.position[1], node.parent.position[1]],
                    [node.position[2], node.parent.position[2]],
                    color="blue", alpha=0.8
                )
            elif dim == 2:
                ax.plot(
                    [node.position[0], node.parent.position[0]],
                    [node.position[1], node.parent.position[1]],
                    color="blue", alpha=0.8
                )

    # Plot the sub-trees (refinement branches) in red
    # if sub_trees:
    #     for start, end in sub_trees:
    #         if dim == 3:
    #             ax.plot(
    #                 [start[0], end[0]],
    #                 [start[1], end[1]],
    #                 [start[2], end[2]],
    #                 color="red", alpha=0.5, linestyle="--"
    #             )
    #         elif dim == 2:
    #             ax.plot(
    #                 [start[0], end[0]],
    #                 [start[1], end[1]],
    #                 color="red", alpha=0.5, linestyle="--"
    #             )