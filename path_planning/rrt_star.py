import numpy as np
from tqdm import tqdm
from .obstacles import is_point_in_collision

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

def rewire_tree(new_node, nearby_nodes):
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
        closest_obstacle_pos = None
        closest_obstacle_rad = None
        min_dist = float('inf')
        
        for obstacle in obstacles:
            dist = np.linalg.norm(point - obstacle.center[:dim])
            if dist < min_dist:
                min_dist = dist
                closest_obstacle_pos = obstacle.center[:dim]
                closest_obstacle_rad = obstacle.radius

        # If the point is within the inflation radius, adjust it
        if min_dist < closest_obstacle_rad + inflation and closest_obstacle_pos is not None:
            direction = (point - closest_obstacle_pos) / np.linalg.norm(point - closest_obstacle_pos)  # Normalize
            adjusted_point = closest_obstacle_pos + direction * (closest_obstacle_rad + inflation)
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


def is_sufficiently_smooth(path, threshold=0.1):
    if not path or len(path) < 2:
        return False
    for i in range(len(path) - 1):
        if distance(path[i], path[i+1]) >= threshold:
            return False
    return True

def a_star(start, goal, obstacles, space_dim, inflation=0.0):
    """
    Placeholder for an A* routine returning the path from start to goal.
    Return None if no path found, or a list of (x, y) positions otherwise.
    """
    # You can replace with your own A* implementation.
    # Here we simply return a direct line if no collision (for illustration).
    steps = 50
    path = []
    start_arr = np.array(start)
    goal_arr = np.array(goal)
    diff = goal_arr - start_arr
    for i in range(steps + 1):
        pt = start_arr + (i / steps) * diff
        if is_point_in_collision(pt, obstacles, inflation):
            return None
        path.append(tuple(pt))
    return path

def find_most_efficient_path(candidate_points, start_pos, goal_pos, obstacles, space_dim, inflation=0.0):
    """
    Uses A* (or a variant) on each candidate path section to produce 
    the most efficient overall path if it exists.
    """
    # For simplicity, pick the best path among candidates
    best_path = None
    best_cost = float('inf')
    for cpath in candidate_points:
        # Combine with A* from start to first candidate, then candidate to goal
        segment1 = a_star(start_pos, cpath, obstacles, space_dim, inflation)
        if segment1 is None:
            continue
        segment2 = a_star(cpath, goal_pos, obstacles, space_dim, inflation)
        if segment2 is None:
            continue
        
        full_path = segment1[:-1] + segment2  # join (avoid duplicating cpath)
        cost = sum(distance(full_path[i], full_path[i+1]) for i in range(len(full_path)-1))
        if cost < best_cost:
            best_cost = cost
            best_path = full_path
    return best_path

def generate_line_samples(start_pos, goal_pos, obstacles, inflation=0.0, n_samples=100):
    """
    Returns up to n_samples points along the line from start to goal
    that are not in collision with the given obstacles.
    """
    start_arr = np.array(start_pos)
    goal_arr = np.array(goal_pos)
    
    valid_samples = []
    # Either use a uniform parametric range [0..1], or random for each sample
    # Here we'll use uniform increments
    for t in np.linspace(0, 1, n_samples):
        candidate_pt = start_arr + t * (goal_arr - start_arr)
        if not is_point_in_collision(candidate_pt, obstacles, inflation):
            valid_samples.append(tuple(candidate_pt))
    
    return valid_samples

# Line-samples based RRT*
def ls_rrt_star(
    start_pos, goal_pos, obstacles, space_dim, 
    samples_per_d=100, smoothness_thresh=0.1, inflation=0.65
):
    """
    RRT*-style controller that samples around a reference line,
    attempts to find a path at each iteration, stores the best path,
    and stops early once smoothness is sufficient or if max_iter is reached.
    """
    if distances_from_line is None:
        distances_from_line = [0.1 * i for i in range(1, 11)]  # 0.1 to 1.0
        samples_per_ = np.ones(len(distances_from_line)) * samples_per_d
    
    # fallback: sample directly from start->goal line
    def line_sample_fn():
        # parametric sample from start to goal
        t = np.random.rand()
        return np.array(start_pos) + t*(np.array(goal_pos) - np.array(start_pos))
    
    def sample_at_d(d):
        line_center_pt = line_sample_fn()  # a random point on the reference line
        # Sample a random direction orthogonal in 2D (can be extended for higher dim)
        random_angle = 2 * np.pi * np.random.rand()
        offset = np.array([np.cos(random_angle), np.sin(random_angle)]) * d
        candidate_pt = line_center_pt + offset
        # Clip to space boundary
        candidate_pt = np.clip(candidate_pt, [0, 0], space_dim)
        # Check collision
        if is_point_in_collision(candidate_pt, obstacles, inflation):
            sample_at_d()
        else:
            return candidate_pt
    
    line_samples = generate_line_samples(start_pos, goal_pos, obstacles, inflation, n_samples=100)
    max_iter = len(distances_from_line) * samples_per_d + len(line_samples)
    
    best_paths_per_iteration = []
    best_path_so_far = None

    iteration = tqdm.tqdm(total=max_iter, desc=f"RRT* Pathfinding", unit="steps", leave=False)
    
    # Generate candidate points around the line
    candidate_points = [line_samples]
    for d in distances_from_line:
        for _ in samples_per_[d]:
            candidate_pt = sample_at_d(d)
            candidate_points.append(candidate_pt)
            iteration.update(1)
            
    
            # Among the candidate points, find the most efficient A* path
            new_best_path = find_most_efficient_path(
                candidate_points, start_pos, goal_pos, obstacles, space_dim, inflation
            )
    
            if new_best_path is not None:
                # Compare with global best so far
                if best_path_so_far is None:
                    best_path_so_far = new_best_path
                else:
                    new_cost = sum(distance(new_best_path[i], new_best_path[i+1]) 
                                    for i in range(len(new_best_path)-1))
                    old_cost = sum(distance(best_path_so_far[i], best_path_so_far[i+1]) 
                                    for i in range(len(best_path_so_far)-1))
                    if new_cost < old_cost:
                        best_path_so_far = new_best_path
    
            # Store best path for the iteration (or None if none is found yet)
            best_paths_per_iteration.append(best_path_so_far)

            # Early stop if we have a path and it's sufficiently smooth
            if best_path_so_far and is_sufficiently_smooth(best_path_so_far, smoothness_thresh):
                print(f"Stopping early at iteration {iteration+1}, path is smooth enough.")
                break
            
    # Final A* run from start to goal (could be any final improvement or post-check)
    final_path = a_star(start_pos, goal_pos, obstacles, space_dim, inflation)
    if final_path is not None:
        # Compare final path cost to best path so far
        if best_path_so_far is not None:
            new_cost = sum(distance(final_path[i], final_path[i+1]) 
                        for i in range(len(final_path)-1))
            old_cost = sum(distance(best_path_so_far[i], best_path_so_far[i+1]) 
                        for i in range(len(best_path_so_far)-1))
            if new_cost < old_cost:
                best_path_so_far = final_path
        else:
            best_path_so_far = final_path

    return best_paths_per_iteration, best_path_so_far












def rrt_star(
    start_pos, goal_pos, obstacles, space_dim, t_ref=None, max_iter=300, step_size=1.5,
    base_radius=0.1, retries=10, dim=2, goal_bias=0.5, inflation=0.65
):
    best_path = None
    best_cost = float("inf")
    tree = []  # Ensure tree is initialized
    
    # start_pos, goal_pos = start_pos[:dim], goal_pos[:dim]
    # start_node = Node(start_pos)
    # goal_node = Node(goal_pos)
    # tree = [start_node, goal_node]
    # best_path = [start_node.position, goal_node.position]
    # best_cost = 0

    
    def calculate_path_cost(path):
        return sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    print(f"Starting RRT* with {retries} retries in {dim}D space...")
    start_pos, goal_pos = start_pos[:dim], goal_pos[:dim]

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

            rewire_tree(new_node, nearby_nodes)

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
                    best_iteration = iteration + 1
                    # print(f"\rRRT* Retry {retry + 1}: Path found with cost {path_cost:.2f} on iteration {iteration + 1}", end="")
                found_path = True
                break
    
    # Refine the best path if found
    if best_path:
        best_path = refine_path_line_segments(best_path, obstacles, inflation, dim)#, **({'n_points': t_ref} if t_ref is not None else {})) # if t_ref is defined
    else:
        sub_trees = []

    print(f"Final best path cost: {best_cost:.2f}" if best_path else "\nNo path found.") #  on iteration {iteration + 1}
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