# astar.py
import numpy as np
from queue import PriorityQueue
from tqdm import tqdm
from obstacles import is_point_in_collision

class Node:
    def __init__(self, position, g=0, h=0, parent=None):
        self.position = position
        self.g = g  # Cost from start to this node
        self.h = h  # Heuristic cost from this node to goal
        self.parent = parent

    @property
    def f(self):
        """Total cost function f = g + h."""
        return self.g + self.h

    def __lt__(self, other):
        """Comparison method for priority queue."""
        return self.f < other.f

def heuristic(node, goal):
    """Euclidean distance heuristic."""
    return np.linalg.norm(np.array(node.position) - np.array(goal.position))

def a_star_3d(start_pos, goal_pos, obstacles, space_dim, max_iter=5000, eps=0.1):
    """
    Perform A* search in a 3D space from start to goal, avoiding obstacles.

    Args:
        start_pos (tuple): Starting position in 3D space.
        goal_pos (tuple): Goal position in 3D space.
        obstacles (list): List of SphereObstacle objects.
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        max_iter (int): Maximum number of iterations for progress tracking.

    Returns:
        tuple: (final path, list of visited positions).
    """
    start_node = Node(position=start_pos)
    goal_node = Node(position=goal_pos)
    
    open_set = PriorityQueue()
    open_set.put((start_node.f, start_node))
    closed_set = set()
    visited_positions = []  # List to store visited positions
    g_values = {tuple(start_node.position): start_node.g}  # Best g-values for each position

    # All 26 possible movements in 3D (including diagonals)
    directions = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if (dx, dy, dz) != (0, 0, 0)
    ]

    # Progress bar for tracking search progress
    with tqdm(total=max_iter, desc="A* 3D Search Progress", unit="steps") as pbar:
        step_count = 0
        while not open_set.empty() and step_count < max_iter:
            current_node = open_set.get()[1]
            visited_positions.append(current_node.position)
            pbar.update(1)
            step_count += 1

            # Check if we are close enough to the goal
            if np.linalg.norm(np.array(current_node.position) - np.array(goal_pos)) < eps:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1], visited_positions  # Return reversed path and visited positions

            closed_set.add(tuple(current_node.position))

            # Generate neighbors in 3D space, including diagonals
            for dx, dy, dz in directions:
                neighbor_pos = (
                    current_node.position[0] + dx,
                    current_node.position[1] + dy,
                    current_node.position[2] + dz
                )
                
                # Ensure neighbor is within boundaries
                if not (0 <= neighbor_pos[0] < space_dim[0] and
                        0 <= neighbor_pos[1] < space_dim[1] and
                        0 <= neighbor_pos[2] < space_dim[2]):
                    continue

                # Check for collision with obstacles
                if tuple(neighbor_pos) in closed_set or is_point_in_collision(neighbor_pos, obstacles, inflation=0.1):
                    continue

                # Calculate new g-cost
                movement_cost = np.sqrt(dx**2 + dy**2 + dz**2)  # Distance for diagonal moves
                tentative_g = current_node.g + movement_cost
                if tuple(neighbor_pos) in g_values and tentative_g >= g_values[tuple(neighbor_pos)]:
                    continue  # Skip if not a better path

                # Create and update neighbor node
                neighbor_node = Node(position=neighbor_pos)
                neighbor_node.g = tentative_g
                neighbor_node.h = heuristic(neighbor_node, goal_node)
                neighbor_node.parent = current_node

                # Add neighbor to open set with updated cost
                g_values[tuple(neighbor_pos)] = neighbor_node.g
                open_set.put((neighbor_node.f, neighbor_node))

        print("Search terminated: No path found or max iterations reached.")
    return None, visited_positions  # Return None if no path is found, along with visited positions

def a_star_2d(start_pos, goal_pos, obstacles, space_dim, step_size=0.1, max_iter=50000, eps=0.05):
    """
    Perform A* search in a 2D space from start to goal, avoiding obstacles.

    Args:
        start_pos (tuple): Starting position in 2D space.
        goal_pos (tuple): Goal position in 2D space.
        obstacles (list): List of obstacles.
        space_dim (tuple): Dimensions of the 2D space (x, y).
        step_size (int): Step size for movement in the grid.
        max_iter (int): Maximum number of iterations for progress tracking.

    Returns:
        tuple: (final path, list of visited positions).
    """
    start_node = Node(position=start_pos)
    goal_node = Node(position=goal_pos)
    
    open_set = PriorityQueue()
    open_set.put((start_node.f, start_node))
    closed_set = set()
    visited_positions = []
    g_values = {tuple(start_node.position): start_node.g}

    # All possible movements in 2D space based on step size
    directions = [
        (dx * step_size, dy * step_size)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        if (dx, dy) != (0, 0)
    ]

    with tqdm(total=max_iter, desc="A* 2D Search Progress", unit="steps") as pbar:
        step_count = 0
        while not open_set.empty() and step_count < max_iter:
            current_node = open_set.get()[1]
            visited_positions.append(current_node.position)
            pbar.update(1)
            step_count += 1

            # Check if within epsilon of the goal
            if np.linalg.norm(np.array(current_node.position) - np.array(goal_pos)) <= eps:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1], visited_positions

            closed_set.add(tuple(current_node.position))

            # Generate neighbors
            for dx, dy in directions:
                neighbor_pos = (
                    current_node.position[0] + dx,
                    current_node.position[1] + dy
                )

                # Ensure neighbor is within bounds
                if not (0 <= neighbor_pos[0] < space_dim[0] and
                        0 <= neighbor_pos[1] < space_dim[1]):
                    continue

                # Check collision with obstacles
                if tuple(neighbor_pos) in closed_set or is_point_in_collision(neighbor_pos, obstacles, inflation=0.5):
                    continue

                # Calculate g-cost
                movement_cost = np.linalg.norm([dx, dy])  # Adjusted for step size
                tentative_g = current_node.g + movement_cost
                if tuple(neighbor_pos) in g_values and tentative_g >= g_values[tuple(neighbor_pos)]:
                    continue

                # Create neighbor node
                neighbor_node = Node(position=neighbor_pos)
                neighbor_node.g = tentative_g
                neighbor_node.h = heuristic(neighbor_node, goal_node)
                neighbor_node.parent = current_node

                g_values[tuple(neighbor_pos)] = neighbor_node.g
                open_set.put((neighbor_node.f, neighbor_node))

        print("Search terminated: No path found or max iterations reached.")
    return None, visited_positions