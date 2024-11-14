# smooth_path.py
from scipy.interpolate import UnivariateSpline
import numpy as np

def is_point_in_collision(point, obstacles):
    """Check if a given point is inside any obstacle."""
    for obstacle in obstacles:
        if obstacle.is_inside(point):
            return True
    return False

def smooth_path_with_collision_avoidance(path, obstacles, num_points=100, smoothing_factor=1):
    """
    Fit a spline with smoothing to the path and check for collisions.
    
    Args:
        path (list): List of waypoints from the A* algorithm.
        obstacles (list): List of obstacles in the environment.
        num_points (int): Number of interpolated points for the spline.
        smoothing_factor (float): Controls the smoothness of the spline.
    
    Returns:
        tuple: (x_smooth, y_smooth, z_smooth), smoothed spline coordinates.
    """
    path = np.array(path)
    t = np.linspace(0, 1, len(path))  # Parameterize the original path with t in [0, 1]

    # Fit smoothing splines to x, y, and z independently with a smoothing factor
    spline_x = UnivariateSpline(t, path[:, 0], k=3, s=smoothing_factor)
    spline_y = UnivariateSpline(t, path[:, 1], k=3, s=smoothing_factor)
    spline_z = UnivariateSpline(t, path[:, 2], k=3, s=smoothing_factor)

    # Generate new parameter values for a smooth path
    t_smooth = np.linspace(0, 1, num_points)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)
    z_smooth = spline_z(t_smooth)

    # Collision-avoidance: Check each point for collisions and adjust if necessary
    x_final, y_final, z_final = [x_smooth[0]], [y_smooth[0]], [z_smooth[0]]  # Start with the first point
    for i in range(1, len(x_smooth)):
        point = np.array([x_smooth[i], y_smooth[i], z_smooth[i]])
        if is_point_in_collision(point, obstacles):
            # Insert extra waypoints between the previous safe point and this point
            prev_point = np.array([x_final[-1], y_final[-1], z_final[-1]])
            intermediate_points = np.linspace(prev_point, point, num=5, endpoint=False)[1:]
            
            # Check each intermediate point for collisions and add only safe points
            for int_point in intermediate_points:
                if not is_point_in_collision(int_point, obstacles):
                    x_final.append(int_point[0])
                    y_final.append(int_point[1])
                    z_final.append(int_point[2])
        else:
            x_final.append(x_smooth[i])
            y_final.append(y_smooth[i])
            z_final.append(z_smooth[i])

    return np.array(x_final), np.array(y_final), np.array(z_final)
