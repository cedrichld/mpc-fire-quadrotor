# smooth_path.py
from scipy.interpolate import UnivariateSpline
import numpy as np
from .obstacles import is_point_in_collision

def smooth_path_with_collision_avoidance(path, obstacles, num_points=200, smoothing_factor=0.2):
    """
    Fit a spline with smoothing to the path and check for collisions.
    
    Args:
        path (list): List of waypoints from the A* algorithm.
        obstacles (list): List of obstacles in the environment.
        num_points (int): Number of interpolated points for the spline.
        smoothing_factor (float): Controls the smoothness of the spline.
    
    Returns:
        tuple: Smoothed spline coordinates (x_smooth, y_smooth, [z_smooth]).
    """
    path = np.array(path)
    t = np.linspace(0, 1, len(path))  # Parameterize the original path with t in [0, 1]

    # Fit smoothing splines to x and y
    spline_x = UnivariateSpline(t, path[:, 0], k=3, s=smoothing_factor)
    spline_y = UnivariateSpline(t, path[:, 1], k=3, s=smoothing_factor)

    # Generate new parameter values for a smooth path
    t_smooth = np.linspace(0, 1, num_points)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)

    # Handle 3D paths
    if path.shape[1] == 3:
        spline_z = UnivariateSpline(t, path[:, 2], k=3, s=smoothing_factor)
        z_smooth = spline_z(t_smooth)
    else:
        z_smooth = None

    # Collision-avoidance: Check each point for collisions and adjust if necessary
    x_final, y_final = [x_smooth[0]], [y_smooth[0]]
    z_final = [z_smooth[0]] if z_smooth is not None else None

    for i in range(1, len(x_smooth)):
        point = np.array([x_smooth[i], y_smooth[i]] if z_smooth is None else [x_smooth[i], y_smooth[i], z_smooth[i]])
        if is_point_in_collision(point, obstacles):
            # Insert extra waypoints between the previous safe point and this point
            prev_point = np.array([x_final[-1], y_final[-1]] if z_final is None else [x_final[-1], y_final[-1], z_final[-1]])
            intermediate_points = np.linspace(prev_point, point, num=5, endpoint=False)[1:]
            
            # Check each intermediate point for collisions and add only safe points
            for int_point in intermediate_points:
                if not is_point_in_collision(int_point, obstacles):
                    x_final.append(int_point[0])
                    y_final.append(int_point[1])
                    if z_final is not None:
                        z_final.append(int_point[2])
        else:
            x_final.append(x_smooth[i])
            y_final.append(y_smooth[i])
            if z_final is not None:
                z_final.append(z_smooth[i])

    if z_smooth is not None:
        return np.array(x_final), np.array(y_final), np.array(z_final)
    return np.array(x_final), np.array(y_final)


def smooth_path_discretized(smooth_path, avg_vel=None, tf=None, max_avg_vel=0.25):
    """
    Generates the total length, segment lengths, and time duration for the smooth path.
    
    Args:
        smooth_path (ndarray): Smoothed path coordinates (N x 2 or N x 3).
        avg_vel (float, optional): Average velocity to traverse the path. Defaults to max_avg_vel.
        tf (float, optional): Total time to complete the trajectory. If None, inferred from avg_vel.
        max_avg_vel (float, optional): Maximum velocity constraint. Default is 2.0 m/s.
    
    Returns:
        tuple: Total length, segment lengths, and total time (tf).
    """
    smooth_path = np.array(smooth_path)  # Ensure it is a numpy array
    total_length = np.linalg.norm(np.diff(smooth_path, axis=0), axis=1)
    
    # Determine avg_vel if not provided
    avg_vel = avg_vel or max_avg_vel
    
    # Calculate tf if not provided
    tf = tf or (total_length / avg_vel)[0]
    
    return tf

def smooth_path_pos_t(smooth_path, t, tf):
    """
    Generates the total length, segment lengths, and time duration for the smooth path.
    
    Args:
        smooth_path (ndarray): Smoothed path coordinates (N x 2 or N x 3).
        avg_vel (float, optional): Average velocity to traverse the path. Defaults to max_avg_vel.
        tf (float, optional): Total time to complete the trajectory. If None, inferred from avg_vel.
        max_avg_vel (float, optional): Maximum velocity constraint. Default is 2.0 m/s.
    
    Returns:
        tuple: Position on the path at time t (x, y, [z]).
    """
    smooth_path = np.array(smooth_path)  # Ensure it is a numpy array
    # print(f"smooth path: {smooth_path} and smooth[0] {smooth_path[0]}")
    t = max(0, min(t, tf)) # Ensure t is within [0, tf]
    num_points = (smooth_path[0]).shape[0] # Total number of points in the path
    index = int(round((t / tf) * (num_points - 1)))

    return tuple(smooth_path[:, index])

def smooth_path_vel_t(smooth_path, t, tf):
    """
    Computes velocity at time t along the smooth path.
    
    Args:
        smooth_path (tuple or ndarray): Smoothed path as ([x], [y]) or (N x 2) array.
        t (float): Time at which velocity is required.
        tf (float): Total time to traverse the path.
    
    Returns:
        tuple: Velocity vector (vx, vy, [vz]) at time t.
    """
    smooth_path = np.array(smooth_path)  # Ensure it is a numpy array
    t = max(0, min(t, tf))  # Clamp t to the range [0, tf]
    num_points = (smooth_path[0]).shape[0] # Total number of points in the path

    # Compute the current and next indices
    index = int(round((t / tf) * (num_points - 1)))
    next_index = min(index + 1, num_points - 1)  # Ensure we don't exceed array bounds
    
    # Compute time difference between indices
    dt = tf / (num_points - 1)  # Time per segment
    
    # Compute velocity as the difference between consecutive points
    velocity = (smooth_path[:, next_index] - smooth_path[:, index]) / dt

    return tuple(velocity)