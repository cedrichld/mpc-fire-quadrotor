# smooth_path.py
from scipy.interpolate import UnivariateSpline
import numpy as np
from obstacles import is_point_in_collision

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

def smooth_path_at_t(smooth_path, t, avg_vel=None, tf=None, max_avg_vel=2.0):
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
    num_points = smooth_path.shape[1] # Total number of points in the path
    
    # Calculate segment lengths and cumulative path length
    segment_lengths = np.linalg.norm(np.diff(smooth_path, axis=0), axis=1)
    total_length = np.sum(segment_lengths)
    
    # Determine avg_vel if not provided
    if avg_vel is None:
        avg_vel = max_avg_vel
    
    # Calculate tf if not provided
    if tf is None:
        tf = total_length / avg_vel
    
    t = max(0, min(t, tf)) # Ensure t is within [0, tf]
    index = int(round((t / tf) * (num_points - 1)))

    return tuple(smooth_path[:, index])




'''
def smooth_path_discretized(smooth_path, avg_vel=None, tf=None, max_avg_vel=2.0):
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
    
    # Calculate segment lengths and cumulative path length
    segment_lengths = np.linalg.norm(np.diff(smooth_path, axis=0), axis=1)
    total_length = np.sum(segment_lengths)
    
    # Determine avg_vel if not provided
    if avg_vel is None:
        avg_vel = max_avg_vel
    
    # Calculate tf if not provided
    if tf is None:
        tf = total_length / avg_vel
    
    return total_length, segment_lengths, tf

def smooth_path_at_t(total_length, smooth_path, segment_lengths, t, tf):
    """
    Computes the position on the smooth path at a given time t.
    
    Args:
        total_length (float): Total length of the path.
        smooth_path (ndarray): Smoothed path coordinates (N x 2 or N x 3).
        segment_lengths (ndarray): Length of each segment in the path.
        t (float): Current time in seconds.
        tf (float): Total time for the trajectory.
    
    Returns:
        tuple: Position on the path at time t (x, y, [z]).
    """
    smooth_path = np.array(smooth_path)  # Ensure it is a numpy array
    
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)  # Include start point
    
    # Map time t to the path's progress
    progress = min(max(t / tf, 0), 1)  # Normalize t within [0, tf]
    target_length = progress * total_length  # Map progress to target length
    
    # Find the segment corresponding to target_length
    segment_index = np.searchsorted(cumulative_lengths, target_length, side='right') - 1
    
    # Calculate the exact position on the segment
    if segment_index < smooth_path.shape[0] - 1:
        seg_start = smooth_path[segment_index]
        seg_end = smooth_path[segment_index + 1]
        seg_length = segment_lengths[segment_index]
        seg_progress = (target_length - cumulative_lengths[segment_index]) / seg_length
        position = seg_start + seg_progress * (seg_end - seg_start)
    else:
        # If t >= tf, return the final point
        position = smooth_path[-1]
    
    return tuple(position)
'''