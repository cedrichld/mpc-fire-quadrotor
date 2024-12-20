# firezone.py
from scipy.interpolate import splprep, splev
import numpy as np
from matplotlib.path import Path

def generate_fire_zone(center, size):
    """
    Generate a fire zone represented as a closed 2D spline approximating an ellipse.

    Args:
        center (tuple): Approximate center of the fire zone.
        size (float): Approximate size of the fire zone.

    Returns:
        tuple: (x, y) coordinates of the fire zone.
    """
    angle = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    radii = size * (0.8 + 0.4 * np.random.rand(len(angle)))  # Randomized radii for variation
    control_points = np.column_stack((radii * np.cos(angle), radii * np.sin(angle))) + center[:2]
    tck, _ = splprep(control_points.T, s=0.5, per=True)
    u = np.linspace(0, 1, 100)
    return splev(u, tck)

def is_point_in_fire_zone(spline, point):
    """
    Determine if a given point is inside the fire zone.

    Args:
        spline (tuple): The fire zone coordinates (x, y).
        point (tuple): The target point (x, y).

    Returns:
        bool: True if the point is inside the fire zone, False otherwise.
    """
    x_spline, y_spline = spline
    vertices = np.column_stack((x_spline, y_spline))
    fire_zone_path = Path(vertices, closed=True)
    return fire_zone_path.contains_point(point)

def closest_point_on_spline(spline, point):
    """
    Find the closest point on a spline to a given point.

    Args:
        spline (tuple): The spline coordinates (x, y).
        point (tuple): The target point (x, y).

    Returns:
        tuple: Closest point on the spline (x, y).
    """
    x_spline, y_spline = spline
    distances = np.sqrt((np.array(x_spline) - point[0])**2 + (np.array(y_spline) - point[1])**2)
    closest_index = np.argmin(distances)
    
    return x_spline[closest_index], y_spline[closest_index]