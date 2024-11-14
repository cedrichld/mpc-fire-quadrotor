# obstacles.py
import random
import numpy as np

class SphereObstacle:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def is_inside(self, position, inflation=0.0):
        """Check if a position is inside the sphere with optional inflation."""
        return np.linalg.norm(np.array(position) - self.center) <= (self.radius + inflation)

def generate_random_spheres(num_spheres, radius_range, space_dim, concentration_factor=0.2):
    """
    Generate random spherical obstacles within a 3D space, concentrated around the center.
    
    Args:
        num_spheres (int): Number of spheres.
        radius_range (tuple): Min and max radius for the spheres.
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        concentration_factor (float): Determines spread; smaller values concentrate obstacles more at the center.
    
    Returns:
        list: List of SphereObstacle objects.
    """
    center_point = np.array([dim / 2 for dim in space_dim])
    spheres = []
    
    for _ in range(num_spheres):
        radius = random.uniform(radius_range[0], radius_range[1])
        
        # Generate a biased position towards the center using a normal distribution
        center = np.random.normal(
            loc=center_point,  # Mean at the center of the space
            scale=concentration_factor * np.array(space_dim)  # Spread based on concentration_factor
        )
        
        # Ensure the generated center is within bounds
        center = np.clip(center, [0, 0, 0], space_dim)
        
        spheres.append(SphereObstacle(center=center, radius=radius))
    
    return spheres


### TREES ###

# Cylinder Class
class CylinderObstacle:
    def __init__(self, center, height, radius):
        self.center = np.array(center)
        self.height = height
        self.radius = radius

    def is_inside(self, point, inflation=0.0):
        """
        Check if a position is inside the cylindrical obstacle.

        Args:
            point (tuple or list): The position to check (x, y), or (x, y, z).
            inflation (float): Safety margin added to the cylinder's radius and height.

        Returns:
            bool: True if the point is inside the cylinder, False otherwise.
        """
        # Handle 2D and 3D inputs
        if len(point) == 2:  # If 2D point
            x, y = point
            z = 0  # Assume z = 0 for 2D positions
        else:  # If 3D point
            x, y, z = point

        # Cylinder center coordinates
        x0, y0, z0 = self.center

        # Check horizontal distance
        horizontal_distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        if horizontal_distance > self.radius + inflation:
            return False

        # Check vertical bounds
        if z < z0 or z > z0 + self.height + inflation:
            return False

        return True
    
def is_point_in_collision(point, obstacles, inflation=0.0):
    """
    Check if a given point is in collision with any of the obstacles. 
    Works both in 2D and 3D.

    Args:
        point (tuple): The point to check (x, y), or (x, y, z).
        obstacles (list): List of Obstacle objects.
        inflation (float): Amount to inflate the obstacle radii for safety.

    Returns:
        bool: True if the position is in collision, False otherwise.
    """
    for obstacle in obstacles:
        if obstacle.is_inside(point, inflation):
            return True
    return False