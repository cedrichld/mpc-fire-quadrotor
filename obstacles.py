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

# Trees
class CylinderObstacle:
    def __init__(self, base_center, height, radius):
        self.base_center = np.array(base_center)
        self.height = height
        self.radius = radius

    def is_inside(self, position):
        x, y, z = position
        x0, y0, z0 = self.base_center

        # Check horizontal distance
        horizontal_distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        if horizontal_distance > self.radius:
            return False

        # Check vertical bounds
        if z < z0 or z > z0 + self.height:
            return False

        return True