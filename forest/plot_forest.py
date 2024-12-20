import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Forest_Plotting(object):
    
    def init_plot(self, space_dim):
        fig = plt.figure(figsize=(20, 9), dpi=150)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.2)
        
        ax_tpv = fig.add_subplot(121, projection='3d')
        ax_non_tpv = fig.add_subplot(122)
        alpha_tpv = 0.75
        
        ax_tpv.set_xlim(0, space_dim[0])
        ax_tpv.set_ylim(0, space_dim[1])
        ax_tpv.set_zlim(0, space_dim[2])
        
        ax_tpv.grid(False) 
        ax_non_tpv.grid(False) 
        ax_tpv.set_axis_off()
        
        return fig, ax_tpv, ax_non_tpv, alpha_tpv

    def visualize_forest(self, space_dim, obstacles, fire_zone, start_pos, goal_pos, fire_zone_trees=None):
        """
        Visualize the forest environment with cylindrical obstacles, fire zone, and start/goal points.

        Args:
            space_dim (tuple): Dimensions of the 3D space (x, y, z).
            obstacles (list): List of CylinderObstacle objects outside the fire zone.
            fire_zone (tuple): Coordinates of the fire zone (x, y).
            start_pos (tuple): Start position of the drone.
            goal_pos (tuple): Goal position of the drone.
            fire_zone_trees (list): Trees inside the fire zone, displayed in orange. (Optional)
        """
        fig, ax_tpv, ax_non_tpv, alpha = self.init_plot(space_dim)
        
        self.visualize_forest_2d(obstacles, fire_zone, start_pos, goal_pos, fire_zone_trees=fire_zone_trees, ax=ax_non_tpv)
        
        for idx, ax in enumerate([ax_tpv]):  
            
            # Plot trees outside fire zone
            for obstacle in obstacles:
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, 1, 10)
                x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.center[0]
                y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.center[1]
                z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.center[2]
                ax.plot_surface(x, y, z, color='green', alpha=alpha)

            # Plot trees inside fire zone (if provided)
            if fire_zone_trees:
                for obstacle in fire_zone_trees:
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, 1, 10)
                    x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.center[0]
                    y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.center[1]
                    z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.center[2]
                    ax.plot_surface(x, y, z, color='orange', alpha=alpha)

            # Plot fire zone as filled red area
            fire_x, fire_y = fire_zone
            fire_z = np.zeros_like(fire_x)
            vertices = [list(zip(fire_x, fire_y, fire_z))]
            fire_zone_polygon = Poly3DCollection(vertices, color='red', alpha=0.4, edgecolor='red')
            ax.add_collection3d(fire_zone_polygon)

            # Plot start and goal positions
            ax.scatter(*start_pos, color='blue', s=100, label='Start')
            ax.scatter(*goal_pos, color='yellow', s=100, label='Goal')

        return fig, ax_tpv, ax_non_tpv

    def visualize_forest_2d(self, obstacles, fire_zone, start_pos, goal_pos, fire_zone_trees=None, ax=None):
        """
        Visualize the forest environment in 2D with cylindrical obstacles, fire zone, and start/goal points.
        """
        
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)

        # Create patches for obstacles outside fire zone
        obstacle_patches = [
            plt.Circle(obstacle.center[:2], obstacle.radius) for obstacle in obstacles
        ]
        obstacle_collection = PatchCollection(obstacle_patches, color='green', alpha=0.75)
        ax.add_collection(obstacle_collection)
        
        # Create patches for obstacles inside fire zone
        if fire_zone_trees:
            fire_patches = [
                plt.Circle(obstacle.center[:2], obstacle.radius) for obstacle in fire_zone_trees
            ]
            fire_collection = PatchCollection(fire_patches, color='orange', alpha=0.75)
            ax.add_collection(fire_collection)

        # Plot fire zone boundary
        fire_x, fire_y = fire_zone
        ax.plot(fire_x, fire_y, color='red', linewidth=4, label='Fire Zone')
        vertices = list(zip(fire_x, fire_y))
        fire_zone_polygon = Polygon(vertices, closed=True, facecolor='red', alpha=0.3, edgecolor='darkred')
        ax.add_patch(fire_zone_polygon)

        # Plot start and goal positions
        ax.scatter(start_pos[0], start_pos[1], color='blue', s=100, label='Start')
        ax.scatter(goal_pos[0], goal_pos[1], color='yellow', s=100, label='Goal')