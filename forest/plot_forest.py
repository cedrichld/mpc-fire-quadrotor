import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle

class Forest_Plotting(object):
    
    def init_plot(self, space_dim):
        fig = plt.figure(figsize=(20, 9), dpi=150)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.2)
        
        ax_tpv = fig.add_subplot(121, projection='3d')
        ax_non_tpv = fig.add_subplot(122)
        
        ax_tpv.set_xlim(0, space_dim[0])
        ax_tpv.set_ylim(0, space_dim[1])
        ax_tpv.set_zlim(0, space_dim[2])
        
        ax_tpv.grid(False) 
        ax_non_tpv.grid(False) 
        ax_tpv.set_axis_off()
        
        return fig, ax_tpv, ax_non_tpv
    
    def visualize_forest(self, fig, ax_tpv, ax_non_tpv, space_dim, trees_outside, fire_zone_trees, fire_zone, start_pos, goal_pos, xlim=None, ylim=None):
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
        # Filter trees based on axis limits if provided
        if xlim and ylim:
            p_thresh = 1.0 
            trees_outside = [
                tree for tree in trees_outside
                if xlim[0] - p_thresh * tree.radius <= tree.center[0] <= xlim[1] + p_thresh * tree.radius 
                and ylim[0] - p_thresh * tree.radius <= tree.center[1] <= ylim[1] + p_thresh * tree.radius 
            ]
            fire_zone_trees = [
                tree for tree in trees_outside
                if xlim[0] - p_thresh * tree.radius <= tree.center[0] <= xlim[1] + p_thresh * tree.radius 
                and ylim[0] - p_thresh * tree.radius <= tree.center[1] <= ylim[1] + p_thresh * tree.radius
            ]
            
        # print(f"x_min: {xlim[0] - 0.5} x_max: {xlim[1] + 0.5}, y_min: {ylim[0] - 0.5} y_max: {ylim[1] + 0.5}")
                
        alpha_tpv = 0.6
        
        for idx, ax in enumerate([ax_tpv]):  
            ## Add Ground
            # Define square vertices
            square_size = space_dim[0]
            ground_vertices = [
                [0, 0, 0],  # Bottom-left corner
                [square_size, 0, 0],  # Bottom-right corner
                [square_size, square_size, 0],  # Top-right corner
                [0, square_size, 0]  # Top-left corner
            ]

            # Create a 3D polygon for the ground
            ground_square = Poly3DCollection([ground_vertices], color='saddlebrown', alpha=0.1)
            ax.add_collection3d(ground_square)
            
            
            ## Plot TREES OUTSIDE fire zone
            for obstacle in trees_outside:
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, 1, 10)
                x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.center[0]
                y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.center[1]
                z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.center[2]
                ax.plot_surface(x, y, z, color='forestgreen', alpha=alpha_tpv)


            ## Plot TREES INSIDE fire zone (if provided)
            if fire_zone_trees:
                for obstacle in fire_zone_trees:
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, 1, 10)
                    x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.center[0]
                    y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.center[1]
                    z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.center[2]
                    ax.plot_surface(x, y, z, color='orange', alpha=alpha_tpv)

            ## Plot FIRE ZONE as filled red area
            fire_x, fire_y = fire_zone
            fire_z = np.zeros_like(fire_x)
            vertices = [list(zip(fire_x, fire_y, fire_z))]
            fire_zone_polygon = Poly3DCollection(vertices, color='red', alpha=0.15, edgecolor='darkred')
            # ax.add_collection3d(fire_zone_polygon)

            # Plot start and goal positions
            ax.scatter(*start_pos, color='blue', s=100, label='Start')
            ax.scatter(*goal_pos, color='yellow', s=100, label='Goal')

        return fig, ax_tpv, ax_non_tpv

    def visualize_forest_2d(self, trees_outside, fire_zone, start_pos, goal_pos, fire_zone_trees, ax=None):
        """
        Visualize the forest environment in 2D with cylindrical obstacles, fire zone, and start/goal points.
        """
        
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        
        # Brown ground (ground square)
        ground_square = Rectangle((0, 0), 80, 80, color="saddlebrown", alpha=.1)
        ax.add_patch(ground_square)

        # Create patches for obstacles outside fire zone
        obstacle_patches = [
            plt.Circle(obstacle.center[:2], obstacle.radius) for obstacle in trees_outside
        ]
        obstacle_collection = PatchCollection(obstacle_patches, color='forestgreen')#, alpha=0.85)
        ax.add_collection(obstacle_collection)
        
        # Plot fire zone boundary
        fire_x, fire_y = fire_zone
        ax.plot(fire_x, fire_y, color='red', linewidth=4, label='Fire Zone')
        vertices = list(zip(fire_x, fire_y))
        fire_zone_polygon = Polygon(vertices, closed=True, facecolor='red', alpha=0.15, edgecolor='darkred')
        ax.add_patch(fire_zone_polygon)
        
        # Create patches for obstacles inside fire zone
        if fire_zone_trees:
            fire_patches = [
                plt.Circle(obstacle.center[:2], obstacle.radius) for obstacle in fire_zone_trees
            ]
            fire_collection = PatchCollection(fire_patches, color='orangered')#, alpha=0.85)
            ax.add_collection(fire_collection)

        # Plot start and goal positions
        ax.scatter(start_pos[0], start_pos[1], color='blue', s=100, label='Start')
        ax.scatter(goal_pos[0], goal_pos[1], color='yellow', s=100, label='Goal')