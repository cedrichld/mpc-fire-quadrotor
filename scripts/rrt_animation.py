import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
import matplotlib.animation as animation
from forest.plot_forest import Forest_Plotting
from flight_controller.trajectory_reference import trajectory_reference, plot_ref_trajectory

class Animation:
    
    def __init__(self):
        # Video Settings
        self.step = 1 # Frame reduction multiplier
    
    def init_plot(self, space_dim):
        fig = plt.figure(figsize=(16, 9), dpi=150)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.2)
        
        ax = fig.add_subplot(122)
        ax.set_xlim(0, space_dim[0])
        ax.set_ylim(0, space_dim[1])
        
        return fig, ax

    def animate_rrt(self, fig, ax, forest_info, total_iterations):
        """
        Animates the quadrotor's 3D trajectory and its body frame.
        """
        # Unpack forest info
        space_dim, trees_outside, fire_zone_trees, fire_zone, start_pos, goal_pos = forest_info # Not used anymore
        forest_plot = Forest_Plotting() # might move
        
        # 2D forest
        self.init_plot(space_dim)
        forest_plot.visualize_forest_2d(
            trees_outside, fire_zone, start_pos, goal_pos, fire_zone_trees, ax=ax
        )
        
        # Dynamic Variables to plot every frame
        text_object  = ax.text(
            0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top',
            fontsize=12, color='blue'
        )
        
        best_trajectory,  = ax.plot([], [], 'g--', lw=1, label="Trajectory")
        discovered_points = ax.plot([], [], 'yo', markersize=0.5)
        
        def update(frame): # a frame represents an iteration
            print(f"Frame: {frame+1}/{total_iterations}", end="\r")

            
            best_trajectory.set_data(best_current_path[:, 0], best_current_path[:, 1])
            discovered_points.set_data([x_ref], [y_ref])
            
            iteration_proportion = "█" * frame / total_iterations
            text_object.set_text(f'Trial: {iteration_proportion} m')
            return (
                drone_body1_tpv, drone_body2_tpv, motors_tpv, trajectory_tpv,
                ref_point_tpv, text_object_tpv, motors_non_tpv,
                trajectory_non_tpv, ref_point_non_tpv, text_object_non_tpv
            )
        
        # Tune step if you want fewer frames in the final video
        step = self.step
        # 10 fps for step 1
        anim = animation.FuncAnimation(
            fig, update, frames=range(0, len(t), step), interval=step, blit=False
        )
        anim.save("quadrotor_trajectory_TPV.mp4", writer="ffmpeg")
        print("\nSaved animation as quadrotor_trajectory_TPV.mp4")
