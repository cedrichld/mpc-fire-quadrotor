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
        # Camera settings
        self.view_range = 1.5  # FOV
        self.camera_offset_body = np.array([self.view_range, self.view_range, 0.1])  # Camera distance
        self.azim_offset = 45  
        
        # Static view initialization
        self.alpha_damp = 0.05  # Damping factor for smoothing azimuth changes
        self.fixed_elev = 9
        
        self.tree_view_range = 25.0
        self.alpha_tree = 1.0
        self.p_thresh = 1.0
        
        self.angle_thresh = 15
        self.alpha_angle_factor = 0.5
        self.alpha_dist_factor = 3
        self.dotp_thresh = -0.2 # Scale to decide which trees get potted (where -1 < dotp < 1)
        
        # Video Settings
        self.step = 1 # Frame reduction multiplier
        ''' 
        Use case:
        '''

        
    def get_motor_positions(self, constants, state):
        X, Y, Z, phi, theta, psi = state[6:]
        R = constants.R_matrix(phi, theta, psi)
        motor_offsets = np.array([
            [constants.l, 0, 0],   # Motor 1
            [0, constants.l, 0],   # Motor 2
            [-constants.l, 0, 0],  # Motor 3
            [0, -constants.l, 0]   # Motor 4
        ]).T
        return R @ motor_offsets + np.array([[X], [Y], [Z]])
        
    def plot_single_tree(self, ax, center, radius, height, color='forestgreen', alpha=1.0):
        """
        Plots a single cylindrical tree on a 3D Axes object.
        Returns the surface handle (Poly3DCollection).
        """
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, 1, 10)

        x = radius * np.outer(np.cos(u), np.ones(len(v))) + center[0]
        y = radius * np.outer(np.sin(u), np.ones(len(v))) + center[1]
        z = height * np.outer(np.ones(len(u)), v) + center[2]

        tree_surf = ax.plot_surface(x, y, z, color=color, alpha=alpha)
        return tree_surf

    def animate_quadrotor(self, fig, ax_tpv, ax_non_tpv, constants, states, t,
                          forest_info, xr=None, yr=None, zr=None):
        """
        Animates the quadrotor's 3D trajectory and its body frame.
        """
        view_range = self.view_range 
        camera_offset_body = self.camera_offset_body
        alpha_damp = self.alpha_damp
        fixed_elev = self.fixed_elev
        azim_offset = self.azim_offset
        
        # Unpack forest info
        space_dim, trees_outside, fire_zone_trees, fire_zone, start_pos, goal_pos = forest_info # Not used anymore
        alpha_tree = self.alpha_tree
        p_thresh = self.p_thresh
        tree_view_range = self.tree_view_range
        # quad_width_thresh = constants.l + 0.15
        forest_plot = Forest_Plotting() # might move
        
        ## Add Ground (from plot_forest.py, visualize_forest)
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
        ax_tpv.add_collection3d(ground_square)
        
        # Plot & store trees as tuples (tree_surface, tree_object)
        self.tree_plots_outside = []
        for tree in trees_outside:
            coll = self.plot_single_tree(
                ax_tpv,
                center=tree.center,
                radius=tree.radius,
                height=tree.height,
                color='forestgreen',
                alpha=alpha_tree
            )
            self.tree_plots_outside.append((coll, tree))
        
        self.tree_plots_fire_zone = []
        if fire_zone_trees:
            self.tree_plots_fire_zone = []
            for tree in fire_zone_trees:
                coll = self.plot_single_tree(
                    ax_tpv,
                    center=tree.center,
                    radius=tree.radius,
                    height=tree.height,
                    color='orangered',
                    alpha=alpha_tree
                )
                self.tree_plots_fire_zone.append((coll, tree))
        
        # 2D forest
        forest_plot.visualize_forest_2d(
            trees_outside, fire_zone, start_pos, goal_pos, fire_zone_trees, ax=ax_non_tpv
        )
        
        # Extract initial data
        ut_init, vt_init,_,_,_,_, xt_init, yt_init, zt_init,_,_,_ = states[0]

        # Initialize smoothed azimuth to heading angle - 180
        if ut_init == 0 and vt_init == 0:
            initial_heading_angle = 0
        else:
            initial_heading_angle = np.degrees(np.arctan2(vt_init, ut_init))
        smoothed_azim = initial_heading_angle - 180
        
        # Initial camera position
        smoothed_azim_rad = np.radians(smoothed_azim)
        camera_pos_x_i = xt_init + camera_offset_body[0]*np.cos(smoothed_azim_rad) \
                                   - camera_offset_body[1]*np.sin(smoothed_azim_rad)
        camera_pos_y_i = yt_init + camera_offset_body[0]*np.sin(smoothed_azim_rad) \
                                   + camera_offset_body[1]*np.cos(smoothed_azim_rad)
        camera_pos_z_i = zt_init + camera_offset_body[2]

        ax_tpv.set_xlim([camera_pos_x_i - view_range, camera_pos_x_i + view_range])
        ax_tpv.set_ylim([camera_pos_y_i - view_range, camera_pos_y_i + view_range])
        ax_tpv.set_zlim([camera_pos_z_i - view_range, camera_pos_z_i + view_range])
        ax_tpv.view_init(elev=fixed_elev, azim=smoothed_azim + azim_offset)

        for axv in [ax_tpv]:
            axv.set_xlabel('X (m)')
            axv.set_ylabel('Y (m)')
            axv.set_zlabel('Z (m)')
            axv.scatter(states[0,6], states[0,7], states[0,8], color='red', label="Start")
            axv.scatter(states[-1,6], states[-1,7], states[-1,8], color='green', label="Goal")
            axv.legend(loc='upper right')
        
        # 3D TPV
        drone_body1_tpv, = ax_tpv.plot([], [], [], 'k-', lw=6)  # Body arms1
        drone_body2_tpv, = ax_tpv.plot([], [], [], 'k-', lw=6)  # Body arms2
        motors_tpv,      = ax_tpv.plot([], [], [], 'ro', markersize=9)  # Motor points
        trajectory_tpv,  = ax_tpv.plot([], [], [], 'b--', lw=1, label="Trajectory")
        ref_point_tpv,   = ax_tpv.plot([], [], [], 'go', markersize=6)
        text_object_tpv  = ax_tpv.text2D(
            0.05, 0.95, '', transform=ax_tpv.transAxes, ha='left', va='top',
            fontsize=12, color='blue'
        )
        
        # 2D Non-TPV
        motors_non_tpv,      = ax_non_tpv.plot([], [], 'ro', markersize=5)
        trajectory_non_tpv,  = ax_non_tpv.plot([], [], 'g--', lw=1, label="Trajectory")
        ref_point_non_tpv,   = ax_non_tpv.plot([], [], 'go', markersize=4)
        text_object_non_tpv  = ax_non_tpv.text(
            0.05, 0.95, '', transform=ax_non_tpv.transAxes, ha='left', va='top',
            fontsize=12, color='blue'
        )
        
        # Initialize forward vector arrow in the 3D plot
        # right_arrow = ax_tpv.quiver(0, 0, 0, 0, 0, 0, color='red', length=2.0, normalize=True, linewidth=2)
        # left_arrow = ax_tpv.quiver(0, 0, 0, 0, 0, 0, color='green', length=2.0, normalize=True, linewidth=2)

        # Minimal angle difference function
        def angle_diff(desired, current):
            diff = (desired - current + 180) % 360 - 180
            return diff

        def update(frame):
            state = states[frame]
            ut, vt,_,_,_,_, xt, yt, zt,_,_,_ = state
            
            print(f"Frame: {frame+1}/{len(t)} -> Drone=({xt:.2f}, {yt:.2f}, {zt:.2f})", end="\r")
            
            # Update heading
            nonlocal smoothed_azim
            if ut == 0 and vt == 0:
                heading_angle = smoothed_azim + 180
            else:
                heading_angle = np.degrees(np.arctan2(vt, ut))
            delta_azim = angle_diff(heading_angle - 180, smoothed_azim)
            smoothed_azim = (smoothed_azim + alpha_damp * delta_azim) % 360
            
            ax_tpv.view_init(elev=fixed_elev, azim=smoothed_azim + azim_offset)
            
            # Camera position
            smoothed_azim_rad = np.radians(smoothed_azim)
            camera_pos_x = xt + camera_offset_body[0]*np.cos(smoothed_azim_rad) \
                                 - camera_offset_body[1]*np.sin(smoothed_azim_rad)
            camera_pos_y = yt + camera_offset_body[0]*np.sin(smoothed_azim_rad) \
                                 + camera_offset_body[1]*np.cos(smoothed_azim_rad)
            camera_pos_z = zt + camera_offset_body[2]

            # Adjust plot bounds
            x_min, x_max = camera_pos_x - view_range, camera_pos_x + view_range
            y_min, y_max = camera_pos_y - view_range, camera_pos_y + view_range
            z_min, z_max = camera_pos_z - view_range, camera_pos_z + view_range
            ax_tpv.set_xlim([x_min, x_max])
            ax_tpv.set_ylim([y_min, y_max])
            ax_tpv.set_zlim([z_min, z_max])
            
            # Determine forward vector
            forward_vector = np.array([
                np.cos(np.radians(smoothed_azim + 180 + 45)),
                np.sin(np.radians(smoothed_azim + 180 + 45))
            ]) # Norm is 1
            
            # Outside trees
            for tree_plot, tree_obj in self.tree_plots_outside:
                # Tree center relative to drone
                tree_rel = np.array([tree_obj.center[0] - xt, tree_obj.center[1] - yt])
                dist = np.linalg.norm(tree_rel) 

                if dist < 1e-9:
                    # Safety check; if the tree is basically at the drone's position
                    tree_plot.set_visible(False)
                    print(f"TREE {np.round(tree_obj.center)} IS IN COLLISION, Drone=({xt:.2f}, {yt:.2f}, {zt:.2f})")
                    continue
                
                tree_rel_unit = tree_rel / dist
                dotp = np.dot(forward_vector, tree_rel_unit) # Like this we have: -1 < dotp < 1
                
                # bounding box logic
                in_front_bounds = (
                    x_min - p_thresh * tree_obj.radius <= tree_obj.center[0] <= x_max + p_thresh * tree_obj.radius + tree_view_range
                    and
                    y_min - p_thresh * tree_obj.radius <= tree_obj.center[1] <= y_max + p_thresh * tree_obj.radius + tree_view_range
                )

                if dotp > self.dotp_thresh and in_front_bounds:
                    tree_plot.set_visible(True)
                    
                    # angle in [0,180], where 0 => directly in front, 180 => directly behind
                    angle_deg = np.degrees(np.arccos(dotp))
                    
                    if angle_deg < self.angle_thresh:
                        alpha_angle = 1 - (self.angle_thresh - self.alpha_angle_factor * angle_deg) / self.angle_thresh
                        alpha_angle = max(alpha_angle, 0.0)
                    else:
                        alpha_angle = 1.0
                        
                    distMin = tree_obj.radius + 0.1
                    
                    if dist < distMin:
                        alpha_dist = 0
                        print("\nDist min JUST HIT, drone at 0.1m from trunk!\n")
                    else:
                        alpha_dist = self.alpha_dist_factor * dist / tree_view_range
                        alpha_dist = max(alpha_dist, 0.0)
                    
                    raw_alpha = alpha_dist * alpha_angle
                    alpha = max(min(raw_alpha, 1.0), 0.0)
                    # print(f"Drone: ({xt:.2f}, {yt:.2f}, {zt:.2f}), distance: {round(dist, ndigits=1)}, angle: {round(angle_deg)}, alpha: {round(alpha, ndigits=3)}")
                    tree_plot.set_alpha(alpha)
                    if alpha == 0:
                        tree_plot.set_visible(False)

                else:
                    tree_plot.set_visible(False)

            # Fire zone trees
            for tree_plot, tree_obj in self.tree_plots_fire_zone:
                # Tree center relative to drone
                tree_rel = np.array([tree_obj.center[0] - xt, tree_obj.center[1] - yt])
                dist = np.linalg.norm(tree_rel) 

                if dist < 1e-9:
                    # Safety check; if the tree is basically at the drone's position
                    tree_plot.set_visible(False)
                    print(f"TREE {np.round(tree_obj.center)} IS IN COLLISION, Drone=({xt:.2f}, {yt:.2f}, {zt:.2f})")
                    continue
                
                tree_rel_unit = tree_rel / dist
                dotp = np.dot(forward_vector, tree_rel_unit) # Like this we have: -1 < dotp < 1
                
                # bounding box logic
                in_front_bounds = (
                    x_min - p_thresh * tree_obj.radius <= tree_obj.center[0] <= x_max + p_thresh * tree_obj.radius + tree_view_range
                    and
                    y_min - p_thresh * tree_obj.radius <= tree_obj.center[1] <= y_max + p_thresh * tree_obj.radius + tree_view_range
                )

                if dotp > self.dotp_thresh and in_front_bounds:
                    tree_plot.set_visible(True)
                    
                    # 1) **Angle-based factor**: 
                    # angle in [0,180], where 0 => directly in front, 180 => directly behind
                    angle_deg = np.degrees(np.arccos(dotp))
                    
                    if angle_deg < self.angle_thresh:
                        alpha_angle = 1 - self.alpha_angle_factor * (90 - angle_deg) / 90
                        alpha_angle = max(min(alpha_angle, 1.0), 0.0)
                    else:
                        alpha_angle = 1.0
                        
                    distMin = tree_obj.radius + 0.1
                    
                    if dist < distMin:
                        alpha_dist = 0
                        print("\nDist min JUST HIT, drone at 0.1m from trunk!\n")
                    else:
                        alpha_dist = self.alpha_dist_factor * dist / tree_view_range
                        alpha_dist = max(min(alpha_dist, 1.0), 0.0)
                    
                    raw_alpha = alpha_dist * alpha_angle
                    alpha = max(min(raw_alpha, 1.0), 0.0)
                    tree_plot.set_alpha(alpha)

                else:
                    tree_plot.set_visible(False)

            # Plot trajectory so far
            trajectory_tpv.set_data(states[:frame, 6], states[:frame, 7])
            trajectory_tpv.set_3d_properties(states[:frame, 8])
            
            # Reference point
            ref_point_tpv.set_data([xt], [yt])
            ref_point_tpv.set_3d_properties([zt])

            # Extract references
            X_ref,_,_,Y_ref,_,_,Z_ref,_,_,_ = trajectory_reference(constants, t, x=xr, y=yr, z=zr)
            x_ref = X_ref[frame, 1]
            y_ref = Y_ref[frame, 1]
            z_ref = Z_ref[frame, 1]
            
            # Error
            error = np.sqrt((xt - x_ref)**2 + (yt - y_ref)**2 + (zt - z_ref)**2)
            
            # Quadrotor body
            motor_positions = self.get_motor_positions(constants, state)
            # Body bar #1
            body_x1 = [motor_positions[0,0], motor_positions[0,2]]
            body_y1 = [motor_positions[1,0], motor_positions[1,2]]
            body_z1 = [motor_positions[2,0], motor_positions[2,2]]
            # Body bar #2
            body_x2 = [motor_positions[0,1], motor_positions[0,3]]
            body_y2 = [motor_positions[1,1], motor_positions[1,3]] #
            body_z2 = [motor_positions[2,1], motor_positions[2,3]]
            
            drone_body1_tpv.set_data(body_x1, body_y1)
            drone_body1_tpv.set_3d_properties(body_z1)
            drone_body2_tpv.set_data(body_x2, body_y2)
            drone_body2_tpv.set_3d_properties(body_z2)
            motors_tpv.set_data(motor_positions[0,:], motor_positions[1,:])
            motors_tpv.set_3d_properties(motor_positions[2,:])

            # Update reference marker in 3D
            ref_point_tpv.set_data([x_ref], [y_ref])
            ref_point_tpv.set_3d_properties([z_ref])

            # Update 2D
            trajectory_non_tpv.set_data(states[:frame,6], states[:frame,7])
            ref_point_non_tpv.set_data([x_ref], [y_ref])

            # Quadrotor in 2D
            motors_non_tpv.set_data(motor_positions[0,:], motor_positions[1,:])

            text_object_tpv.set_text(f'Error: {error:.2f} m')
            text_object_non_tpv.set_text(f'Error: {error:.2f} m')

            return (
                drone_body1_tpv, drone_body2_tpv, motors_tpv, trajectory_tpv,
                ref_point_tpv, text_object_tpv, motors_non_tpv,
                trajectory_non_tpv, ref_point_non_tpv, text_object_non_tpv
            )
        
        # Tune step if you want fewer frames in the final video
        step = self.step
        anim = animation.FuncAnimation(
            fig, update, frames=range(0, len(t), step), interval=30*step, blit=False
        )
        anim.save("quadrotor_trajectory_TPV.mp4", writer="ffmpeg")
        print("\nSaved animation as quadrotor_trajectory_TPV.mp4")



            # # Unit forward & left vectors
            # orth_vector = np.array([-forward_vector[1], forward_vector[0]]) # 90 degree turn
            # left_arrow.remove()
            # right_arrow.remove()
            # left_arrow = ax_tpv.quiver(
            #     xt + orth_vector[0] * 2 * quad_width_thresh, yt + orth_vector[1] * 2 * quad_width_thresh, zt,  # Starting point of the arrow
            #     forward_vector[0], forward_vector[1], 0,  # Direction of the arrow
            #     color='red', length=2.0, normalize=True, linewidth=2
            # )
            # right_arrow = ax_tpv.quiver(
            #     xt - orth_vector[0] * 2 * quad_width_thresh, yt - orth_vector[1] * 2 * quad_width_thresh, zt,  # Starting point of the arrow
            #     forward_vector[0], forward_vector[1], 0,  # Direction of the arrow
            #     color='green', length=2.0, normalize=True, linewidth=2
            # )
            
            # Outside trees
            # for tree_plot, tree_obj in self.tree_plots_outside:
            #     # Tree center relative to drone
            #     tree_rel = np.array([tree_obj.center[0] - xt, tree_obj.center[1] - yt])
            #     dist = np.linalg.norm(tree_rel)
                
            #     if dist < 1e-9:
            #         # Safety check; if the tree is basically at the drone's position
            #         tree_plot.set_visible(False)
            #         print(f"TREE {np.round(tree_obj.center)} IS IN COLLISION, Drone=({xt:.2f}, {yt:.2f}, {zt:.2f})")
            #         continue
                
            #     tree_rel_rad1 = tree_rel + np.array([tree_obj.radius, tree_obj.radius])
            #     tree_rel_rad2 = tree_rel - np.array([tree_obj.radius, tree_obj.radius])
                
            #     # side_dist is the minimal left/right offset among the corners
            #     side_dist1 = abs(np.dot(tree_rel_rad1, orth_vector))
            #     side_dist2 = abs(np.dot(tree_rel_rad2, orth_vector))
            #     side_dist = min(side_dist1, side_dist2)
                
            #     if side_dist <= quad_width_thresh:
            #         tree_plot.set_visible(False)                      
            #         ax_non_tpv.plot([tree_obj.center[0]], [tree_obj.center[1]],
            #                         'ro', markersize=4)
            #         # print(f"Tree too close to drone view: {side_dist}")
            #         continue                

            #     # Also check if the tree is "in front" / within plotting bounds by dot product with forward_vector
            #     tree_rel_unit = tree_rel / dist
            #     dotp = np.dot(forward_vector, tree_rel_unit)  # -1 < dotp < 1

            #     # bounding box logic
            #     in_front_bounds = (
            #         x_min - p_thresh * tree_obj.radius <= tree_obj.center[0] <= x_max + p_thresh * tree_obj.radius + tree_view_range
            #         and
            #         y_min - p_thresh * tree_obj.radius <= tree_obj.center[1] <= y_max + p_thresh * tree_obj.radius + tree_view_range
            #     )
                
            #     if dotp > self.dotp_thresh and in_front_bounds:
                    
            #         raw_side_alpha = (side_dist - quad_width_thresh) / quad_width_thresh # quad_width_thresh=/2
            #         # clamp to [0,1]
            #         alpha = max(min(raw_side_alpha, 1.0), 0.0)
                    
            #         tree_plot.set_visible(True)
            #         tree_plot.set_alpha(alpha)
                    
            #         if alpha < 1.0:
            #             ax_non_tpv.plot([tree_obj.center[0]], [tree_obj.center[1]], 'bo', markersize=4) 
            #     else:
            #         tree_plot.set_visible(False)
            #         ax_non_tpv.plot([tree_obj.center[0]], [tree_obj.center[1]],
            #                             'ro', markersize=4)
                    