import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from flight_controller.trajectory_reference import trajectory_reference, plot_ref_trajectory

class Animation:
    def get_motor_positions(self, constants, state):
        X, Y, Z, phi, theta, psi = state[6:]
        R = constants.R_matrix(phi, theta, psi)
        motor_offsets = np.array([
            [constants.l, 0, 0],  # Motor 1
            [0, constants.l, 0],  # Motor 2
            [-constants.l, 0, 0],  # Motor 3
            [0, -constants.l, 0]  # Motor 4
        ]).T
        return R @ motor_offsets + np.array([[X], [Y], [Z]])
        
    def animate_quadrotor(self, fig, ax_tpv, ax_non_tpv, constants, x, t,
                          xr=None, yr=None, zr=None):
        """
        Animates the quadrotor's 3D trajectory and its body frame.
        """
        
        # Fixed axis limits based on camera frame
        view_range = 2.0 
        camera_offset_body = np.array([view_range, view_range, 0.25])
        
        # Static view initialization
        alpha_damp = 0.05  # Damping factor for smoothing azimuth changes
        fixed_elev = 35
        
        # Initialize smoothed azimuth to the first heading angle minus 180 degrees (to position camera behind)
        ut_init, vt_init,_,_,_,_, xt_init, yt_init, zt_init,_,_,_ = x[0]
        
        if ut_init == 0 and vt_init == 0:
            initial_heading_angle = 0
        else:
            initial_heading_angle = np.degrees(np.arctan2(vt_init, ut_init))

        smoothed_azim = initial_heading_angle - 180
        
        # Compute camera position based on smoothed azimuth and camera_offset_body
        smoothed_azim_rad = np.radians(smoothed_azim)
        camera_pos_x_i = xt_init + camera_offset_body[0] * np.cos(smoothed_azim_rad) - camera_offset_body[1] * np.sin(smoothed_azim_rad)
        camera_pos_y_i = yt_init + camera_offset_body[0] * np.sin(smoothed_azim_rad) + camera_offset_body[1] * np.cos(smoothed_azim_rad)
        camera_pos_z_i = zt_init + camera_offset_body[2]

        ax_tpv.set_xlim([camera_pos_x_i - view_range, camera_pos_x_i + view_range])
        ax_tpv.set_ylim([camera_pos_y_i - view_range, camera_pos_y_i + view_range])
        ax_tpv.set_zlim([camera_pos_z_i - view_range, camera_pos_z_i + view_range])
        ax_tpv.view_init(elev=fixed_elev, azim=smoothed_azim+45)  # Fixed viewpoint
        
        for ax in [ax_tpv]:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
            # Markers for start and goal positions
            ax.scatter(x[0, 6], x[0, 7], x[0, 8], color='red', label="Start")
            ax.scatter(x[-1, 6], x[-1, 7], x[-1, 8], color='green', label="Goal")
            ax.legend(loc='upper right')
        
        # TPV
        drone_body1_tpv, = ax_tpv.plot([], [], [], 'k-', lw=2)  # Body arms1
        drone_body2_tpv, = ax_tpv.plot([], [], [], 'k-', lw=2)  # Body arms2
        motors_tpv, = ax_tpv.plot([], [], [], 'ro', markersize=4)  # Motor points
        trajectory_tpv, = ax_tpv.plot([], [], [], 'b--', lw=1, label="Trajectory")  # Trajectory line in blue
        ref_point_tpv, = ax_tpv.plot([], [], [], 'go', markersize=4)  # Reference point
        text_object_tpv = ax_tpv.text2D(0.05, 0.95, '', transform=ax_tpv.transAxes, 
                                    ha='left', va='top', fontsize=12, color='blue')  # Increased font size
        
        # Non-TPV
        drone_body1_non_tpv, = ax_non_tpv.plot([], [], 'k-', lw=2)  # Body arms1
        drone_body2_non_tpv, = ax_non_tpv.plot([], [], 'k-', lw=2)  # Body arms2
        motors_non_tpv, = ax_non_tpv.plot([], [], 'ro', markersize=4)  # Motor points
        trajectory_non_tpv, = ax_non_tpv.plot([], [], 'g--', lw=1, label="Trajectory")  # Trajectory line in green
        ref_point_non_tpv, = ax_non_tpv.plot([], [], 'go', markersize=4)  # Reference point
        text_object_non_tpv = ax_non_tpv.text(0.05, 0.95, '', transform=ax_non_tpv.transAxes, 
                                            ha='left', va='top', fontsize=12, color='blue')  # Increased font size
        
        # Plot reference trajectory on both subplots
        # plot_ref_trajectory(constants, t, ax_tpv, x=xr, y=yr, z=zr)
        # plot_ref_trajectory(constants, t, ax_non_tpv, x=xr, y=yr, z=zr)
        
        
        def angle_diff(desired, current):
            """Compute minimal difference between two angles."""
            diff = (desired - current + 180) % 360 - 180
            return diff
        
        def update_tpv(frame):
            nonlocal text_object_tpv, text_object_non_tpv, smoothed_azim
            print(f"Frame: {frame + 1} / {len(t)}", end="\r")
            
            # Extract current state
            state = x[frame]
            ut, vt, wt, pt, qt, rt, xt, yt, zt, phit, thetat, psit = state
            
            # Compute heading angle based on velocity components
            if ut == 0 and vt == 0:
                heading_angle = smoothed_azim + 180
            else:
                heading_angle = np.degrees(np.arctan2(vt, ut))
            
            # Compute minimal azimuth difference
            delta_azim = angle_diff(heading_angle - 180, smoothed_azim)
            
            smoothed_azim = (smoothed_azim + alpha_damp * delta_azim) % 360
            
            # Update view angles
            ax_tpv.view_init(elev=fixed_elev, azim=smoothed_azim+45)
                
            # Compute camera position based on smoothed azimuth and camera_offset_body
            smoothed_azim_rad = np.radians(smoothed_azim)
            camera_pos_x = xt + camera_offset_body[0] * np.cos(smoothed_azim_rad) - camera_offset_body[1] * np.sin(smoothed_azim_rad)
            camera_pos_y = yt + camera_offset_body[0] * np.sin(smoothed_azim_rad) + camera_offset_body[1] * np.cos(smoothed_azim_rad)
            camera_pos_z = zt + camera_offset_body[2]

            # Center the plot around the camera's position
            ax_tpv.set_xlim([camera_pos_x - view_range, camera_pos_x + view_range])
            ax_tpv.set_ylim([camera_pos_y - view_range, camera_pos_y + view_range])
            ax_tpv.set_zlim([camera_pos_z - view_range, camera_pos_z + view_range])
            
            # Extract references (only once, outside of update if possible, for efficiency)
            (X_ref, X_dot_ref, X_dot_dot_ref,
            Y_ref, Y_dot_ref, Y_dot_dot_ref,
            Z_ref, Z_dot_ref, Z_dot_dot_ref,
            psi_ref) = traj_ref = trajectory_reference(
                constants, t, x=xr, y=yr, z=zr
            )

            # Extract position references (second column)
            x_positions = X_ref[:,1]
            y_positions = Y_ref[:,1]
            z_positions = Z_ref[:,1]

            # Get current reference point
            x_ref = x_positions[frame]
            y_ref = y_positions[frame]
            z_ref = z_positions[frame]
                        
            # Compute error factor (Euclidean distance)
            error = np.sqrt((xt - x_ref)**2 + (yt - y_ref)**2 + (zt - z_ref)**2)
            
            # Precomputed motor positions from Quadrotor
            motor_positions = self.get_motor_positions(constants, state)
                        
            # Update quadrotor body and motors for TPV ###
            body_x1_tpv = [motor_positions[0, 0], motor_positions[0, 2]]
            body_y1_tpv = [motor_positions[1, 0], motor_positions[1, 2]]
            body_z1_tpv = [motor_positions[2, 0], motor_positions[2, 2]]
            
            body_x2_tpv = [motor_positions[0, 1], motor_positions[0, 3]]
            body_y2_tpv = [motor_positions[1, 1], motor_positions[1, 3]]
            body_z2_tpv = [motor_positions[2, 1], motor_positions[2, 3]]
            
            drone_body1_tpv.set_data(body_x1_tpv, body_y1_tpv)
            drone_body1_tpv.set_3d_properties(body_z1_tpv)
            
            drone_body2_tpv.set_data(body_x2_tpv, body_y2_tpv)
            drone_body2_tpv.set_3d_properties(body_z2_tpv)
            
            motors_tpv.set_data(motor_positions[0, :], motor_positions[1, :])
            motors_tpv.set_3d_properties(motor_positions[2, :])
            
            # Update trajectory and reference point for TPV
            trajectory_tpv.set_data(x[:frame, 6], x[:frame, 7])
            trajectory_tpv.set_3d_properties(x[:frame, 8])
            ref_point_tpv.set_data([x_ref], [y_ref])
            ref_point_tpv.set_3d_properties([z_ref])
            
            # Update trajectory and reference point for Non-TPV
            trajectory_non_tpv.set_data(x[:frame, 6], x[:frame, 7])
            ref_point_non_tpv.set_data([x_ref], [y_ref])
            
            # Update quadrotor body and motors for Non-TPV (in 2D)
            body_x1_non_tpv = [motor_positions[0, 0], motor_positions[0, 2]] # first  bar x
            body_y1_non_tpv = [motor_positions[1, 0], motor_positions[1, 2]] # first  bar y
            # body_z1_non_tpv = [motor_positions[2, 0], motor_positions[2, 2]] # first  bar z
            body_x2_non_tpv = [motor_positions[0, 1], motor_positions[0, 3]] # second bar x
            body_y2_non_tpv = [motor_positions[1, 1], motor_positions[1, 3]] # second bar y
            # body_z2_non_tpv = [motor_positions[2, 1], motor_positions[2, 3]] # second bar z
            
            drone_body1_non_tpv.set_data(body_x1_non_tpv, body_y1_non_tpv)            
            drone_body2_non_tpv.set_data(body_x2_non_tpv, body_y2_non_tpv)            
            motors_non_tpv.set_data(motor_positions[0, :], motor_positions[1, :])
            
            # Error Update
            text_object_tpv.set_text(f'Error: {error:.2f} m')
            text_object_non_tpv.set_text(f'Error: {error:.2f} m')
            
            return (drone_body1_tpv, drone_body2_tpv, motors_tpv, trajectory_tpv, ref_point_tpv, text_object_tpv,
                    drone_body1_non_tpv, drone_body2_non_tpv, motors_non_tpv, trajectory_non_tpv, ref_point_non_tpv, text_object_non_tpv)
        
        step = 1 # Divide frame rate
        
        anim = animation.FuncAnimation(fig, update_tpv, frames=range(0, len(t), step), interval=30*step, blit=True)
        anim.save("quadrotor_trajectory_TPV.mp4", writer="ffmpeg")
        print("\nSaved animation as quadrotor_trajectory_TPV.mp4")
        