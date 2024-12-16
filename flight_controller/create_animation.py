import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# from quadrotor import Quadrotor

from trajectory_testing import trajectory_reference, plot_ref_trajectory

class Animation:
    def __init__(self):
        self.text_object = None  # Store reference to the text object
        
    def get_motor_positions(self, constants, state):
        X, Y, Z, phi, theta, psi = state[6:]
        # print(f"states create anim: {state}")
        R = constants.R_matrix(phi, theta, psi)
        motor_offsets = np.array([
            [constants.l, 0, 0],  # Motor 1
            [0, constants.l, 0],  # Motor 2
            [-constants.l, 0, 0],  # Motor 3
            [0, -constants.l, 0]  # Motor 4
        ]).T
        return R @ motor_offsets + np.array([[X], [Y], [Z]])
        
    
    def animate_quadrotor(self, constants, x, t, x_limits, y_limits, z_limits):
        """
        Animates the quadrotor's 3D trajectory and its body frame.
        """
        # quadrotor = Quadrotor()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # print(f"x_limits, y_limits, z_limits: {x_limits, y_limits, z_limits}")
        ax.set_xlim3d(x_limits)
        ax.set_ylim3d(y_limits)
        ax.set_zlim3d(z_limits)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.scatter(x[0, 6], x[0, 7], x[0, 8], color='red', label="Start")
        ax.scatter(x[-1, 6], x[-1, 7], x[-1, 8], color='green', label="Goal")
        ax.set_title("Quadrotor 3D Animation")
        ax.legend()

        # Initial quadrotor position
        drone_body1, = ax.plot([], [], [], 'k-', lw=2)  # Body arms1        
        drone_body2, = ax.plot([], [], [], 'k-', lw=2)  # Body arms2
        motors, = ax.plot([], [], [], 'ro', markersize=2)  # Motor points
        trajectory, = ax.plot([], [], [], 'r--', lw=1, label="Trajectory")  # Trajectory line
        ref_point, = ax.plot([], [], [], 'go', markersize=2)  # Motor points
        
        plot_ref_trajectory(constants, t, ax)
        
        
        # x_ref, y_ref, z_ref = [], [] , []
        
        text_object = None  # Store reference to the text object

        def update_bf(frame):
            print(f"Frame: {frame + 1} / {len(t)}", end="\r")
            # Extract current state
            frame = frame
            state = x[frame]
            # Extract position and orientation
            x_state, y_state, z_state = state[6:9]
            phi, theta, psi = state[9:12]
            
            # Get rotation and translation matrices
            R = constants.R_matrix(phi, theta, psi)
            T = np.array([x_state, y_state, z_state])
            
            # Transform all coordinates into the body frame
            def transform_to_body_frame(coords):
                world_coords = np.array(coords).T
                body_coords = R.T @ (world_coords - T.reshape(-1, 1))
                return body_coords.T
            
            # Extract references (only once, outside of update if possible, for efficiency)
            (X_ref, X_dot_ref, X_dot_dot_ref,
            Y_ref, Y_dot_ref, Y_dot_dot_ref,
            Z_ref, Z_dot_ref, Z_dot_dot_ref,
            psi_ref) = trajectory_reference(constants, t)

            # Extract position references (second column)
            x_positions = X_ref[:,1]
            y_positions = Y_ref[:,1]
            z_positions = Z_ref[:,1]

            # Get current reference point
            x_ref = x_positions[frame]
            y_ref = y_positions[frame]
            z_ref = z_positions[frame]
            
            # Compute error factor (Euclidean distance)
            error = np.sqrt((x_state - x_ref)**2 + (y_state - y_ref)**2 + (z_state - z_ref)**2)
            
            # Precomputed motor positions from Quadrotor
            motor_positions = self.get_motor_positions(constants, state)
            
            

            # Update body arms
            body_x1 = [motor_positions[0, 0], motor_positions[0, 2]]
            body_y1 = [motor_positions[1, 0], motor_positions[1, 2]]
            body_z1 = [motor_positions[2, 0], motor_positions[2, 2]]
            
            body_x2 = [motor_positions[0, 1], motor_positions[0, 3]]
            body_y2 = [motor_positions[1, 1], motor_positions[1, 3]]
            body_z2 = [motor_positions[2, 1], motor_positions[2, 3]]
            
            drone_body1.set_data(body_x1, body_y1)
            drone_body1.set_3d_properties(body_z1)
            
            drone_body2.set_data(body_x2, body_y2)
            drone_body2.set_3d_properties(body_z2)

            # Update ref point
            ref_point.set_data([x_ref], [y_ref])
            ref_point.set_3d_properties([z_ref])
            
            # Update motor points
            motors.set_data(motor_positions[0, :], motor_positions[1, :])
            motors.set_3d_properties(motor_positions[2, :])

            # Update trajectory
            trajectory.set_data(x[:frame, 6], x[:frame, 7])
            trajectory.set_3d_properties(x[:frame, 8])
            
            if self.text_object is not None:
                self.text_object.remove()
            # Display Error
            self.text_object = ax.text2D(
                0.95, 0.95, f'Error: {error.item():.2f} m',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, color='blue'
            )

            
            return drone_body1, drone_body2, motors, trajectory, ref_point
        
        
        def update(frame):
            print(f"Frame: {frame + 1} / {len(t)}", end="\r")
            # Extract current state
            frame = frame
            state = x[frame]
            # Extract references (only once, outside of update if possible, for efficiency)
            (X_ref, X_dot_ref, X_dot_dot_ref,
            Y_ref, Y_dot_ref, Y_dot_dot_ref,
            Z_ref, Z_dot_ref, Z_dot_dot_ref,
            psi_ref) = trajectory_reference(constants, t)

            # Extract position references (second column)
            x_positions = X_ref[:,1]
            y_positions = Y_ref[:,1]
            z_positions = Z_ref[:,1]

            # Get current reference point
            x_ref = x_positions[frame]
            y_ref = y_positions[frame]
            z_ref = z_positions[frame]
            
            # Current position
            x_state, y_state, z_state = state[6:9]
            # Compute error factor (Euclidean distance)
            error = np.sqrt((x_state - x_ref)**2 + (y_state - y_ref)**2 + (z_state - z_ref)**2)
            
            # Precomputed motor positions from Quadrotor
            motor_positions = self.get_motor_positions(constants, state)
            
            

            # Update body arms
            body_x1 = [motor_positions[0, 0], motor_positions[0, 2]]
            body_y1 = [motor_positions[1, 0], motor_positions[1, 2]]
            body_z1 = [motor_positions[2, 0], motor_positions[2, 2]]
            
            body_x2 = [motor_positions[0, 1], motor_positions[0, 3]]
            body_y2 = [motor_positions[1, 1], motor_positions[1, 3]]
            body_z2 = [motor_positions[2, 1], motor_positions[2, 3]]
            
            drone_body1.set_data(body_x1, body_y1)
            drone_body1.set_3d_properties(body_z1)
            
            drone_body2.set_data(body_x2, body_y2)
            drone_body2.set_3d_properties(body_z2)

            # Update ref point
            ref_point.set_data([x_ref], [y_ref])
            ref_point.set_3d_properties([z_ref])
            
            # Update motor points
            motors.set_data(motor_positions[0, :], motor_positions[1, :])
            motors.set_3d_properties(motor_positions[2, :])

            # Update trajectory
            trajectory.set_data(x[:frame, 6], x[:frame, 7])
            trajectory.set_3d_properties(x[:frame, 8])
            
            if self.text_object is not None:
                self.text_object.remove()
            # Display Error
            self.text_object = ax.text2D(
                0.95, 0.95, f'Error: {error.item():.2f} m',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, color='blue'
            )

            
            return drone_body1, drone_body2, motors, trajectory, ref_point

        anim = animation.FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
        anim.save("quadrotor_trajectory.mp4", writer="ffmpeg")
        print("\nSaved animation as quadrotor_trajectory.mp4")
        