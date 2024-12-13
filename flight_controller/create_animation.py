import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from quadrotor import Quadrotor

from trajectory_testing import trajectory_references, plot_ref_trajectory

class Animation:
    def __init__(self):
        self.text_object = None  # Store reference to the text object
        
    def animate_quadrotor(self, x, t, x_limits, y_limits, z_limits):
        """
        Animates the quadrotor's 3D trajectory and its body frame.

        Parameters:
        x : ndarray
            State trajectory (n_steps x 12).
        t : ndarray
            Time steps (n_steps).
        arm_length : float
            Length of the quadrotor's arms.
        """
        quadrotor = Quadrotor()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim3d(x_limits)
        # ax.set_ylim3d(y_limits + np.array([-0.25, 0.25]))
        # ax.set_zlim3d(z_limits)
        ax.set_xlim3d([-10, 0])
        ax.set_ylim3d([-0.5, 10])
        ax.set_zlim3d([0, 15])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.scatter(x[0, 0], x[0, 1], x[0, 2], color='red', label="Start")
        ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], color='green', label="Goal")
        ax.set_title("Quadrotor 3D Animation")
        ax.legend()

        # Initial quadrotor position
        drone_body1, = ax.plot([], [], [], 'k-', lw=2)  # Body arms1        
        drone_body2, = ax.plot([], [], [], 'k-', lw=2)  # Body arms2
        motors, = ax.plot([], [], [], 'ro', markersize=2)  # Motor points
        trajectory, = ax.plot([], [], [], 'b--', lw=1, label="Trajectory")  # Trajectory line
        ref_point, = ax.plot([], [], [], 'go', markersize=2)  # Motor points
        
        plot_ref_trajectory(ax)
        # x_ref, y_ref, z_ref = [], [] , []
        
        text_object = None  # Store reference to the text object

        def update(frame):
            # Extract current state
            frame = frame * 5
            state = x[frame]
            x_ref, y_ref, z_ref,_,_,_,_,_,_,_,_,_ = trajectory_references(t[frame])
            
            x_ref, y_ref, z_ref = np.array([x_ref]), np.array([y_ref]), np.array([z_ref])
            
            # Current position
            x_state, y_state, z_state = state[0:3]
            # Compute error factor (Euclidean distance)
            error = np.sqrt((x_state - x_ref)**2 + (y_state - y_ref)**2 + (z_state - z_ref)**2)
            
            # Precomputed motor positions from Quadrotor
            motor_positions = quadrotor.get_motor_positions(state)

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
            ref_point.set_data(x_ref, y_ref)
            ref_point.set_3d_properties(z_ref)
            
            # Update motor points
            motors.set_data(motor_positions[0, :], motor_positions[1, :])
            motors.set_3d_properties(motor_positions[2, :])

            # Update trajectory
            trajectory.set_data(x[:frame, 0], x[:frame, 1])
            trajectory.set_3d_properties(x[:frame, 2])
            
            if self.text_object is not None:
                self.text_object.remove()
            # Display Error
            self.text_object = ax.text2D(
                0.95, 0.95, f'Error: {error.item():.2f} m',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, color='blue'
            )

            
            return drone_body1, drone_body2, motors, trajectory, ref_point

        anim = animation.FuncAnimation(fig, update, frames=(int)(len(t) / 5), interval=50, blit=True)
        anim.save("quadrotor_trajectory.mp4", writer="ffmpeg")
        print("Saved animation as quadrotor_trajectory.mp4")
        