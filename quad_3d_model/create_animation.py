import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from quadrotor import Quadrotor

def animate_quadrotor(x, t, Q, R, Qf, arm_length=0.5):
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
    quadrotor = Quadrotor(Q, R, Qf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-2, 2])
    ax.set_ylim3d([-2, 2])
    ax.set_zlim3d([0, 4])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("Quadrotor 3D Animation")

    # Initial quadrotor position
    drone_body, = ax.plot([], [], [], 'k-', lw=2)  # Body arms
    motors, = ax.plot([], [], [], 'ro', markersize=5)  # Motor points
    trajectory, = ax.plot([], [], [], 'b--', lw=1)  # Trajectory line

    def update(frame):
        # Extract current state
        state = x[frame]
        X, Y, Z = state[0:3]
        phi, theta, psi = state[6:9]

        # Precomputed motor positions from Quadrotor
        motor_positions = quadrotor.get_motor_positions(state)

        # Update body arms
        body_x = [motor_positions[0, 0], motor_positions[0, 2], motor_positions[0, 1], motor_positions[0, 3], motor_positions[0, 0]]
        body_y = [motor_positions[1, 0], motor_positions[1, 2], motor_positions[1, 1], motor_positions[1, 3], motor_positions[1, 0]]
        body_z = [motor_positions[2, 0], motor_positions[2, 2], motor_positions[2, 1], motor_positions[2, 3], motor_positions[2, 0]]
        drone_body.set_data(body_x, body_y)
        drone_body.set_3d_properties(body_z)

        # Update motor points
        motors.set_data(motor_positions[0, :], motor_positions[1, :])
        motors.set_3d_properties(motor_positions[2, :])

        # Update trajectory
        trajectory.set_data(x[:frame, 0], x[:frame, 1])
        trajectory.set_3d_properties(x[:frame, 2])

        return drone_body, motors, trajectory

    anim = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)
    anim.save("quadrotor_trajectory.mp4", writer="ffmpeg")
    print("Saved animation as quadrotor_trajectory.mp4")