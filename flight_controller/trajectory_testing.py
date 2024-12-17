import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from flight_controller.init_constants import Quad_Constants

def trajectory_reference(constants, t, r=2, f=0.025, height_i=2, height_f=5, x=None, y=None, z=None):
    """
    Python version of the MATLAB trajectory_generator function.
    
    Parameters:
    - t: numpy array of time steps
    - r: radius or a trajectory scale factor
    - f: frequency factor
    - height_i: initial height
    - height_f: final height
    - constants: dictionary of constants (from init_constants), must contain:
        Ts: sampling time
        innerDyn_length: number of inner control loop iterations
        trajectory: trajectory selection index (1, 2, 3, or 4)
    
    Returns:
    - X_ref, X_dot_ref, X_dot_dot_ref,
      Y_ref, Y_dot_ref, Y_dot_dot_ref,
      Z_ref, Z_dot_ref, Z_dot_dot_ref,
      psi_ref
      Each returned as a 2D numpy array of shape (len(t), 2):
         column 0: time
         column 1: corresponding trajectory value
    """
    Ts = constants.Ts
    innerDyn_length = constants.innerDyn_length
    trajectory = 0 #constants.trajectory

    alpha = 2 * np.pi * f * t
    d_height = height_f - height_i

    # Compute trajectories based on the selected trajectory
    if trajectory == 1:
        print(f"Whoops: {x,y,z}")
        x = 1.5 * t / 10 + 1 + 3 * np.cos(t / 5)
        y = 1.5 * t / 10 - 2 + 3 * np.sin(t / 5)
        z = height_i + (d_height / t[-1]) * t + 4 * np.sin(0.3 * t)

    elif trajectory == 2 and (x, y, z == None, None, None):
        x = (r / 10 * t + 2) * np.cos(alpha + t / 5)
        y = (r / 10 * t + 2) * np.sin(alpha + t / 5)
        z = height_i + (d_height / t[-1]) * t * np.sin(t / 5)

    elif trajectory == 3 and (x, y, z == None, None, None):
        x = 2 * t / 20 + 1 + np.cos(t / 2)
        y = 2 * t / 20 - 2 + np.sin(t / 2)
        z = height_i + (d_height / t[-1]) * t + 10 * np.sin(0.3 * t)

    elif trajectory == 4 and (x, y, z == None, None, None):
        x = -4 * t / 20 + 1 + np.cos(t / 4)
        y = 2 * t / 20 - 2 + np.sin(t / 4)
        z = height_i + (d_height / t[-1]) * t + 5 * np.sin(0.3 * t)
    # else:
    #     print("Using custom trajectory")

    # Compute derivatives using the same finite difference approach as MATLAB
    # The MATLAB code:
    # dx=[x(2)-x(1),x(2:end)-x(1:end-1)];
    # In Python, we can use np.diff. Note that MATLAB indexing starts at 1, Python at 0.
    # We'll replicate the logic directly:
    dx = np.zeros_like(x)
    dx[0] = x[1] - x[0]
    dx[1:] = x[1:] - x[:-1]

    dy = np.zeros_like(y)
    dy[0] = y[1] - y[0]
    dy[1:] = y[1:] - y[:-1]

    dz = np.zeros_like(z)
    dz[0] = z[1] - z[0]
    dz[1:] = z[1:] - z[:-1]

    # Compute velocities
    # x_dot = round(dx*(1/(Ts*innerDyn_length)),8);
    factor = 1/(Ts*innerDyn_length)
    x_dot = np.round(dx * factor, 8)
    y_dot = np.round(dy * factor, 8)
    z_dot = np.round(dz * factor, 8)

    # Compute accelerations similarly
    ddx = np.zeros_like(x_dot)
    ddx[0] = x_dot[1] - x_dot[0]
    ddx[1:] = x_dot[1:] - x_dot[:-1]

    ddy = np.zeros_like(y_dot)
    ddy[0] = y_dot[1] - y_dot[0]
    ddy[1:] = y_dot[1:] - y_dot[:-1]

    ddz = np.zeros_like(z_dot)
    ddz[0] = z_dot[1] - z_dot[0]
    ddz[1:] = z_dot[1:] - z_dot[:-1]

    x_dot_dot = np.round(ddx * factor, 8)
    y_dot_dot = np.round(ddy * factor, 8)
    z_dot_dot = np.round(ddz * factor, 8)

    # Compute psi
    # psi(1)=atan2(y(1),x(1))+pi/2;
    psi = np.zeros_like(x)
    psi[0] = np.arctan2(y[0], x[0]) + np.pi/2
    # psi(2:end)=atan2(dy(2:end),dx(2:end));
    # dy and dx at index 1 represent second element (MATLAB indexing)
    # We'll be careful. In MATLAB: dy(2:end), dx(2:end) means from second difference forward.
    # Here, since dy and dx are same length and represent differences, we can do:
    # psi(2:end)=atan2(dy(2:end),dx(2:end));
    # In Python indexing, that's psi[1:]=atan2(dy[1:], dx[1:])
    psi[1:] = np.arctan2(dy[1:], dx[1:])

    # Make psi continuous as done in MATLAB
    psiInt = np.zeros_like(psi)
    psiInt[0] = psi[0]
    dpsi = psi[1:] - psi[:-1]

    for i in range(1, len(psiInt)):
        if dpsi[i-1] < -np.pi:
            psiInt[i] = psiInt[i-1] + (dpsi[i-1] + 2*np.pi)
        elif dpsi[i-1] > np.pi:
            psiInt[i] = psiInt[i-1] + (dpsi[i-1] - 2*np.pi)
        else:
            psiInt[i] = psiInt[i-1] + dpsi[i-1]

    psiInt = np.round(psiInt, 8)

    X_ref = np.column_stack((t, x))
    X_dot_ref = np.column_stack((t, x_dot))
    X_dot_dot_ref = np.column_stack((t, x_dot_dot))

    Y_ref = np.column_stack((t, y))
    Y_dot_ref = np.column_stack((t, y_dot))
    Y_dot_dot_ref = np.column_stack((t, y_dot_dot))

    Z_ref = np.column_stack((t, z))
    Z_dot_ref = np.column_stack((t, z_dot))
    Z_dot_dot_ref = np.column_stack((t, z_dot_dot))

    psi_ref = np.column_stack((t, psiInt))

    return np.array([X_ref, X_dot_ref, X_dot_dot_ref, 
            Y_ref, Y_dot_ref, Y_dot_dot_ref,
            Z_ref, Z_dot_ref, Z_dot_dot_ref, psi_ref])


def plot_ref_trajectory(constants, t, ax, r=2, f=0.025, height_i=2, height_f=5,
                        x=None, y=None, z=None
):
    """
    Plots the reference trajectory on the given 3D axes.

    Parameters:
    - ax: A Matplotlib 3D axis object to plot on. If None, a new figure and axis are created.

    Returns:
    - ax: The Matplotlib 3D axis object used for the plot.
    """
    # Call trajectory_reference once with the whole time_points array
    (X_ref, X_dot_ref, X_dot_dot_ref, 
     Y_ref, Y_dot_ref, Y_dot_dot_ref,
     Z_ref, Z_dot_ref, Z_dot_dot_ref, 
     psi_ref) = traj_ref = trajectory_reference(
                constants, t, x=x, y=y, z=z)

    # Extract trajectory data
    x_vals = X_ref[:,1]
    y_vals = Y_ref[:,1]
    z_vals = Z_ref[:,1]

    # Plot the trajectory
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(x_vals, y_vals, z_vals, label="Reference Trajectory", color="blue")
    

def plot_ref_trajectory_with_arrows(t, r, f, height_i, height_f, 
                                    constants, num_arrows=100, c_arw=20):
    # Call trajectory_reference once with the whole time_points array
    (X_ref, X_dot_ref, X_dot_dot_ref, 
     Y_ref, Y_dot_ref, Y_dot_dot_ref,
     Z_ref, Z_dot_ref, Z_dot_dot_ref, 
     psi_ref) = trajectory_reference(constants, t)

    # Extract trajectory data
    x_vals = X_ref[:,1]
    y_vals = Y_ref[:,1]
    z_vals = Z_ref[:,1]
    x_dot_vals = X_dot_ref[:,1]
    y_dot_vals = Y_dot_ref[:,1]
    z_dot_vals = Z_dot_ref[:,1]

    # Plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals, y_vals, z_vals, label="Reference Trajectory", color="blue")

    # Add arrows at intervals
    arrow_indices = np.linspace(0, len(t) - 1, num_arrows, dtype=int)
    for i, idx in enumerate(arrow_indices):
        if i == 0:
            label = "Velocities"
        else:
            # Use a special label to avoid multiple entries in the legend
            label = "_nolegend_"
            
        ax.quiver(
            x_vals[idx], y_vals[idx], z_vals[idx],  # Starting point
            c_arw * x_dot_vals[idx], c_arw * y_dot_vals[idx], c_arw * z_dot_vals[idx],  # Velocity components
            length=1.0, normalize=False, color="red", linewidth=1.0, label=label
        )

    ax.set_title("3D Trajectory with Velocity Arrows")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    constants = Quad_Constants()
    
    tf = 100
    innerDyn_length = constants.innerDyn_length
    hz = constants.hz
    Ts = constants.Ts

    # Generate a time array for the entire simulation
    t0 = 0.0
    t = np.linspace(t0, tf, int(tf / Ts) * innerDyn_length)

    r = 2
    f = 0.025
    height_i = 2
    height_f = 5
    
    # Now call the plotting function, passing the entire time array `t`
    plot_ref_trajectory_with_arrows(t, r, f, height_i, height_f, constants)
