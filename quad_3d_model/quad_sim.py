import numpy as np
from scipy.integrate import solve_ivp
from quadrotor import Quadrotor
from create_animation import Animation
import matplotlib.pyplot as plt

def simulate_quadrotor_3d(
  zeta0, tf, quadrotor, use_mpc_with_clf=False, use_integral_action=False, use_fb_lin=True
  ):
  """
  Simulates the stabilized maneuver of a 3D quadrotor system.

  Parameters:
  zeta0 : ndarray
      Initial state vector (12-dimensional for 3D system).
  tf : float
      Simulation end time.
  quadrotor : Quadrotor object
      Quadrotor object containing the dynamics and controllers.
  """
  t0 = 0.0
  dt = 0.01  # Time step for integration

  zeta = [zeta0]
  omega = [np.zeros(quadrotor.n_u)]
  t = [t0]
  index = 0
  print_U = False
  
  while t[-1] < tf:
    current_time = t[-1]
    current_zeta = zeta[-1]
    
    if not np.all(np.isfinite(current_zeta)):
      raise ValueError(f"State vector contains invalid values: {current_zeta}")
    
    print_U = False
    if not (index % 10.0):
      print(f"Current_zeta: {np.round(current_zeta, decimals=2)}")
      print_U = True
    index += 1

    # Compute remaining time to avoid overshooting
    remaining_time = tf - current_time
    dt = min(dt, remaining_time)  # Adjust dt dynamically

    # Compute control input
    current_omega_command = quadrotor.compute_mpc_feedback(
      current_zeta, print_U=print_U, use_clf=use_mpc_with_clf, 
      use_integral_action=use_integral_action, use_fb_lin=use_fb_lin
    )

    # Apply input limits
    current_omega_real = np.clip(current_omega_command, -np.inf, np.inf)

    # Define ODE for current state
    def f(t, zeta):
      return quadrotor.continuous_time_full_dynamics(current_zeta, current_omega_real)

    # Integrate one time step
    sol = solve_ivp(f, (0, dt), current_zeta, first_step=dt)

    # Record results
    t.append(t[-1] + dt)
    zeta.append(sol.y[:, -1])
    omega.append(current_omega_command)
    
  zeta = np.array(zeta)
  omega = np.array(omega)
  t = np.array(t)

  return zeta, omega, t

def plot_results_3d(zeta, omega, t, name):
  """
  Plots the trajectory and control inputs of the 3D quadrotor.
  """
  # trajectory
  plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot3D(zeta[:, 0], zeta[:, 1], zeta[:, 2], label="Trajectory")
  ax.scatter(zeta[0, 0], zeta[0, 1], zeta[0, 2], color='red', label="Start")
  ax.scatter(zeta[-1, 0], zeta[-1, 1], zeta[-1, 2], color='green', label="End")
  ax.set_xlabel("X (m)")
  ax.set_ylabel("Y (m)")
  ax.set_zlabel("Z (m)")
  ax.legend()
  ax.set_title(f"3D Trajectory ({name})")
  plt.savefig(f"{name}_trajectory.png") 
  print(f"Saved trajectory plot as {name}_trajectory.png")
  
  # Retrieve limits
  x_limits = ax.get_xlim()
  y_limits = ax.get_ylim()
  z_limits = ax.get_zlim()

  # Angles
  plt.figure()
  plt.plot(t[1:], zeta[1:, 3], label="Phi")
  plt.plot(t[1:], zeta[1:, 4], label="Theta")
  plt.plot(t[1:], zeta[1:, 5], label="Psi")
  plt.xlabel("Time (s)")
  plt.ylabel("Angles (rad)")
  plt.title(f"Angles over time ({name})")
  plt.legend()
  # plt.show()
  plt.savefig(f"{name}_angles.png")
  print(f"Saved angles plot as {name}_angles.png")
  
  plt.figure()
  plt.plot(t[1:], omega[1:])
  plt.xlabel("Time (s)")
  plt.ylabel("Control Inputs (rad/s)")
  plt.title(f"Control Inputs ({name})")
  plt.legend([f"omega{i+1}" for i in range(omega.shape[1])])
  # plt.show()
  plt.savefig(f"{name}_inputs.png")
  print(f"Saved input plot as {name}_inputs.png")
  
  return x_limits, y_limits, z_limits

if __name__ == '__main__':
  # Define system parameters
  # R = np.eye(4)
  Q = np.diag([10, 10, 100, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
  R = np.diag([0.1, 10, 100, 10])
  # Q = np.eye(12) * 0.1
  Qf = Q

  quadrotor = Quadrotor(Q, R, Qf)

  # Initial state (position, orientation, velocities)
  zeta0 = np.array([0.75, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Hovering at

  # Simulation duration
  tf = 5.0

  # Simulate the quadrotor
  zeta, omega, t = simulate_quadrotor_3d(zeta0, tf, quadrotor)

  # Plot results
  x_limits, y_limits, z_limits = plot_results_3d(zeta, omega, t, "3D Quadrotor")

  # Animate the quadrotor
  Animation().animate_quadrotor(zeta, t, Q, R, Qf, x_limits, y_limits, z_limits)