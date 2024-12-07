import numpy as np
from scipy.integrate import solve_ivp
from quadrotor import Quadrotor
from create_animation import animate_quadrotor
import matplotlib.pyplot as plt

def simulate_quadrotor_3d(zeta0, tf, quadrotor, use_mpc=True, use_mpc_with_clf=False, use_clf_qp=False):
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

  while t[-1] < tf:
    current_time = t[-1]
    current_zeta = zeta[-1]
    
    if not np.all(np.isfinite(current_zeta)):
      raise ValueError(f"State vector contains invalid values: {current_zeta}")
    
    if not (index % 5.0):
      print(f"Current_zeta: {np.round(current_zeta, decimals=2)}")
    index += 1

    # Compute remaining time to avoid overshooting
    remaining_time = tf - current_time
    dt = min(dt, remaining_time)  # Adjust dt dynamically

    # print(f"Current time: {current_time}, Remaining time: {remaining_time}, Step size: {dt}")
    # Compute control input
    if use_mpc:
      current_omega_command = quadrotor.compute_mpc_feedback(current_zeta, use_mpc_with_clf)
    elif use_clf_qp:
      current_omega_command = quadrotor.compute_clf_qp_feedback(current_zeta)
    else:
      current_omega_command = quadrotor.compute_lqr_feedback(current_zeta)

    # if (i == False for i in current_omega_command):
    #   print(f"Breaking the MPC Loop. i: {i for i in current_omega_command}")
    #   break
    # Apply input limits
    current_omega_real = np.clip(current_omega_command, 0, np.inf)

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

  plt.figure()
  plt.plot(t[1:], omega[1:])
  plt.xlabel("Time (s)")
  plt.ylabel("Control Inputs (N)")
  plt.title(f"Control Inputs ({name})")
  plt.legend([f"omega{i+1}" for i in range(omega.shape[1])])
  # plt.show()
  plt.savefig(f"{name}_inputs.png")
  print(f"Saved input plot as {name}_inputs.png")

if __name__ == '__main__':
  # Define system parameters
  # R = np.eye(4)
  Q = np.diag([10, 10, 1000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
  R = np.diag([0.05, 1, 1, 1])
  # Q = np.eye(12) * 0.1
  Qf = Q

  quadrotor = Quadrotor(Q, R, Qf)

  # Initial state (position, orientation, velocities)
  zeta0 = np.array([0.75, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Hovering at 0.5m

  # Simulation duration
  tf = 10.0

  # Simulate the quadrotor
  zeta, omega, t = simulate_quadrotor_3d(zeta0, tf, quadrotor)

  # Plot results
  plot_results_3d(zeta, omega, t, "3D Quadrotor")

  # Animate the quadrotor
  animate_quadrotor(zeta, t, Q, R, Qf)