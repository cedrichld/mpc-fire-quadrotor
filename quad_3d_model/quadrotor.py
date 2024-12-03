import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos, sqrt
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_continuous_are

from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables

# Quadrotor Configuration:
# 
#      1:CW    2:CCW
#         \     /
#          \___/
#          /   \
#         /     \
#      4:CCW    3:CW
#

class Quadrotor(object):
  def __init__(self, Q, R, Qf):
    # Parameters
    self.m = 0.468  # Mass of the quadrotor (kg)
    self.g = 9.81   # Gravitational acceleration (m/s^2)
    self.I = np.diag([4.856e-3, 4.856e-3, 8.801e-3])  # Inertia tensor (kg*m^2)
    self.L = 0.225      # Arm length (m)
    self.K = 2.980e-6   # Thrust coefficient (N/(rad/s)^2)
    self.b = 1.14e-7    # Drag coefficient (N*m/(rad/s)^2)
    self.Ax = 0.25 * 0  # Aerodynamic drag in x (NOT USED YET)
    self.Ay = 0.25 * 0  # Aerodynamic drag in y (NOT USED YET)
    self.Az = 0.25 * 0  # Aerodynamic drag in z (NOT USED YET)
    
    # Control-related parameters
    self.Q = Q       # State cost matrix
    self.R = R       # Input cost matrix
    self.Qf = Qf     # Terminal cost matrix

    # State and control dimensions
    self.n_zeta = 12  # State dimension (x, y, z, phi, theta, psi + their velocities)
    self.n_u = 4      # Input dimension (4 rotors)
    
    # Input limits
    self.umin = 0.0
    self.umax = 0.5 * 1.075 * sqrt(1 / self.K)

  def zeta_d(self):
    # Nominal state
    return np.zeros(12)  # Hovering at the origin

  def u_d(self):
    # Nominal input (hover condition)
    return (1.075 * sqrt(1 / self.K)) * np.ones(self.n_u) # self.m * self.g / (self.n_u)

  def continuous_time_full_dynamics(self, zeta, u):
    # Unpack parameters
    m, g, L = self.m, self.g, self.L  # Mass, gravity, arm length
    I_x, I_y, I_z = self.I[0, 0], self.I[1, 1], self.I[2, 2]  # Inertia tensor components
    K, b = self.K, self.b  # Thrust and drag coefficients
    Ax, Ay, Az = self.Ax, self.Ay, self.Az

    # Extract state variables
    X, Y, Z, phi, theta, psi, X_dot, Y_dot, Z_dot, phi_dot, theta_dot, psi_dot = zeta
    
    R = self.rotation_matrix(phi, theta, psi) # Rotation matrix from body to world

    # Extract rotor speeds
    omega1, omega2, omega3, omega4 = u

    # Compute thrust and torques
    T = K * (omega1**2 + omega2**2 + omega3**2 + omega4**2)  # Total thrust
    tau_phi = K * L * (omega4**2 - omega2**2)  # Roll torque
    tau_theta = K * L * (omega3**2 - omega1**2)  # Pitch torque
    tau_psi = b * (omega1**2 - omega2**2 + omega3**2 - omega4**2)  # Yaw torque

    # Translational accelerations
    acc_body = np.array([0, 0, T / m])  # Acceleration in body frame
    acc_world = R @ acc_body - np.array([0, 0, g])  # Transform to world frame and add gravity
    X_ddot, Y_ddot, Z_ddot = acc_world

    # Rotational accelerations
    omega_dot = np.linalg.inv(np.diag([I_x, I_y, I_z])) @ np.array([
        tau_phi + (I_y - I_z) * theta_dot * psi_dot,
        tau_theta + (I_z - I_x) * phi_dot * psi_dot,
        tau_psi + (I_x - I_y) * phi_dot * theta_dot
    ])
    phi_ddot, theta_ddot, psi_ddot = omega_dot

    # Pack derivatives into state derivative vector
    zeta_dot = np.zeros(12)
    zeta_dot[0] = X_dot
    zeta_dot[1] = Y_dot
    zeta_dot[2] = Z_dot
    
    zeta_dot[3] = phi_dot
    zeta_dot[4] = theta_dot
    zeta_dot[5] = psi_dot
    
    zeta_dot[6] = X_ddot
    zeta_dot[7] = Y_ddot
    zeta_dot[8] = Z_ddot
    
    zeta_dot[9] = phi_ddot 
    zeta_dot[10] = theta_ddot
    zeta_dot[11] = psi_ddot

    return zeta_dot
  
  '''
    
  def continuous_time_full_dynamics(self, zeta, u):
    # Extract parameters
    m, I_x, I_y, I_z = self.m, self.I[0, 0], self.I[1, 1], self.I[2, 2]
    g = self.g

    # Construct A matrix
    A = np.array([
        [m, 0, 0, 0, 0, 0],
        [0, m, 0, 0, 0, 0],
        [0, 0, m, 0, 0, 0],
        [0, 0, 0, I_x, 0, 0],
        [0, 0, 0, 0, I_y, 0],
        [0, 0, 0, 0, 0, I_z]
    ])

    # Construct B vector based on forces and torques
    T = self.K * np.sum(u**2)
    tau_phi = self.K * self.L * (u[3]**2 - u[1]**2)
    tau_theta = self.K * self.L * (u[2]**2 - u[0]**2)
    tau_psi = self.b * (u[0]**2 - u[1]**2 + u[2]**2 - u[3]**2)
    B = np.array([
        0,
        0,
        T - m * g,
        tau_phi,
        tau_theta,
        tau_psi
    ])

    # Solve for accelerations
    accelerations = self.solve_quadrotor_dynamics(A, B)

    return accelerations '''
  
  def solve_quadrotor_dynamics(self, A, B):
    try:
        # Solve the linear system A * X = B
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to solve the system. Matrix A might be singular or ill-conditioned: {e}")
    
    return X

  def continuous_time_linearized_dynamics(self):
    """
    Computes the linearized dynamics at the hover point using numerical values.
    """
    # Unpack parameters
    m, I, g, L, K, b = self.m, self.I, self.g, self.L, self.K, self.b
    I_x, I_y, I_z = I[0, 0], I[1, 1], I[2, 2]

    # Initialize A and B matrices
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))

    # Fill A matrix
    # Velocity-to-position coupling
    A[0, 6] = 1
    A[1, 7] = 1
    A[2, 8] = 1
    A[3, 9] = 1
    A[4, 10] = 1
    A[5, 11] = 1

    # Gravity coupling
    A[6, 4] = g  # g * theta for x dynamics
    A[7, 3] = -g  # -g * phi for y dynamics

    # Rotational inertia coupling
    A[9, 11] = (I_y - I_z) / I_x
    A[10, 9] = (I_z - I_x) / I_y

    # Fill B matrix
    # Translational dynamics (thrust inputs)
    B[8, :] = [K * sqrt(g * m / K) / m,
               K * sqrt(g * m / K) / m,
               K * sqrt(g * m / K) / m,
               K * sqrt(g * m / K) / m]

    # Rotational dynamics (torques from inputs)
    B[9, :] =  [0,
                - K * L * sqrt(g * m / K) / I_x, 
                0, 
                K * L * sqrt(g * m / K) / I_x]
    B[10, :] = [- K * L * sqrt(g * m / K) / I_y, 
                0, 
                K * L * sqrt(g * m / K) / I_y, 
                0]
    B[11, :] = [b * sqrt(g * m / K) / I_z,
                - b * sqrt(g * m / K) / I_z, 
                b * sqrt(g * m / K) / I_z,
                - b * sqrt(g * m / K) / I_z]

    return A, B


  '''  
  def continuous_time_linearized_dynamics(self):
    """
    Computes the linearized dynamics at the hover point using numerical values.
    """
    # Unpack parameters
    m, I, g, L = self.m, self.I, self.g, self.L
    I_x, I_y, I_z = I[0, 0], I[1, 1], I[2, 2]

    # Initialize A and B matrices
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))

    # Fill A matrix
    # Velocity-to-position coupling
    A[0, 6] = 1
    A[1, 7] = 1
    A[2, 8] = 1
    A[3, 9] = 1
    A[4, 10] = 1
    A[5, 11] = 1

    # Gravity coupling
    A[6, 4] = g  # g * theta for x dynamics
    A[7, 3] = -g  # -g * phi for y dynamics

    # Rotational inertia coupling
    A[9, 11] = (I_y - I_z) / I_x
    A[10, 9] = (I_z - I_x) / I_y

    # Fill B matrix
    # Translational dynamics (thrust inputs)
    B[8, :] = [1 / m, 1 / m, 1 / m, 1 / m]

    # Rotational dynamics (torques from inputs)
    B[9, :] =  [-1 / I_x,
                (L / I_x) - (1 / I_x), 
                -1 / I_x, 
                -(L / I_x) - (1 / I_x)]
    B[10, :] = [-(L / I_y) + (1 / I_y), 
                1 / I_y, 
                (L / I_y) + (1 / I_y), 
                1 / I_y]
    B[11, :] = [1 / I_z,
                -1 / I_z, 
                1 / I_z,
                -1 / I_z]

    return A, B '''
  
  def discrete_time_linearized_dynamics(self, T):
    # Discrete time version of the linearized dynamics at the fixed point
    # This function returns A and B matrix of the discrete time dynamics
    A_c, B_c = self.continuous_time_linearized_dynamics()
    A_d = np.identity(12) + A_c * T
    B_d = B_c * T

    return A_d, B_d

  def add_initial_state_constraint(self, prog, zeta, zeta_current):
    # Impose initial state constraint.
    prog.AddBoundingBoxConstraint(zeta_current, zeta_current, zeta[0])
    

  def add_input_saturation_constraint(self, prog, u, N):
    # Impose input limit constraint.
    for i in range(N - 1):
        prog.AddBoundingBoxConstraint(self.umin - self.u_d(), self.umax - self.u_d(), u[i])

  def add_dynamics_constraint(self, prog, zeta, u, N, T):
    """
    Adds dynamics constraints to the MPC optimization problem.

    Parameters:
    prog: MathematicalProgram
        The optimization problem.
    zeta: ndarray
        State trajectory decision variables.
    u: ndarray
        Input trajectory decision variables.
    N: int
        Prediction horizon.
    T: float
        Time step for discretization.
    """
    # Get linearized discrete-time dynamics
    A_d, B_d = self.discrete_time_linearized_dynamics(T)

    # Add constraints for each time step
    for i in range(N - 1):
      dynamics_constraint = (
        zeta[i + 1] - A_d @ zeta[i] - B_d @ u[i]
      )
      prog.AddLinearEqualityConstraint(
        dynamics_constraint, np.zeros_like(zeta[0])
      )

  def add_cost(self, prog, zeta, u, N):
    cost = 0
    for i in range(N - 1):
      cost += (zeta[i] - self.zeta_d()).T @ self.Q @ (zeta[i] - self.zeta_d()) + u[i].T @ self.R @ u[i]
    
    # cost += zeta[N - 1].T @ self.Qf @ zeta[N - 1]
    prog.AddQuadraticCost(cost)

  def compute_mpc_feedback(self, zeta_current, use_clf=False):
    '''
    This function computes the MPC controller input u
    '''

    # Parameters for the QP
    N = 10 # Prediction Horizon
    T = 0.1 # Timestep

    # Initialize mathematical program and decalre decision variables
    prog = MathematicalProgram()

    zeta = np.zeros((N, self.n_zeta), dtype="object")
    for i in range(N):
      zeta[i] = prog.NewContinuousVariables(self.n_zeta, "z_" + str(i))
    u = np.zeros((N-1, self.n_u), dtype="object")
    for i in range(N-1):
      u[i] = prog.NewContinuousVariables(self.n_u, "u_" + str(i))

    # Add constraints and cost
    self.add_initial_state_constraint(prog, zeta, zeta_current)
    self.add_input_saturation_constraint(prog, u, N)
    self.add_dynamics_constraint(prog, zeta, u, N, T)
    self.add_cost(prog, zeta, u, N)

    # Solve the QP
    solver = OsqpSolver()
    result = solver.Solve(prog)

    if result.is_success():
        u_mpc = result.GetSolution(u[0]) + self.u_d()
        # print(f"Control input at this step: {u_mpc}") 
    else:
        print(f"Solver failed to find a solution at {zeta_current}.")
        u_mpc = np.zeros(self.n_u) 
        
    return u_mpc
  
  def rotation_matrix(self, phi, theta, psi):
    # Compute rotation matrix from body to world
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x  # Combined rotation matrix
    
    return R
  
  def get_motor_positions(self, state):
    X, Y, Z, phi, theta, psi = state[:6]
    # R = self.rotation_matrix(phi, theta, psi)
    motor_offsets = np.array([
        [self.L / 2, 0, 0],  # Motor 1
        [0, self.L / 2, 0],  # Motor 2
        [-self.L / 2, 0, 0],  # Motor 3
        [0, -self.L / 2, 0]  # Motor 4
    ]).T
    return motor_offsets + np.array([[X], [Y], [Z]])