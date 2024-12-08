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
from scipy.linalg import solve_discrete_lyapunov

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
    self.m = 0.698  # quadrotor mass (kg)
    self.g = 9.81   # g (m/s^2)
    self.I = np.diag([3.4e-3, 3.4e-3, 6.0e-3])  # inertia tensor (kg*m^2)
    self.L = 0.171 # arm length (m)
    self.Jtp = 1.302e-6 # Rotational moment of inertia (N*m*s^2 = kg*m^2)
    self.ct = (7.6184e-8) * (60 / (2 * np.pi))**2; # Thrust Coef: N*s^2
    self.cq = (2.6839e-9) * (60 / (2 * np.pi))**2; # Torque/Drag Coef: N*m^2
    
    # Control-related parameters
    self.Q = Q       # State cost matrix
    self.R = R       # Input cost matrix
    self.Qf = Qf     # Terminal cost matrix
    self.Q_integral = Q  # Penalize integral error less than state error
    
    
    self.P_clf = None; # Matrix PP for Lyapunov Function (changed discretely)
    self.alpha_clf = 0.5 # Lyapunov Decay Rate

    # State and control dimensions
    self.n_zeta = 12  # State dimension (x, y, z, phi, theta, psi + their velocities)
    self.n_u = 4      # Input dimension (4 rotors)
    
    # Input limits
    # self.omega_min = self.omega_ref * 0.5 # no motor input
    # self.omega_max = self.omega_ref * 2 # max rotation speed of 3 times required speed
    self.U1_min, self.U1_max = - (self.m * self.g * 0.25), (self.m * self.g * 3)
    self.U2_min, self.U2_max = - math.inf, math.inf
    self.U3_min, self.U3_max = - math.inf, math.inf
    self.U4_min, self.U4_max = - math.inf, math.inf
    
    self.U_min = np.array([self.U1_min, self.U2_min, self.U3_min, self.U4_min])
    self.U_max = np.array([self.U1_max, self.U2_max, self.U3_max, self.U4_max])
    
    # M matrix from U to omega^2
    self.M = np.array([
        [self.ct, self.ct, self.ct, self.ct],
        [0, self.ct * self.L, 0, -self.ct  *self.L],
        [-self.ct * self.L, 0,    self.ct * self.L, 0],
        [-self.cq, self.cq, -self.cq, self.cq]
    ])
    
    self.M_inv = np.linalg.inv(self.M)
    
    # X^T = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]
    # U^T = [F_z, tau_phi, tau_theta, tau_psi] or [U1, U2, U3, U4]
    
    
  def zeta_d(self):
    # Nominal state
    zeta = np.zeros(12)
    zeta[0] = 0.
    zeta[2] = 0.
    # zeta = np.array([0.09, 0., 12.14, -0., 0., 0., -0.02, -0., 0.07, 0., -0., -0.])
    return zeta # Hovering at the origin

  def U_d(self):
    # Nominal input (hover condition)
    # print(f"u_d is {self.omega_ref * np.ones(self.n_u)}")
    return np.array([self.m * self.g, 0, 0, 0])
  
  def U_to_omega(self, U):
    """
    Given U, solve for omega^2
    """
    
    omega_sqr = self.M_inv @ U
    
    # Not physically feasible
    if np.any(omega_sqr < 0):
      omega_sqr = np.clip(omega_sqr, 0, None)
    omega = np.sqrt(omega_sqr)
    
    return omega
  
  def U_calculator(self, omega):
    U1 = self.ct * (omega[0]**2 + omega[1]**2 + omega[2]**2 + omega[3]**2)
    U2 = self.ct * self.L * (omega[1]**2 - omega[3]**2)
    U3 = self.ct * self.L * (omega[2]**2 - omega[0]**2)
    U4 = self.cq * (-omega[0]**2 + omega[1]**2 - omega[2]**2 + omega[3]**2)
    omega_total = omega[0] - omega[1] + omega[2] - omega[3]
    
    # print(f"U1, U2, U3, U4, omega_total are {U1, U2, U3, U4, omega_total}")
    
    return U1, U2, U3, U4, omega_total

  def continuous_time_full_dynamics(self, zeta, omega):
    '''
    Input: Takes in the current zeta and omega
    Output: Returns the current dzeta
    '''
    # Constants
    m, g, Jtp = self.m, self.g, self.Jtp  # Mass, gravity, torque precession constant
    Ix, Iy, Iz = self.I[0, 0], self.I[1, 1], self.I[2, 2]  # Moments of inertia

    # State: [x, y, z, phi, theta, psi, u, v, w, p, q, r]
    # x, y, z = zeta[0:3]
    phi, theta, psi = zeta[3:6] # Euler angles
    u, v, w = zeta[6:9]  # Linear velocities
    p, q, r = zeta[9:12] # Angular velocities
    
    # Input forces and torques
    U1, U2, U3, U4, omega_total = self.U_calculator(omega)
    # print(f"U1, U2, U3, U4, omega_total: {U1, U2, U3, U4, omega_total}")

    # Rotation matrix relating body frame velocities to inertial frame velocities
    # Transformation matrix relating angular velocities to Euler angle derivatives
    R_matrix, T_matrix = self.R_T_matrices(phi, theta, psi)

    # Compute nonlinear dynamics
    dzeta = np.zeros(12)
    
    # Position derivatives
    dzeta[0:3] = R_matrix @ np.array([u, v, w])  # [x_dot, y_dot, z_dot]
    
    # Euler angle derivatives
    euler_dots = T_matrix @ np.array([p, q, r])
    dzeta[3] = euler_dots[0]  # phi_dot
    dzeta[4] = euler_dots[1]  # theta_dot
    dzeta[5] = euler_dots[2]  # psi_dot

    # Translational dynamics
    dzeta[6] = (v * r - w * q) + g * np.sin(theta)  # u_dot
    dzeta[7] = (w * p - u * r) - g * np.cos(theta) * np.sin(phi)  # v_dot
    dzeta[8] = (u * q - v * p) - g * np.cos(theta) * np.cos(phi) + U1 / m  # w_dot

    # Rotational dynamics
    dzeta[9] = (q * r * (Iy - Iz) - Jtp * q * omega_total + U2) / Ix  # p_dot
    dzeta[10] = (p * r * (Iz - Ix) + Jtp * p * omega_total + U3) / Iy  # q_dot
    dzeta[11] = (p * q * (Ix - Iy) + U4) / Iz  # r_dot

    return dzeta

  def continuous_time_linearized_dynamics(self):
    """
    Computes the linearized dynamics at the hover point using the hover angular velocity (omega_total)
    and the system parameters.
    
    Parameters:
    - omega_total: Total rotor angular velocity at hover.

    Returns:
    - A: Linearized state matrix (12x12).
    - B: Linearized control matrix (12x4).
    """
    # Unpack parameters
    g = self.g  # Gravity
    m = self.m  # Mass
    Jtp = self.Jtp  # Torque precession constant
    I_x, I_y, I_z = self.I[0, 0], self.I[1, 1], self.I[2, 2]  # Inertia tensor elements
    # _, _, _, _, omega_total = self.U_calculator(omega_current)
    # print(f"omega: {omega}, omega_total: {omega_total}")


    # Initialize A and B matrices
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))

    # Fill A matrix
    # Velocity-to-position coupling
    A[0, 6] = 1  # x_dot coupling with u (x velocity)
    A[1, 7] = 1  # y_dot coupling with v (y velocity)
    A[2, 8] = 1  # z_dot coupling with w (z velocity)
    A[3, 9] = 1  # phi_dot coupling with p (roll rate)
    A[4, 10] = 1  # theta_dot coupling with q (pitch rate)
    A[5, 11] = 1  # psi_dot coupling with r (yaw rate)

    # Gravity coupling
    A[6, 4] = g  # g * theta for x dynamics
    A[7, 3] = -g  # -g * phi for y dynamics
    
    # Translational dynamics (thrust inputs)
    B[8, 0] = 1 / m
    B[9, 1] = 1 / I_x
    B[10, 2] = 1 / I_y
    B[11, 3] = 1 / I_z

    return A, B

  def discrete_time_linearized_dynamics(self, T):
    # Discrete time version of the linearized dynamics at the fixed point
    # This function returns A and B matrix of the discrete time dynamics
    A_c, B_c = self.continuous_time_linearized_dynamics()
    A_d = np.identity(12) + A_c * T
    B_d = B_c * T

    return A_d, B_d

  def feedback_linearization(self, zeta_current):
    """
    Feedback linearization

    Parameters:
    - zeta_current: Current state vector [u, v, w, x, y, z, phi, theta, psi].

    Returns:
    - phi_ref: Desired roll angle.
    - theta_ref: Desired pitch angle.
    - U1: Desired thrust.
    """
    # Extract constants
    m = self.m
    g = self.g
    px = [-1.0, -2.0]
    py = [-1.0, -2.0]
    pz = [-1.0, -2.0]

    # Extract current states
    x, y, z, phi, theta, psi, u, v, w,_,_,_ = zeta_current

    # Extract desired references from self.zeta_d()
    zeta_ref = self.zeta_d()
    x_ref, y_ref, z_ref = zeta_ref[:3]
    phi_ref_d, theta_ref_d, psi_ref = zeta_ref[3:6]
    x_dot_ref, y_dot_ref, z_dot_ref = zeta_ref[6:9]
    x_dot_dot_ref, y_dot_dot_ref, z_dot_dot_ref = 0, 0, 0  # Assume no desired accelerations

    # Rotational matrix
    R_matrix, _ = self.R_T_matrices(phi, theta, psi)
    x_dot, y_dot, z_dot = R_matrix @ np.array([u, v, w])
    
    # Derive x_dot_dot based on previous state or equations of motion
    x_dot_dot = -px[0] * (x_ref - x) - px[1] * (x_dot_ref - x_dot)
    

    # Position errors
    ex, ex_dot = x_ref - x, x_dot_ref - x_dot
    ey, ey_dot = y_ref - y, y_dot_ref - y_dot
    ez, ez_dot = z_ref - z, z_dot_ref - z_dot
    
    # Compute Kp, Kd gains for stabilization
    kx1 = (px[0] - (px[0] + px[1]) / 2) ** 2 - (px[0] + px[1]) ** 2 / 4
    kx2 = px[0] + px[1]
    ky1 = (py[0] - (py[0] + py[1]) / 2) ** 2 - (py[0] + py[1]) ** 2 / 4
    ky2 = py[0] + py[1]
    kz1 = (pz[0] - (pz[0] + pz[1]) / 2) ** 2 - (pz[0] + pz[1]) ** 2 / 4
    kz2 = pz[0] + pz[1]
    
    # Compute desired accelerations
    x_dot_dot_ref = -kx1 * ex - kx2 * ex_dot
    y_dot_dot_ref = -ky1 * ey - ky2 * ey_dot
    z_dot_dot_ref = -kz1 * ez - kz2 * ez_dot

    # Compute position control inputs
    ux = kx1 * ex + kx2 * ex_dot
    uy = ky1 * ey + ky2 * ey_dot
    uz = kz1 * ez + kz2 * ez_dot

    # Virtual accelerations
    vx = x_dot_dot_ref - ux
    vy = y_dot_dot_ref - uy
    vz = z_dot_dot_ref - uz

    # Compute phi, theta, U1
    a = vx / (vz + g)  # Add small epsilon to avoid division by zero
    b = vy / (vz + g)
    c = np.cos(psi_ref)
    d = np.sin(psi_ref)
    tan_theta = a * c + b * d
    theta_ref = np.arctan(tan_theta)

    # Handle singularity in psi_ref
    psi_ref_singularity = psi_ref % (2 * np.pi)
    if (abs(psi_ref_singularity) < np.pi / 4 or abs(psi_ref_singularity) > 7 * np.pi / 4 or 
      (abs(psi_ref_singularity) > 3 * np.pi / 4 and abs(psi_ref_singularity) < 5 * np.pi / 4)):
      tan_phi = np.cos(theta_ref) * (tan_theta * d - b) / c
    else:
      tan_phi = np.cos(theta_ref) * (a - tan_theta * c) / d


    phi_ref = np.arctan(tan_phi)
    U1 = (vz + g) * m / (np.cos(phi_ref) * np.cos(theta_ref))

    return phi_ref, theta_ref, U1


  #############################
  # Controls and Optimization #
  #############################

  def add_initial_state_constraint(self, prog, zeta, zeta_current):
    # Impose initial state constraint.
    prog.AddBoundingBoxConstraint(zeta_current, zeta_current, zeta[0])    

  def add_input_saturation_constraint(self, prog, U, N):
    # Impose input limit constraint.
    for i in range(N - 1):
        prog.AddBoundingBoxConstraint(self.U_min, self.U_max, U[i])

  def add_dynamics_constraint(self, prog, zeta, U, N, T):
    # Adds dynamics constraints to the MPC optimization problem.
    
    # Get linearized discrete-time dynamics
    A_d, B_d = self.discrete_time_linearized_dynamics(T)
    
    # To compute P_clf efficiently
    # self.P_clf = solve_discrete_lyapunov(A_d.T, self.Q)

    # Add constraints for each time step
    for i in range(N - 1):
      dynamics_constraint = (
        zeta[i + 1] - A_d @ zeta[i] - B_d @ U[i]
      )
      prog.AddLinearEqualityConstraint(
        dynamics_constraint, np.zeros_like(zeta[0])
      )

  def add_integral_dynamics_constraint(self, prog, zeta, zeta_integral, zeta_current, N):
      # Initial integral state
      prog.AddBoundingBoxConstraint(np.zeros_like(zeta_current), np.zeros_like(zeta_current), zeta_integral[0])
      
      for i in range(N - 1):
          integral_update = zeta_integral[i + 1] - zeta_integral[i] - (zeta[i] - self.zeta_d())
          prog.AddLinearEqualityConstraint(integral_update, np.zeros_like(zeta_current))

   
  def add_cost(self, prog, zeta, U, N):
    cost = 0
    
    for i in range(N - 1):
      # Penalize state deviation and input effort
      cost += (zeta[i] - self.zeta_d()).T @ self.Q @ (zeta[i] - self.zeta_d())
      cost += (U[i] - self.U_d()).T @ self.R @ (U[i] - self.U_d())
      
    # cost += zeta[N - 1].T @ self.Qf @ zeta[N - 1]
    prog.AddQuadraticCost(cost)
    
  def add_cost_w_integral_action(self, prog, zeta, zeta_integral, U, N):
    cost = 0
    
    for i in range(N - 1):
      # Penalize state deviation and input effort
      cost += (zeta[i] - self.zeta_d()).T @ self.Q @ (zeta[i] - self.zeta_d())
      cost += (U[i] - self.U_d()).T @ self.R @ (U[i] - self.U_d())
      
      # Penalize integral of error
      cost += zeta_integral[i].T @ self.Q_integral @ zeta_integral[i]
    
    prog.AddQuadraticCost(cost)
    
  def add_clf_constraint(self, prog, zeta, zeta_current, N):
    """
    Adds a Control Lyapunov Function (CLF) constraint to ensure stability.
    """
    
    # Define the Lyapunov function: V(zeta) = (zeta - zeta_d)^T * P * (zeta - zeta_d)
    P = self.P_clf  # Positive definite matrix for Lyapunov function
    alpha = self.alpha_clf  # Decay rate for CLF constraint

    for i in range(N - 1):
      V_next = (zeta[i + 1] - self.zeta_d()).T @ P @ (zeta[i + 1] - self.zeta_d())
      V_current = (zeta[i] - self.zeta_d()).T @ P @ (zeta[i] - self.zeta_d())
      clf_constraint = V_next - V_current + alpha * V_current
      prog.AddLinearConstraint(clf_constraint <= 0)
    
    
  def add_fb_lin_constraints(self, prog, phi_ref, theta_ref, U1_fb, zeta, U, N):
    """
    Add constraints from feedback linearization outputs to the MPC problem.
    """
    U_min, U_max = self.U_min, self.U_max
    
    assert not np.isnan(U1_fb), "U1 is NaN!"
    U_min[0], U_max[0] = U1_fb - 0.1, U1_fb + 0.1
    for i in range(N - 1):
        # prog.AddBoundingBoxConstraint(phi_ref - 0.2, phi_ref + 0.2, zeta[i][3])  # Roll angle
        # prog.AddBoundingBoxConstraint(theta_ref - 0.25, theta_ref + 0.25, zeta[i][4])  # Pitch angle
        prog.AddBoundingBoxConstraint(U_min, U_max, U[i])  # Thrust control
     
  def compute_mpc_feedback(
    self, zeta_current, print_U=False, use_clf=False, use_integral_action=False, use_fb_lin=False
    ):
    '''
    This function computes the MPC controller input omega
    '''

    # Parameters for the QP
    N = 10 # Prediction Horizon
    T = 0.1 # Timestep

    # Initialize mathematical program and decalre decision variables
    prog = MathematicalProgram()

    # State variables
    zeta = np.zeros((N, self.n_zeta), dtype="object")
    for i in range(N):
      zeta[i] = prog.NewContinuousVariables(self.n_zeta, "z_" + str(i))
    
    # U variables
    U = np.zeros((N-1, self.n_u), dtype="object")
    for i in range(N-1):
      U[i] = prog.NewContinuousVariables(self.n_u, "U_" + str(i))
      
    # Add integral action variables
    if use_integral_action:
        zeta_integral = np.zeros((N, self.n_zeta), dtype="object")
        for i in range(N):
            zeta_integral[i] = prog.NewContinuousVariables(self.n_zeta, "z_int_" + str(i))

    # Apply feedback linearization if enabled
    if use_fb_lin:
      phi_ref, theta_ref, U1_fb = self.feedback_linearization(zeta_current)
      # Add feedback linearization constraints
      self.add_fb_lin_constraints(prog, phi_ref, theta_ref, U1_fb, zeta, U, N)

    # Add constraints and cost
    self.add_initial_state_constraint(prog, zeta, zeta_current)
    # self.add_input_saturation_constraint(prog, U, N)
    self.add_dynamics_constraint(prog, zeta, U, N, T)
    
    if use_integral_action:
      self.add_integral_dynamics_constraint(prog, zeta, zeta_integral, zeta_current, N)
      self.add_cost_w_integral_action(prog, zeta, zeta_integral, U, N)
    else:
      self.add_cost(prog, zeta, U, N)
    
    if use_clf:
      self.add_clf_constraint(prog, zeta, zeta_current, N)

    # Solve the QP
    solver = OsqpSolver()
    result = solver.Solve(prog)

    if result.is_success():
        U_mpc = result.GetSolution(U[0])
        
        omega_mpc = self.U_to_omega(U_mpc)
        # print(f"Control input at this step: {omega_mpc}") 
    else:
        print(f"Solver failed to find a solution at {zeta_current}.")
        # return False
        omega_mpc = np.zeros(self.n_u) 
    
    if print_U:
      print(f"U solution: {np.round(result.GetSolution(U[0]), decimals=2)}")
        
    return omega_mpc
  
  
  
  
  
  def R_T_matrices(self, phi, theta, psi):
    # Rotation matrix relating body frame velocities to inertial frame velocities
    R = np.array([
        [np.cos(theta) * np.cos(psi), 
         np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
         np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
        [np.cos(theta) * np.sin(psi), 
         np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
         np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
        [-np.sin(theta), 
         np.sin(phi) * np.cos(theta), 
         np.cos(phi) * np.cos(theta)]
    ])
    
    # Transformation matrix relating angular velocities to Euler angle derivatives
    T = np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
    ])
    
    return R, T
  
  def get_motor_positions(self, state):
    X, Y, Z, phi, theta, psi = state[:6]
    R,_ = self.R_T_matrices(phi, theta, psi)
    motor_offsets = np.array([
        [self.L, 0, 0],  # Motor 1
        [0, self.L, 0],  # Motor 2
        [-self.L, 0, 0],  # Motor 3
        [0, -self.L, 0]  # Motor 4
    ]).T
    return R @ motor_offsets + np.array([[X], [Y], [Z]])