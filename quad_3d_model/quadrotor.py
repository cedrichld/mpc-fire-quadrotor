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

    # State and control dimensions
    self.n_zeta = 12  # State dimension (x, y, z, phi, theta, psi + their velocities)
    self.n_u = 4      # Input dimension (4 rotors)
    
    # Input limits
    # self.omega_min = self.omega_ref * 0.5 # no motor input
    # self.omega_max = self.omega_ref * 2 # max rotation speed of 3 times required speed
    self.U1_min, self.U1_max = (self.m * self.g * 0.65), (self.m * self.g * 2)
    self.U2_min, self.U2_max = - math.inf, math.inf
    self.U3_min, self.U3_max = - math.inf, math.inf
    self.U4_min, self.U4_max = - math.inf, math.inf
    
    self.U_min = np.array([self.U1_min, self.U2_min, self.U3_min, self.U4_min])
    self.U_max = np.array([self.U1_max, self.U2_max, self.U3_max, self.U4_max])
    
    # M matrix from U to omega^2
    self.M = np.array([
        [self.ct,    self.ct,    self.ct,    self.ct   ],
        [0,          self.ct*self.L, 0,      -self.ct*self.L],
        [-self.ct*self.L, 0,    self.ct*self.L, 0       ],
        [-self.cq,   self.cq,   -self.cq,   self.cq    ]
    ])
    
    self.M_inv = np.linalg.inv(self.M)
    
    # X^T = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]
    # U^T = [F_z, tau_phi, tau_theta, tau_psi] or [U1, U2, U3, U4]

  def omega_d(self):
    return 0.5 * np.sqrt((self.m * self.g) / (self.n_u * self.ct)) # * 0.5?
    
    
  def zeta_d(self):
    # Nominal state
    zeta = np.zeros(12)
    zeta[0] = 0
    zeta[2] = -0.5
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
    R_matrix = np.array([
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
    T_matrix = np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)] # May lead to divides by zero
    ])
    
    if np.cos(theta) == 0:
      print("np.cos(theta) is 0!!")

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
    
    # A[4, 9] = Jtp * omega_total / I_y
    # A[3, 10] = - Jtp * omega_total / I_x

    # Rotational dynamics coupling
    # A[9, 3] = (I_y - I_z) / I_x - Jtp * omega_total / I_x  # Roll dynamics (phi)
    # A[10, 4] = (I_z - I_x) / I_y + Jtp * omega_total / I_y  # Pitch dynamics (theta)
    # A[11, 10] = (I_x - I_y) / I_z  # Yaw dynamics (psi)

    # Fill B matrix
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
    """
    Adds dynamics constraints to the MPC optimization problem.

    Parameters:
    prog: MathematicalProgram
        The optimization problem.
    zeta: ndarray
        State trajectory decision variables.
    omega: ndarray
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
        zeta[i + 1] - A_d @ zeta[i] - B_d @ U[i]
      )
      prog.AddLinearEqualityConstraint(
        dynamics_constraint, np.zeros_like(zeta[0])
      )

  def add_cost(self, prog, zeta, U, N):
    cost = 0
    
    for i in range(N - 1):
      cost += (zeta[i] - self.zeta_d()).T @ self.Q @ (zeta[i] - self.zeta_d())
      cost += (U[i] - self.U_d()).T @ self.R @ (U[i] - self.U_d())
    # cost += zeta[N - 1].T @ self.Qf @ zeta[N - 1]
    prog.AddQuadraticCost(cost)

  def compute_mpc_feedback(self, zeta_current, use_clf=False):
    '''
    This function computes the MPC controller input omega
    '''

    # Parameters for the QP
    N = 10 # Prediction Horizon
    T = 0.1 # Timestep

    # Initialize mathematical program and decalre decision variables
    prog = MathematicalProgram()

    zeta = np.zeros((N, self.n_zeta), dtype="object")
    for i in range(N):
      zeta[i] = prog.NewContinuousVariables(self.n_zeta, "z_" + str(i))
    # Based on thrust
    U = np.zeros((N-1, self.n_u), dtype="object")
    for i in range(N-1):
      U[i] = prog.NewContinuousVariables(self.n_u, "U_" + str(i))

    # Add constraints and cost
    self.add_initial_state_constraint(prog, zeta, zeta_current)
    self.add_input_saturation_constraint(prog, U, N)
    self.add_dynamics_constraint(prog, zeta, U, N, T)
    self.add_cost(prog, zeta, U, N)

    # Solve the QP
    solver = OsqpSolver()
    result = solver.Solve(prog)

    if result.is_success():
        U_mpc = result.GetSolution(U[0])# + self.omega_d()
        
        omega_mpc = self.U_to_omega(U_mpc)
        # print(f"Control input at this step: {omega_mpc}") 
    else:
        print(f"Solver failed to find a solution at {zeta_current}.")
        # return False
        omega_mpc = np.zeros(self.n_u) 
    
    print(f"U solution: {np.round(result.GetSolution(U[0]), decimals=2)}")
        
    return omega_mpc
  
  
  
  
  
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