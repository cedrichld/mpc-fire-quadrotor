import numpy as np
import math

class Quad_Constants(object):
    # Constants
    def __init__(self):
        # Constants
        self.Ix  = 0.0034
        self.Iy  = 0.0034
        self.Iz  = 0.006
        self.m   = 0.698
        self.g   = 9.81
        self.Jtp = 1.302e-6
        self.Ts  = 0.1

        # Matrix weights for the cost function
        self.Q = np.diag([10, 10, 10])
        self.S = np.diag([10, 10, 10])
        self.R = np.diag([10, 10, 10])

        self.ct = 7.6184e-8 * (60/(2*np.pi))**2
        self.cq = 2.6839e-9 * (60/(2*np.pi))**2
        self.l = 0.171

        self.controlled_states = 3
        self.hz = 4  # horizon period
        self.inner_loop = 4

        self.px = [-1, -2]
        self.py = [-1, -2]
        self.pz = [-1, -2]

        self.trajectory = 0 # Custom
        
        # Input Limits: Not implemented currently
        self.U1_min, self.U1_max = - (self.m * self.g * 0.25), (self.m * self.g * 2)
        self.U2_min, self.U2_max = - math.inf, math.inf
        self.U3_min, self.U3_max = - math.inf, math.inf
        self.U4_min, self.U4_max = - math.inf, math.inf
        
        self.U_min = np.array([self.U1_min, self.U2_min, self.U3_min, self.U4_min])
        self.U_max = np.array([self.U1_max, self.U2_max, self.U3_max, self.U4_max])
        
        # M matrix from U to omega^2
        self.M_inv = np.linalg.inv(np.array([
            [self.ct, self.ct, self.ct, self.ct],
            [0, self.ct * self.l, 0, -self.ct  *self.l],
            [-self.ct * self.l, 0,    self.ct * self.l, 0],
            [-self.cq, self.cq, -self.cq, self.cq]
        ]))
    
        # Nominal input (hover condition)
        self.U_d = np.array([self.m * self.g, 0, 0, 0])
        
    # Rotation matrix R
    def R_matrix(self, phi, theta, psi):
        return np.array([
            [np.cos(theta)*np.cos(psi), 
            np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
            np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],

            [np.cos(theta)*np.sin(psi), 
            np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),
            np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],

            [-np.sin(theta),
            np.sin(phi)*np.cos(theta),
            np.cos(phi)*np.cos(theta)]
        ])
    
    # Transformation matrix T
    def T_matrix(self, phi, theta):
        return np.array([
            [1, np.sin(phi)*np.tan(theta),  np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi),               -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta),  np.cos(phi)/np.cos(theta)]
        ])
