import numpy as np

class full_dynamics(object):
    # Continuous full time dynamics given to the simulator
    def continuous(self, constants, zeta, U1, U2, U3, U4, omega_total):
        '''
        Input: Takes in the current zeta and omega
        Output: Returns the current dzeta
        '''
        # Constants
        m, g, Jtp = constants.m, constants.g, constants.Jtp  # Mass, gravity, torque precession constant
        Ix, Iy, Iz = constants.Ix, constants.Iy, constants.Iz  # Moments of inertia

        # State: [u, v, w, p, q, r, x, y, z, phi, theta, psi]
        # Extract states
        u, v, w = zeta[0], zeta[1], zeta[2]
        p, q, r = zeta[3], zeta[4], zeta[5]
        x, y, z = zeta[6], zeta[7], zeta[8]
        phi, theta, psi = zeta[9], zeta[10], zeta[11]
        
        # Rotation matrix relating body frame velocities to inertial frame velocities
        # Transformation matrix relating angular velocities to Euler angle derivatives
        R_matrix = constants.R_matrix(phi, theta, psi)
        T_matrix = constants.T_matrix(phi, theta)

        # Compute nonlinear dynamics
        dzeta = np.zeros(12)

        # Nonlinear equations of motion (from drone_plant.m)
        # du/dt
        dzeta[0] = (v*r - w*q) + g * np.sin(theta)
        # dv/dt
        dzeta[1] = (w*p - u*r) - g * np.cos(theta) * np.sin(phi)
        # dw/dt
        dzeta[2] = (u*q - v*p) - g * np.cos(theta) * np.cos(phi) + U1 / m
        # dp/dt
        dzeta[3] = (q*r*(Iy - Iz) - Jtp*q*omega_total + U2) / Ix
        # dq/dt
        dzeta[4] = (p*r*(Iz - Ix) + Jtp*p*omega_total + U3) / Iy
        # dr/dt
        dzeta[5] = (p*q*(Ix - Iy) + U4) / Iz
        # dx/dt, dy/dt, dz/dt
        xyz_dot = R_matrix @ np.array([u, v, w])
        dzeta[6] = xyz_dot[0]  # dx/dt
        dzeta[7] = xyz_dot[1]  # dy/dt
        dzeta[8] = xyz_dot[2]  # dz/dt
        # dphi/dt, dtheta/dt, dpsi/dt
        euler_dots = T_matrix @ np.array([p, q, r])
        dzeta[9]  = euler_dots[0]  # dphi/dt
        dzeta[10] = euler_dots[1]  # dtheta/dt
        dzeta[11] = euler_dots[2]  # dpsi/dt

        return dzeta

    def U_calculator(self, constants, omega):
        U1 = constants.ct * (omega[0]**2 + omega[1]**2 + omega[2]**2 + omega[3]**2)
        U2 = constants.ct * constants.l * (omega[1]**2 - omega[3]**2)
        U3 = constants.ct * constants.l * (omega[2]**2 - omega[0]**2)
        U4 = constants.cq * (-omega[0]**2 + omega[1]**2 - omega[2]**2 + omega[3]**2)
        omega_total = omega[0] - omega[1] + omega[2] - omega[3]
        
        # print(f"U1, U2, U3, U4, omega_total are {U1, U2, U3, U4, omega_total}")
        
        return U1, U2, U3, U4, omega_total