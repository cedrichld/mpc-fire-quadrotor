import sympy as sp

def compute_linearized_dynamics(m, g, Jtp, Ix, Iy, Iz, x_d, u_d):
    """
    Computes and evaluates the linearized dynamics (A and B matrices)
    at the desired state x_d and input u_d.
    """
    # Define state variables and inputs
    q = sp.Matrix(sp.symbols('x y z phi theta psi u v w p q r'))
    u = sp.Matrix(sp.symbols('U1 U2 U3 U4'))

    # Extract state components
    
    phi, theta, psi = q[3:6]  # Euler angles
    u_vel, v, w = q[6:9]  # Linear velocities
    p, q_ang, r = q[9:12]  # Angular velocities

    # Define nonlinear dynamics
    # Translational kinematics
    R_matrix = sp.Matrix([
        [sp.cos(theta) * sp.cos(psi), 
         sp.sin(phi) * sp.sin(theta) * sp.cos(psi) - sp.cos(phi) * sp.sin(psi),
         sp.cos(phi) * sp.sin(theta) * sp.cos(psi) + sp.sin(phi) * sp.sin(psi)],
        [sp.cos(theta) * sp.sin(psi), 
         sp.sin(phi) * sp.sin(theta) * sp.sin(psi) + sp.cos(phi) * sp.cos(psi),
         sp.cos(phi) * sp.sin(theta) * sp.sin(psi) - sp.sin(phi) * sp.cos(psi)],
        [-sp.sin(theta), 
         sp.sin(phi) * sp.cos(theta), 
         sp.cos(phi) * sp.cos(theta)]
    ])

    T_matrix = sp.Matrix([
        [1, sp.sin(phi) * sp.tan(theta), sp.cos(phi) * sp.tan(theta)],
        [0, sp.cos(phi), -sp.sin(phi)],
        [0, sp.sin(phi) / sp.cos(theta), sp.cos(phi) / sp.cos(theta)]
    ])

    dx = sp.Matrix.zeros(12, 1)

    # Translational kinematics
    velocity_body = sp.Matrix([u_vel, v, w])
    translational_kinematics = R_matrix @ velocity_body
    dx[0] = translational_kinematics[0]  # x_dot
    dx[1] = translational_kinematics[1]  # y_dot
    dx[2] = translational_kinematics[2]  # z_dot

    # Rotational kinematics
    angular_vel_body = sp.Matrix([p, q_ang, r])
    rotational_kinematics = T_matrix @ angular_vel_body
    dx[3] = rotational_kinematics[0]  # phi_dot
    dx[4] = rotational_kinematics[1]  # theta_dot
    dx[5] = rotational_kinematics[2]  # psi_dot

    # Translational dynamics
    dx[6] = v * r - w * q_ang + g * sp.sin(theta)  # u_dot
    dx[7] = w * p - u_vel * r - g * sp.cos(theta) * sp.sin(phi)  # v_dot
    dx[8] = u_vel * q_ang - v * p - g * sp.cos(theta) * sp.cos(phi) + u[0] / m  # w_dot

    # Rotational dynamics
    dx[9] = (q_ang * r * (Iy - Iz) - Jtp * q_ang * sp.symbols('omega_total') + u[1]) / Ix  # p_dot
    dx[10] = (p * r * (Iz - Ix) + Jtp * p * sp.symbols('omega_total') + u[2]) / Iy  # q_dot
    dx[11] = (p * q_ang * (Ix - Iy) + u[3]) / Iz  # r_dot

    # Compute Jacobians
    A = dx.jacobian(q)  # Partial derivatives wrt state
    B = dx.jacobian(u)  # Partial derivatives wrt control

    # Substitute equilibrium state and control inputs
    substitutions = {q[i]: x_d[i] for i in range(len(q))}
    substitutions.update({u[i]: u_d[i] for i in range(len(u))})

    A_evaluated = A.subs(substitutions)
    B_evaluated = B.subs(substitutions)

    # Return evaluated A and B matrices
    return A_evaluated, B_evaluated

# Example usage
m, g, Jtp, Ix, Iy, Iz = sp.symbols('m g Jtp Ix Iy Iz')
x_d = [0] * 12  # Hover at origin, zero velocities
u_d = [sp.sqrt(m * g / 4)] * 4  # Equal thrust from all rotors
A_evaluated, B_evaluated = compute_linearized_dynamics(m, g, Jtp, Ix, Iy, Iz, x_d, u_d)

# Display results
sp.pprint(A_evaluated)
sp.pprint(B_evaluated)
