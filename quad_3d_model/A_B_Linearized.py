import sympy as sp

def compute_linearized_dynamics(x_d, u_d):
    """
    Computes and evaluates the linearized dynamics (A and B matrices)
    at the desired state x_d and input u_d.
    """
    # Define state variables (q = [X, Y, Z, phi, theta, psi, X_dot, Y_dot, Z_dot, phi_dot, theta_dot, psi_dot])
    q = sp.Matrix(sp.symbols('X Y Z phi theta psi X_dot Y_dot Z_dot phi_dot theta_dot psi_dot'))
    q_dot = sp.Matrix(sp.symbols('X_ddot Y_ddot Z_ddot phi_ddot theta_ddot psi_ddot'))

    # Define control inputs (u = [u1, u2, u3, u4])
    u = sp.Matrix(sp.symbols('u1 u2 u3 u4'))

    # Define parameters
    m, g, I_x, I_y, I_z, L = sp.symbols('m g I_x I_y I_z L')
    phi, theta, psi = q[3], q[4], q[5]
    phi_dot, theta_dot, psi_dot = q[9], q[10], q[11]

    # Define thrust-to-force mappings
    T = sp.Matrix([sp.cos(phi) * sp.sin(theta) * sp.cos(psi) + sp.sin(phi) * sp.sin(psi),
                   sp.cos(phi) * sp.sin(theta) * sp.sin(psi) - sp.sin(phi) * sp.cos(psi),
                   sp.cos(phi) * sp.cos(theta)]) * (u[0] + u[1] + u[2] + u[3])

    # Translational Dynamics
    f_trans = sp.Matrix([
        T[0] / m,
        T[1] / m,
        T[2] / m - g
    ])

    # Rotational Dynamics
    f_rot = sp.Matrix([
        (L / I_x) * (u[1] - u[3]) + psi_dot * (I_y - I_z) / I_x - (u[0] + u[1] + u[2] + u[3]) / I_x,
        (L / I_y) * (u[2] - u[0]) + phi_dot * (I_z - I_x) / I_y + (u[0] + u[1] + u[2] + u[3]) / I_y,
        (1 / I_z) * (u[0] - u[1] + u[2] - u[3])
    ])

    # Combine full dynamics
    dynamics = sp.Matrix.vstack(sp.Matrix(q[6:]), f_trans, f_rot)

    # Compute Jacobians for linearization
    A = dynamics.jacobian(q)  # Partial derivatives of dynamics wrt state
    B = dynamics.jacobian(u)  # Partial derivatives of dynamics wrt inputs

    # Substitute desired state x_d and input u_d
    substitutions = {q[i]: x_d[i] for i in range(len(q))}
    substitutions.update({u[i]: u_d[i] for i in range(len(u))})
    A_evaluated = A.subs(substitutions)
    B_evaluated = B.subs(substitutions)

    # Return evaluated A and B
    return A_evaluated, B_evaluated

# Example usage:
# Desired state (x_d) and input (u_d) for hover condition

m, g = sp.symbols('m g')

x_d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Hover at origin, zero velocity
u_d = [m * g / 4, m * g / 4, m * g / 4, m * g / 4]  # Thrust balances gravity
# Compute and evaluate linearized matrices
A_evaluated, B_evaluated = compute_linearized_dynamics(x_d, u_d)

# Display results
sp.pprint(A_evaluated)
sp.pprint(B_evaluated)
