import sympy as sp

def compute_linearized_dynamics(m, g, L, I_x, I_y, I_z, x_d, u_d):
    """
    Computes and evaluates the linearized dynamics (A and B matrices)
    at the desired state x_d and input u_d.
    """
    # Define state variables
    q = sp.Matrix(sp.symbols('X Y Z phi theta psi X_dot Y_dot Z_dot phi_dot theta_dot psi_dot'))
    u = sp.Matrix(sp.symbols('omega1 omega2 omega3 omega4'))

    # Extract state components
    X, Y, Z, phi, theta, psi = q[:6]
    X_dot, Y_dot, Z_dot, phi_dot, theta_dot, psi_dot = q[6:]
    
    # Rotation matrix components (simplified for linearization)
    R = sp.Matrix([
        [sp.cos(theta)*sp.cos(psi), -sp.cos(phi)*sp.sin(psi)+sp.sin(phi)*sp.sin(theta)*sp.cos(psi), sp.sin(phi)*sp.sin(psi)+sp.cos(phi)*sp.sin(theta)*sp.cos(psi)],
        [sp.cos(theta)*sp.sin(psi), sp.cos(phi)*sp.cos(psi)+sp.sin(phi)*sp.sin(theta)*sp.sin(psi), -sp.sin(phi)*sp.cos(psi)+sp.cos(phi)*sp.sin(theta)*sp.sin(psi)],
        [-sp.sin(theta), sp.sin(phi)*sp.cos(theta), sp.cos(phi)*sp.cos(theta)]
    ])
    
    # Compute thrust and torques
    K, b = sp.symbols('K b')  # Thrust and drag coefficients
    T = K * (u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2)
    tau_phi = K * L * (u[3]**2 - u[1]**2)
    tau_theta = K * L * (u[2]**2 - u[0]**2)
    tau_psi = b * (u[0]**2 - u[1]**2 + u[2]**2 - u[3]**2)

    # Translational accelerations
    acc_body = sp.Matrix([0, 0, T / m])
    acc_world = R @ acc_body - sp.Matrix([0, 0, g])
    X_ddot, Y_ddot, Z_ddot = acc_world

    # Rotational accelerations
    omega_dot = sp.Matrix([
        (tau_phi + (I_y - I_z) * theta_dot * psi_dot) / I_x,
        (tau_theta + (I_z - I_x) * phi_dot * psi_dot) / I_y,
        (tau_psi + (I_x - I_y) * phi_dot * theta_dot) / I_z
    ])
    phi_ddot, theta_ddot, psi_ddot = omega_dot

    # Full dynamics vector
    dynamics = sp.Matrix.vstack(
        sp.Matrix(q[6:]),  # Velocities as derivatives of positions
        sp.Matrix([X_ddot, Y_ddot, Z_ddot]),
        sp.Matrix(omega_dot)
    )


    # Compute Jacobians
    A = dynamics.jacobian(q)  # Partial derivatives wrt state
    B = dynamics.jacobian(u)  # Partial derivatives wrt control

    # Substitute equilibrium state and control inputs
    substitutions = {q[i]: x_d[i] for i in range(len(q))}
    substitutions.update({u[i]: u_d[i] for i in range(len(u))})

    A_evaluated = A.subs(substitutions)
    B_evaluated = B.subs(substitutions)

    # Return evaluated A and B matrices
    return A_evaluated, B_evaluated

# Example usage:
m, g, L, I_x, I_y, I_z, K, b = sp.symbols('m g L I_x I_y I_z K b')
x_d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Hover at origin, zero velocities
u_d = [sp.sqrt(m * g / (4 * K))] * 4  # Equal thrust from all rotors
A_evaluated, B_evaluated = compute_linearized_dynamics(m, g, L, I_x, I_y, I_z, x_d, u_d)

# Display results
sp.pprint(A_evaluated)
sp.pprint(B_evaluated)
