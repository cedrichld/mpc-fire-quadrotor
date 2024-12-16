import numpy as np

def LPV(constants, states, omega_total):
    Ix = constants.Ix
    Iy = constants.Iy
    Iz = constants.Iz
    Jtp = constants.Jtp
    Ts = constants.Ts
    
    # states: [u, v, w, p, q, r, x, y, z, phi, theta, psi]
    u = states[0]
    v = states[1]
    w = states[2]
    p = states[3]
    q = states[4]
    r = states[5]
    phi = states[9]
    theta = states[10]
    psi = states[11]

    R_matrix = constants.R_matrix(phi, theta, psi)
    T_matrix = constants.T_matrix(phi, theta)
    
    vec_uvw = np.array([u, v, w])
    x_dot = R_matrix[0,:] @ vec_uvw
    y_dot = R_matrix[1,:] @ vec_uvw
    z_dot = R_matrix[2,:] @ vec_uvw

    vec_pqr = np.array([p, q, r])
    phi_dot = T_matrix[0,:] @ vec_pqr
    theta_dot = T_matrix[1,:] @ vec_pqr
    psi_dot = T_matrix[2,:] @ vec_pqr

    # Intermediate terms for A
    A12 = 1
    A24 = -omega_total*Jtp/Ix
    A26 = theta_dot*(Iy - Iz)/Ix
    A34 = 1
    A42 = omega_total*Jtp/Iy
    A46 = phi_dot*(Iz - Ix)/Iy
    A56 = 1
    A62 = (theta_dot/2)*(Ix - Iy)/Iz
    A64 = (phi_dot/2)*(Ix - Iy)/Iz

    # Continuous A, B, C, D matrices
    A = np.array([
        [0,   A12, 0,   0,   0,   0],
        [0,    0,  0,  A24,  0,  A26],
        [0,    0,  0,  A34,  0,   0],
        [0,   A42, 0,   0,   0,  A46],
        [0,    0,  0,   0,   0,  A56],
        [0,   A62, 0,  A64,  0,   0]
    ])

    B = np.array([
        [0,      0,     0    ],
        [1/Ix,   0,     0    ],
        [0,      0,     0    ],
        [0,    1/Iy,     0    ],
        [0,      0,     0    ],
        [0,      0,    1/Iz  ]
    ])

    C = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ])

    # D = np.zeros((3,3)) # not used

    # Discretize the system using Forward Euler
    Ad = np.eye(A.shape[0]) + Ts * A
    Bd = Ts * B
    Cd = C
    # Dd = D

    return Ad, Bd, Cd, x_dot, y_dot, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot
