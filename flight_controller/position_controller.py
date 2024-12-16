import numpy as np

def feedback_linearization(X_ref, X_dot_ref, X_dot_dot_ref,
                           Y_ref, Y_dot_ref, Y_dot_dot_ref,
                           Z_ref, Z_dot_ref, Z_dot_dot_ref,
                           Psi_ref, states, constants):
    """
    Python version of the MATLAB Feedback_Linearization function.

    Parameters:
    - X_ref, X_dot_ref, X_dot_dot_ref
    - Y_ref, Y_dot_ref, Y_dot_dot_ref
    - Z_ref, Z_dot_ref, Z_dot_dot_ref
    - Psi_ref: desired yaw angle
    - states: current state vector [u, v, w, p, q, r, x, y, z, phi, theta, psi]
    - constants: dictionary or class with required constants {m, g, px, py, pz}

    Returns:
    - Phi_ref: Desired roll angle
    - Theta_ref: Desired pitch angle
    - U1: Desired thrust
    """
    m  = constants.m
    g  = constants.g
    px = constants.px
    py = constants.py
    pz = constants.pz

    # Extract states
    u     = states[0]
    v     = states[1]
    w     = states[2]
    # p     = states[3]
    # q     = states[4]
    # r     = states[5]
    x     = states[6]
    y     = states[7]
    z     = states[8]
    phi   = states[9]
    theta = states[10]
    psi   = states[11]

    # Rotation matrix
    R_matrix = constants.R_matrix(phi, theta, psi)

    uvw = np.array([u, v, w])
    x_dot, y_dot, z_dot = R_matrix @ uvw

    # Errors
    ex = X_ref - x
    ex_dot = X_dot_ref - x_dot
    ey = Y_ref - y
    ey_dot = Y_dot_ref - y_dot
    ez = Z_ref - z
    ez_dot = Z_dot_ref - z_dot

    # Compute gains kx1, kx2
    kx1 = (px[0]-(px[0]+px[1])/2)**2 - ((px[0]+px[1])**2)/4
    kx2 = px[0]+px[1]
    kx1 = np.real(kx1)
    kx2 = np.real(kx2)

    # ky1, ky2
    ky1 = (py[0]-(py[0]+py[1])/2)**2 - ((py[0]+py[1])**2)/4
    ky2 = py[0]+py[1]
    ky1 = np.real(ky1)
    ky2 = np.real(ky2)

    # kz1, kz2
    kz1 = (pz[0]-(pz[0]+pz[1])/2)**2 - ((pz[0]+pz[1])**2)/4
    kz2 = pz[0]+pz[1]
    kz1 = np.real(kz1)
    kz2 = np.real(kz2)

    # ux, uy, uz
    ux = kx1*ex + kx2*ex_dot
    uy = ky1*ey + ky2*ey_dot
    uz = kz1*ez + kz2*ez_dot

    # vx, vy, vz
    vx = X_dot_dot_ref - ux
    vy = Y_dot_dot_ref - uy
    vz = Z_dot_dot_ref - uz

    # Compute Theta_ref
    a = vx/(vz+g)
    b = vy/(vz+g)
    c_ = np.cos(Psi_ref)
    d_ = np.sin(Psi_ref)
    tan_theta = a*c_ + b*d_
    Theta_ref = np.arctan(tan_theta)

    # Compute Psi_ref_singularity
    if Psi_ref >= 0:
        Psi_ref_singularity = Psi_ref - (np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi)
    else:
        Psi_ref_singularity = Psi_ref + (np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi)

    # Compute Phi_ref
    if (abs(Psi_ref_singularity)<np.pi/4 or abs(Psi_ref_singularity)>7*np.pi/4) or \
       ((abs(Psi_ref_singularity)>3*np.pi/4) and (abs(Psi_ref_singularity)<5*np.pi/4)):
        tan_phi = np.cos(Theta_ref)*((tan_theta)*d_ - b)/c_
    else:
        tan_phi = np.cos(Theta_ref)*(a - (tan_theta)*c_)/d_

    Phi_ref = np.arctan(tan_phi)
    U1 = (vz+g)*m/(np.cos(Phi_ref)*np.cos(Theta_ref))

    return Phi_ref, Theta_ref, U1