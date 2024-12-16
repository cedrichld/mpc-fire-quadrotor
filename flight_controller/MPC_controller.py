import numpy as np

def MPC(constants, Ad, Bd, Cd, hz):
    Q = constants.Q
    S = constants.S
    R = constants.R

    # Dimensions
    nx = Ad.shape[0]   # states
    nu = Bd.shape[1]   # inputs
    ny = Cd.shape[0]   # outputs

    # Augmented state-space
    # A_aug = [Ad    Bd
    #          0     I ]
    A_aug = np.block([
        [Ad, Bd],
        [np.zeros((nu, nx)), np.eye(nu)]
    ])

    B_aug = np.block([
        [Bd],
        [np.eye(nu)]
    ])

    C_aug = np.block([
        [Cd, np.zeros((ny, nu))]
    ])

    # D_aug = Dd # D_aug is zero here

    # Compute intermediate matrices
    CQC = C_aug.T @ Q @ C_aug
    CSC = C_aug.T @ S @ C_aug
    QC = Q @ C_aug
    SC = S @ C_aug
 
    # Dimensions of block matrices
    # CQC and CSC are (nx + nu) x (nx + nu)
    # QC and SC are ny x (nx + nu)
    # R is nu x nu
    # B_aug is (nx + nu) x nu
    # A_aug is (nx + nu) x (nx + nu)

    Qdb = np.zeros(((nx + nu) * hz, (nx + nu) * hz))
    Tdb = np.zeros((ny * hz, (nx + nu) * hz))
    Rdb = np.zeros((nu * hz, nu * hz))
    Cdb = np.zeros(((nx + nu) * hz, nu * hz))
    Adc = np.zeros(((nx + nu) * hz, (nx + nu)))

    for i in range(hz):
        # Index ranges
        # For Qdb and Tdb blocks:
        Q_row_start = i * (nx + nu)
        Q_row_end   = (i + 1) * (nx + nu)
        Q_col_start = i * (nx + nu)
        Q_col_end   = (i + 1) * (nx + nu)

        T_row_start = i * ny
        T_row_end   = (i + 1) * ny
        T_col_start = i * (nx + nu)
        T_col_end   = (i + 1) * (nx + nu)

        # For Rdb blocks:
        R_row_start = i * nu
        R_row_end   = (i + 1) * nu
        R_col_start = i * nu
        R_col_end   = (i + 1) * nu

        # Fill Qdb and Tdb
        if i == hz - 1:
            Qdb[Q_row_start:Q_row_end, Q_col_start:Q_col_end] = CSC
            Tdb[T_row_start:T_row_end, T_col_start:T_col_end] = SC
        else:
            Qdb[Q_row_start:Q_row_end, Q_col_start:Q_col_end] = CQC
            Tdb[T_row_start:T_row_end, T_col_start:T_col_end] = QC

        # Fill Rdb
        Rdb[R_row_start:R_row_end, R_col_start:R_col_end] = R

        # Fill Cdb and Adc
        # We need A_aug^i  and A_aug^(i-j)
        A_aug_i = np.linalg.matrix_power(A_aug, (i + 1))
        Adc[(i * (nx + nu)):((i + 1) * (nx + nu)), 0:(nx + nu)] = A_aug_i

        for j in range(hz):
            if j <= i:
                A_aug_power = np.linalg.matrix_power(A_aug, i - j)
                Cdb_row_start = i * (nx + nu)
                Cdb_row_end   = (i + 1) * (nx + nu)
                Cdb_col_start = j * nu
                Cdb_col_end   = (j + 1) * nu
                Cdb[Cdb_row_start:Cdb_row_end, Cdb_col_start:Cdb_col_end] = A_aug_power @ B_aug

    # Compute Hdb and Fdbt
    Hdb = Cdb.T @ Qdb @ Cdb + Rdb
    Fdbt = np.vstack((Adc.T @ Qdb @ Cdb, -Tdb @ Cdb))

    return Hdb, Fdbt, Cdb, Adc


# Given U, solve for omega
def U_to_omega(U, M_inv):
    omega_squared = M_inv @ U

    # Not physically feasible
    if np.any(omega_squared < 0):
        omega_squared = np.clip(omega_squared, 0, None)
    
    return np.sqrt(omega_squared)