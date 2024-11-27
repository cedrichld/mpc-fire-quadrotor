import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Parameters
params = {
    "m": 0.468,
    "g": 9.81,
    "Ixx": 4.856e-3,
    "Iyy": 4.856e-3,
    "Izz": 8.801e-3,
    "l": 0.225,
    "K": 2.980e-6,
    "b": 1.14e-7,
    "Ax": 0.25 * 0,
    "Ay": 0.25 * 0,
    "Az": 0.25 * 0,
}

# Motor speeds
omega = 1.075
speed = omega * np.sqrt(1 / params["K"])
dspeed = 0.05 * speed
params["omega1"] = speed-0.5*dspeed
params["omega2"] = speed+0.5*dspeed
params["omega3"] = speed-0.5*dspeed
params["omega4"] = speed+0.5*dspeed

# Initial conditions
x0, y0, z0 = 0, 0, 0
vx0, vy0, vz0 = 0, 0, 0
phi0, theta0, psi0 = 0, 0, 0
phidot0, thetadot0, psidot0 = 0, 0, 0

Z0 = [x0, y0, z0, phi0, theta0, psi0, vx0, vy0, vz0, phidot0, thetadot0, psidot0]
t_span = (0, 1)
t_eval = np.linspace(*t_span, 500)

# Equations of motion
def eom(t, Z):
    m, g, Ixx, Iyy, Izz, l, K, b, Ax, Ay, Az = (
        params["m"],
        params["g"],
        params["Ixx"],
        params["Iyy"],
        params["Izz"],
        params["l"],
        params["K"],
        params["b"],
        params["Ax"],
        params["Ay"],
        params["Az"],
    )
    omega1, omega2, omega3, omega4 = params["omega1"], params["omega2"], params["omega3"], params["omega4"]

    x, y, z, phi, theta, psi, vx, vy, vz, phidot, thetadot, psidot = Z

    A = np.zeros((6, 6))
    A[0, 0] = m
    A[1, 1] = m
    A[2, 2] = m
    A[3, 3] = Ixx
    A[3, 5] = -Ixx * np.sin(theta)
    A[4, 4] = Iyy - Iyy * np.sin(phi) ** 2 + Izz * np.sin(phi) ** 2
    A[4, 5] = np.cos(phi) * np.cos(theta) * np.sin(phi) * (Iyy - Izz)
    A[5, 3] = -Ixx * np.sin(theta)
    A[5, 4] = np.cos(phi) * np.cos(theta) * np.sin(phi) * (Iyy - Izz)
    A[5, 5] = (
        Ixx * np.sin(theta) ** 2
        + Izz * np.cos(phi) ** 2 * np.cos(theta) ** 2
        + Iyy * np.cos(theta) ** 2 * np.sin(phi) ** 2
    )

    T = K * (omega1**2 + omega2**2 + omega3**2 + omega4**2)
    tau_x = K * l * (omega4**2 - omega2**2)
    tau_y = K * l * (omega3**2 - omega1**2)
    tau_z = b * (omega1**2 - omega2**2 + omega3**2 - omega4**2)

    B = np.array([
        T * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)) - Ax * vx,
        -T * (np.cos(psi) * np.sin(phi) - np.sin(psi) * np.cos(phi) * np.sin(theta)) - Ay * vy,
        T * np.cos(phi) * np.cos(theta) - m * g - Az * vz,
        tau_x,
        tau_y,
        tau_z,
    ])

    X = np.linalg.solve(A, B)
    return [
        vx, vy, vz,
        phidot, thetadot, psidot,
        X[0], X[1], X[2],
        X[3], X[4], X[5]
    ]

# Solve ODE
sol = solve_ivp(eom, t_span, Z0, t_eval=t_eval, rtol=1e-12, atol=1e-12)

# Plot results
plt.figure()
for i, label in enumerate(['x', 'y', 'z', r'$\phi$', r'$\theta$', r'$\psi$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$', r'$\dot{\phi}$', r'$\dot{\theta}$', r'$\dot{\psi}$']):
    plt.plot(sol.t, sol.y[i], label=label)
plt.legend(fontsize=8)
plt.xlabel('Time (s)')
plt.ylabel('State Variables')
plt.grid()
plt.show()

# Animation
def get_rotation(phi, theta, psi):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x

positions = sol.y[:3].T
angles = sol.y[3:6].T
l = params["l"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def animate(i):
    ax.clear()
    x, y, z = positions[i]
    phi, theta, psi = angles[i]
    R = get_rotation(phi, theta, psi)

    # Axles
    axle_x = np.array([[-l / 2, 0, 0], [l / 2, 0, 0]])
    axle_y = np.array([[0, -l / 2, 0], [0, l / 2, 0]])

    # Transform to world
    axle_x = x + R @ axle_x.T
    axle_y = y + R @ axle_y.T

    ax.plot(axle_x[0], axle_x[1], axle_x[2], 'r')
    ax.plot(axle_y[0], axle_y[1], axle_y[2], 'g')

anim = animation.FuncAnimation(fig, animate, frames=len(positions), interval=50)
anim.save("quadrotor_trajectory.mp4", writer="ffmpeg")
print("Saved animation as quadrotor_trajectory.mp4")
