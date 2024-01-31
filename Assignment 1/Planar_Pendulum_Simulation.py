
import numpy as np
import matplotlib.pyplot as plt

# Function Definitions
def explicit_euler(f, y0, dt, t_end):
    t = np.arange(0, t_end, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i - 1] + dt * f(y[i - 1], t[i - 1])
    return t, y

def implicit_euler_pendulum(y0, dt, t_end):
    t = np.arange(0, t_end, dt)
    y = np.zeros((len(t), 2))
    y[0] = y0
    for i in range(1, len(t)):
        theta, omega = y[i - 1]
        omega_new = omega + dt * (-np.sin(theta + dt * omega))
        theta_new = theta + dt * omega_new
        y[i] = np.array([theta_new, omega_new])
    return t, y

def symplectic_euler_pendulum(y0, dt, t_end):
    t = np.arange(0, t_end, dt)
    y = np.zeros((len(t), 2))
    y[0] = y0
    for i in range(1, len(t)):
        theta, omega = y[i - 1]
        omega_new = omega + dt * (-np.sin(theta))
        theta_new = theta + dt * omega
        y[i] = np.array([theta_new, omega_new])
    return t, y

def planar_pendulum(y, t):
    theta, omega = y
    dthetadt = omega
    domegadt = -np.sin(theta)
    return np.array([dthetadt, domegadt])

# Simulation parameters
dt_values = [0.01, 0.001, 0.0001]
t_end = 25
theta0, omega0 = np.pi / 4, 0  # initial conditions

# Create a figure with 3 subplots
plt.figure(figsize=(12, 18))
plt.rcParams.update({'font.size': 14}) 
# Iterate over each time step
for i, dt in enumerate(dt_values):
    # Perform simulations for each method
    t_explicit, y_explicit = explicit_euler(planar_pendulum, [theta0, omega0], dt, t_end)
    t_implicit, y_implicit = implicit_euler_pendulum([theta0, omega0], dt, t_end)
    t_symplectic, y_symplectic = symplectic_euler_pendulum([theta0, omega0], dt, t_end)

    # Select the subplot
    plt.subplot(3, 1, i + 1)

    # Plot each method
    plt.plot(t_explicit, y_explicit[:, 0], label=f'Explicit Euler (dt={dt})')
    plt.plot(t_implicit, y_implicit[:, 0], label=f'Implicit Euler (dt={dt})')
    plt.plot(t_symplectic, y_symplectic[:, 0], label=f'Symplectic Euler (dt={dt})')

    plt.title(f'Planar Pendulum Simulation with dt={dt}')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (θ)')
    plt.legend()

plt.tight_layout()
plt.show()