# Re-importing necessary libraries and redefining functions due to code execution state reset
import numpy as np
import matplotlib.pyplot as plt

dt_values = [0.01, 0.001, 0.0001]
t_end = 25
theta0, omega0 = np.pi / 4, 0  # initial conditions: 45 degrees, no initial angular velocity


# Redefining functions for Explicit Euler, Implicit Euler, and Symplectic Euler for the Planar Pendulum
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

# Correcting the error in the Symplectic Euler method implementation
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

# Perform simulations for each time step
results = {}
for dt in dt_values:
    t_explicit, y_explicit = explicit_euler(planar_pendulum, [theta0, omega0], dt, t_end)
    t_implicit, y_implicit = implicit_euler_pendulum([theta0, omega0], dt, t_end)
    t_symplectic, y_symplectic = symplectic_euler_pendulum([theta0, omega0], dt, t_end)

    results[dt] = {
        "explicit": (t_explicit, y_explicit),
        "implicit": (t_implicit, y_implicit),
        "symplectic": (t_symplectic, y_symplectic)
    }

# Plotting in a 4x2 grid
plt.figure(figsize=(18, 24))

plot_count = 1
for dt in dt_values:
    for method in ["explicit", "implicit", "symplectic"]:
        t, y = results[dt][method]

        # Method plot
        plt.subplot(4, 2, plot_count)
        plt.plot(t, y[:, 0], label=f'{method.capitalize()} Euler (dt={dt})')
        plt.title(f'{method.capitalize()} Euler Method (dt={dt})')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (θ)')
        plt.legend()
        plot_count += 1

        # Error plot (assuming small angle approximation for simplicity)
        # Note: The actual analytical solution for a non-linear pendulum is complex,
        # so we use the small angle approximation for the error calculation.
        analytical_theta = theta0 * np.cos(np.sqrt(9.81 / 1) * t)  # small angle approximation
        error = np.abs(y[:, 0] - analytical_theta)

        plt.subplot(4, 2, plot_count)
        plt.plot(t, error, label=f'Error in {method.capitalize()} Euler (dt={dt})')
        plt.title(f'Error in {method.capitalize()} Euler Method (dt={dt})')
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plot_count += 1

plt.tight_layout()
plt.show()

