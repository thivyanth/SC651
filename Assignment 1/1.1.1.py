import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve



# Function for the Symplectic Euler method
def symplectic_euler_spring_mass_damper(y, dt):
    x, v = y
    v_new = v + dt * (-x - v)
    x_new = x + dt * v_new
    return np.array([x_new, v_new])

# Analytical solution function for Spring-Mass-Damper system
def analytical_solution_smd(t, x0, v0):
    return np.exp(-0.5 * t) * (x0 * np.cos(np.sqrt(3)/2 * t) + (v0 + 0.5 * x0) / (np.sqrt(3)/2) * np.sin(np.sqrt(3)/2 * t))

# Perform Symplectic Euler simulation
y_symplectic = np.zeros((len(t_explicit), 2))
y_symplectic[0] = y0

for i in range(1, len(t_explicit)):
    y_symplectic[i] = symplectic_euler_spring_mass_damper(y_symplectic[i - 1], dt)

# Analytical solution
x_analytical_corrected = analytical_solution_smd(t_analytical, x0, v0)

# Plotting in a 3x2 grid
plt.figure(figsize=(18, 12))

# Explicit Euler
plt.subplot(3, 2, 1)
plt.plot(t_explicit, y_explicit[:, 0], label='Explicit Euler')
plt.plot(t_analytical, x_analytical_corrected, label='Analytical Solution', linestyle='dashed')
plt.title('Explicit Euler vs Analytical Solution')
plt.xlabel('Time (s)')
plt.ylabel('Position (x)')
plt.legend()

# Error for Explicit Euler
plt.subplot(3, 2, 2)
plt.plot(t_explicit, np.abs(y_explicit[:, 0] - np.interp(t_explicit, t_analytical, x_analytical_corrected)))
plt.title('Error in Explicit Euler')
plt.xlabel('Time (s)')
plt.ylabel('Error')

# Implicit Euler
plt.subplot(3, 2, 3)
plt.plot(t_implicit, y_implicit[:, 0], label='Implicit Euler')
plt.plot(t_analytical, x_analytical_corrected, label='Analytical Solution', linestyle='dashed')
plt.title('Implicit Euler vs Analytical Solution')
plt.xlabel('Time (s)')
plt.ylabel('Position (x)')
plt.legend()

# Error for Implicit Euler
plt.subplot(3, 2, 4)
plt.plot(t_implicit, np.abs(y_implicit[:, 0] - np.interp(t_implicit, t_analytical, x_analytical_corrected)))
plt.title('Error in Implicit Euler')
plt.xlabel('Time (s)')
plt.ylabel('Error')

# Symplectic Euler
plt.subplot(3, 2, 5)
plt.plot(t_explicit, y_symplectic[:, 0], label='Symplectic Euler')
plt.plot(t_analytical, x_analytical_corrected, label='Analytical Solution', linestyle='dashed')
plt.title('Symplectic Euler vs Analytical Solution')
plt.xlabel('Time (s)')
plt.ylabel('Position (x)')
plt.legend()

# Error for Symplectic Euler
plt.subplot(3, 2, 6)
plt.plot(t_explicit, np.abs(y_symplectic[:, 0] - np.interp(t_explicit, t_analytical, x_analytical_corrected)))
plt.title('Error in Symplectic Euler')
plt.xlabel('Time (s)')
plt.ylabel('Error')

plt.tight_layout()
plt.show()

