import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

t_span = (0, 10)
t_steps = np.linspace(t_span[0], t_span[1], 1000)

initial_conditions_spring = [0, 1] 
initial_conditions_pendulum = [np.pi/4, 0] 

def spring_mass_damper_system(t, y):
    return [y[1], -y[0] - y[1]]

def planar_pendulum_system(t, z):
    return [z[1], -np.sin(z[0])]

solution_spring = solve_ivp(spring_mass_damper_system, t_span, initial_conditions_spring, t_eval=t_steps)
solution_pendulum = solve_ivp(planar_pendulum_system, t_span, initial_conditions_pendulum, t_eval=t_steps)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(solution_spring.t, solution_spring.y[0])
plt.title("Spring-Mass-Damper System")
plt.xlabel("Time")
plt.ylabel("Displacement")

plt.subplot(1, 2, 2)
plt.plot(solution_pendulum.t, solution_pendulum.y[0])
plt.title("Planar Pendulum")
plt.xlabel("Time")
plt.ylabel("Angle θ")

plt.tight_layout()
plt.show()
