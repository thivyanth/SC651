import numpy as np
import matplotlib.pyplot as plt

# Number of time steps
n = 50
time_steps = np.arange(n)

# True initial state
x0_true = np.array([0, 10])  # Initial position 0m, velocity 10m/s

# State Transition Matrix (F)
dt = 1  # time step
F = np.array([[1, dt],
              [0,  1]])

# Process noise (Q) - representing random accelerations
process_noise_std = 0.5
Q = np.array([[0.25*dt**4, 0.5*dt**3],
              [0.5*dt**3,  dt**2]]) * process_noise_std**2

# Measurement Matrix (H)
H = np.array([1, 0]).reshape(1, 2)  # We only measure position

# Measurement noise (R)
measurement_noise_std = 10
R = np.array([measurement_noise_std**2])

# Initial State Estimate and Covariance
x0_estimate = np.array([0, 0])  # Estimated initial position and velocity
P0 = np.eye(2) * 1000  # Initial high uncertainty

# Kalman Filter Simulation
x_true = x0_true
x_estimate = x0_estimate
P = P0

# Arrays for storing the results
x_true_vals = np.zeros((n, 2))
x_estimate_vals = np.zeros((n, 2))
measurements = np.zeros(n)

for i in range(n):
    # Simulate the true system dynamics
    x_true = F @ x_true + np.random.normal(0, process_noise_std, size=2)
    z = H @ x_true + np.random.normal(0, measurement_noise_std)  # Measurement

    # Kalman Filter Prediction
    x_predict = F @ x_estimate
    P_predict = F @ P @ F.T + Q

    # Kalman Filter Update
    y = z - H @ x_predict
    S = H @ P_predict @ H.T + R
    K = P_predict @ H.T @ np.linalg.inv(S)
    x_estimate = x_predict + K @ y
    P = (np.eye(2) - K @ H) @ P_predict

    # Store results
    x_true_vals[i] = x_true
    x_estimate_vals[i] = x_estimate
    measurements[i] = z

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(time_steps, x_true_vals[:, 0], label='True Position', color='green')
plt.plot(time_steps, x_estimate_vals[:, 0], label='Estimated Position', color='red')
plt.scatter(time_steps, measurements, label='Measurements', color='blue', alpha=0.5)
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter Simulation for Train Tracking')
plt.legend()
plt.grid(True)
plt.show()

