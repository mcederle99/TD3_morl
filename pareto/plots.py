import numpy as np
import matplotlib.pyplot as plt

velocities = np.load('pareto_vel_prior.npy')
costs = np.load('pareto_cost_prior.npy')

plt.scatter(velocities, -costs)
plt.xlabel("Velocity")
plt.ylabel("-Control cost")
plt.grid()
plt.show()
