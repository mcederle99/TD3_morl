import numpy as np
import matplotlib.pyplot as plt

velocities = np.load('pareto_vel_prova.npy')
costs = np.load('pareto_cost_prova.npy')

plt.scatter(-velocities, costs)
plt.show()
