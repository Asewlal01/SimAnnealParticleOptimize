import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from simulations_jit.SimulatedAnnealing import *

@jit(nopython=True)
def force(p, i):
    """
    Function to calculate the direction of the force on a particle

    :param p: Positions of the particles
    :param i: Particle index
    :return: Force on particle i
    """
    f = np.zeros(2)
    for j in range(len(p)):
        if j != i:
            f += (p[i] - p[j]) / np.linalg.norm(p[i] - p[j]) ** 4
    return f

@jit(nopython=True)
def forcedPerturbation(i, R, current_points, step_length, current_energy, current_temp):
    new_points = current_points.copy()

    f = force(new_points, i)
    new_points[i] += f * np.random.uniform(0, step_length)

    # If the new point is outside the circle, move it to the edge
    if np.linalg.norm(new_points[i]) > R:
        new_points[i] *= R / np.linalg.norm(new_points[i])

    # Accept the new points if the new energy is lower or by a probability depending on the temperature
    new_energy = calculate_energy(new_points)
    energy_change = new_energy - current_energy
    if energy_change < 0 or np.exp(-energy_change / current_temp) > np.random.rand():
        current_points = new_points
        current_energy = new_energy

    return current_points, current_energy

@jit(nopython=True)
def simulated_annealing(N, R, Temp_max, Temp_min, alpha, iter_num, cooling_schedule="exponential"):
    """
    Simulated annealing algorithm to minimize the energy of the system, with charges within the circle.
    Update the points and energy immediately.

    :param N: Number of points.
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :param cooling_schedule: Cooling schedule. "exponential" or "linear".
    :return: Final points and history of energy.
    """

    # Initialize current temperature, points, and energy
    current_temp = Temp_max
    current_points = generate_points(N, R)
    current_energy = calculate_energy(current_points)

    # Initialize array of energy history
    E = [current_energy]

    step_length = R / N

    # Run simulated annealing
    while current_temp > Temp_min:
        # Run iter_num iterations at current temperature
        for i in range(iter_num):
            # Attempt to change each point
            for j in range(len(current_points)):
                if np.random.rand() < np.exp(current_temp - T_max):
                    current_points, current_energy = perturbation(
                        j, R, current_points, step_length, current_energy,current_temp)

                else:
                    current_points, current_energy = forcedPerturbation(
                        j, R, current_points, step_length, current_energy, current_temp)

        # Save the energy and temperature
        E.append(current_energy)

        # Decrease the temperature
        if cooling_schedule == "exponential":
            current_temp *= alpha
        elif cooling_schedule == "linear":
            current_temp -= alpha

    return current_points, E