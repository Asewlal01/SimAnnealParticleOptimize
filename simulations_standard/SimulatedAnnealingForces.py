import numpy as np
from simulations_jit.SimulatedAnnealing import *



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



def simulated_annealing_forces(N, R, Temp_max, Temp_min, alpha, iter_num, cooling_schedule="exponential"):
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

    # Exponential sigma for perturbations
    b = np.log(N / 2) / (Temp_max - Temp_min)
    a = R / 2 * np.exp(-b * Temp_max)

    # Linear variables for probability of forced perturbation
    c = 0.99 / np.log(Temp_max / Temp_min)
    d = 1 - c * np.log(Temp_max)

    # Run simulated annealing
    while current_temp > Temp_min:
        # Calculate sigma
        sigma = a * current_temp + b

        # Calculate probability
        p = c * np.log(current_temp) + d

        # Run iter_num iterations at current temperature
        for _ in range(iter_num):

            # Attempt to change each point
            for i in range(len(current_points)):
                # Use random perturbation to change the point
                if np.random.rand() < p:
                    # Generate a random perturbation to change the point
                    new_points = current_points.copy()
                    new_points[i] += np.random.normal(0, sigma, size=2)

                # Forced perturbation
                else:
                    # Get force
                    f = force(current_points, i)

                    # Generate a forced perturbation to change the point
                    new_points = current_points.copy()
                    new_points[i] += f * np.random.uniform(0, sigma)

                # If the new point is outside the circle, move it to the edge
                if np.linalg.norm(new_points[i]) > R:
                    new_points[i] *= R / np.linalg.norm(new_points[i])

                # Accept the new points if the new energy is lower or by a probability depending on the temperature
                new_energy = calculate_energy(new_points)
                energy_change = new_energy - current_energy
                if energy_change < 0 or np.exp(
                        -energy_change / current_temp) > np.random.rand():
                    current_points = new_points
                    current_energy = new_energy

        # Save the energy and temperature
        E.append(current_energy)

        # Decrease the temperature
        if cooling_schedule == "exponential":
            current_temp *= alpha
        elif cooling_schedule == "linear":
            current_temp -= alpha

    return current_points, E
