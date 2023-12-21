import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def generate_points(n, R):
    """
    Generate n random points within a circle of radius R.

    :param n: Number of points to generate.
    :param R: Radius of the circle.
    :return: Array of points.
    """

    # Initialize array of points
    points = np.zeros((n, 2))

    # Generate n points
    for i in range(n):
        # Generate polar coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        r = R * np.sqrt(np.random.uniform(0, 1))

        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Add point to array
        points[i] = [x, y]

    return points


@jit(nopython=True, cache=True)
def calculate_energy(points, k=1):
    """
    Calculate the total energy of a system of charges.
    
    :param points: Array of points.
    :param k: Coulomb's constant.
    :return: Total energy of the system.
    """

    # Initialize energy
    energy = 0

    # Calculate energy
    for i in range(len(points)):
        # Calculate energy of point i with all other points
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            energy += k / distance

    return energy


@jit(nopython=True, cache=True)
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

    # Exponential sigma for perturbations
    b = np.log(N / 2) / (Temp_max - Temp_min)
    a = R / 2 * np.exp(-b * Temp_max)

    # Run simulated annealing
    while current_temp > Temp_min:
        # Compute the sigma for perturbations
        sigma = a * np.exp(b * current_temp)

        # Run iter_num iterations at current temperature
        for _ in range(iter_num):
            # Attempt to change each point
            for i in range(len(current_points)):
                # Generate a random perturbation to change the point
                new_points = current_points.copy()
                new_points[i] += np.random.normal(0, sigma, size=2)

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


@jit(nopython=True, cache=True, parallel=True)
def multi_simulate(N, R, Temp_max, Temp_min, alpha, iter_num, run_num):
    """
    Run the simulated annealing algorithm several times.
    
    :param N: Number of points.
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :param run_num: Number of times to run the algorithm.
    :return: Energy and positions of particles at the end of each run.
    """
    # Initialize arrays of energy and positions
    E, ps = [], []

    # Run simulated annealing several times
    for i in prange(run_num):
        # Run simulated annealing
        p, e = simulated_annealing(N, R, Temp_max, Temp_min, alpha, iter_num)

        # Save the energy and positions
        E.append(e)
        ps.append(p)

    return E, ps


def optimal_configuration(N, R, Temp_max, Temp_min, alpha, iter_num, run_num):
    """
    Run the simulated annealing algorithm several times to find the optimal configuration.

    :param N: Number of points.
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :param run_num: Number of times to run the algorithm.
    :return: Best points and energy.
    """
    # Simulate annealing several times
    E, ps = multi_simulate(N, R, Temp_max, Temp_min, alpha, iter_num, run_num)

    # Find the index of the minimum energy
    index = np.where(E == np.min(E))[0][0]

    # Return the final points and energy
    return ps[index], E[index]


def get_T_history(Temp_max, Temp_min, alpha):
    """
    Get the temperature history.
    
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :return: Temperature history.
    """

    # Initialize array of temperature
    T = [Temp_max]

    # Calculate temperature history
    while T[-1] > Temp_min:
        T.append(T[-1] * alpha)

    # Return the temperature history
    return np.array(T)


def get_E_T(N, R, Temp_max, Temp_min, alpha, iter_num, run_num):
    """
    Run the simulated annealing algorithm several times and return the energy at each temperature.
    
    :param N: Number of points.
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :param run_num: Number of times to run the algorithm.
    :return: Energy history under different temperature and temperature history.
    """

    # Initialize array of temperature and energy
    T = get_T_history(Temp_max, Temp_min, alpha)
    E = np.zeros((run_num, len(T)))

    # Run simulated annealing several times
    for i in range(run_num):
        points, energy_history = simulated_annealing(N, R, Temp_max, Temp_min, alpha, iter_num)
        E[i] = energy_history

    # Return energy history and temperature history
    return E, T
