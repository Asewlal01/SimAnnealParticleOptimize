import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def generate_points(n, R):
    """
    Generate n random points within a circle of radius R.

    :param n: Number of points to generate.
    :param R: Radius of the circle.
    :return: Array of points.
    """

    # Initialize array of points
    points = []

    # Generate n points
    for _ in range(n):
        # Generate polar coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        r = R * np.sqrt(np.random.uniform(0, 1))

        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Add point to array
        points.append((x, y))

    return np.array(points)


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


def calculate_pecific_heat(T, E):
    """
    Calculate the specific heat of the system.

    :param T: Temperature.
    :param E: Array of energies.
    :return: Specific heat.
    """

    return np.var(E) / T**2


def simulated_annealing(points, Temp_max, Temp_min, alpha, R, iter_num):
    """
    Simulated annealing algorithm to minimize the energy of the system, with charges within the circle.
    
    :param points: Array of points.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param R: Radius of the circle.
    :param iter_num: Number of iterations at each temperature.
    :return: Final points, energy, and specific heat.
    """

    # Initialize current temperature, points, and energy
    current_temp = Temp_max
    current_points = points.copy()
    current_energy = calculate_energy(current_points)

    # Initialize array of specific heats
    C = []

    # Run simulated annealing
    while current_temp > Temp_min:
        # Initialize array of energies
        E = []

        # Run iter_num iterations at current temperature
        for _ in range(iter_num):
            # Initialize array of perturbations
            perturbations = np.zeros_like(points)

            # Attempt to change each point
            for i in range(len(points)):
                # Generate a random perturbation to change the point
                new_points = current_points.copy()
                perturbation = np.random.normal(0, current_temp, size=2)
                new_points[i] += perturbation

                # If the new point is outside the circle, move it to the edge
                if np.linalg.norm(new_points[i]) > R:
                    new_points[i] *= R / np.linalg.norm(new_points[i])
                    perturbation = new_points[i] - current_points[i]

                # Save the perturbations if the new energy is lower or by a probability depending on the temperature
                new_energy = calculate_energy(new_points)
                energy_change = new_energy - current_energy
                if energy_change < 0 or np.exp(
                        -energy_change / current_temp) > np.random.rand():
                    perturbations[i] = perturbation

                # Save the new energy
                E.append(new_energy)

            # Update the points and energy
            current_points += perturbations
            current_energy = calculate_energy(current_points)

        # Save the specific heat
        C.append(calculate_pecific_heat(current_temp, E))

        # Decrease the temperature
        current_temp *= alpha

    return current_points, current_energy, C


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
    :return: Final points and energy.
    """

    # Generate initial random points
    initial_points = generate_points(N, R)

    # Initialize best points and minimum energy
    best_points = np.zeros_like(initial_points)
    min_energy = np.inf

    # Run simulated annealing several times
    for _ in range(run_num):
        final_points, final_energy, _ = simulated_annealing(
            initial_points, Temp_max, Temp_min, alpha, R, iter_num)

        # Update best points and minimum energy if necessary
        if final_energy < min_energy:
            min_energy = final_energy
            best_points = final_points

    # Return the final points and energy
    return best_points, min_energy


def show_points(points, R, dpi=100):
    """
    Plot the points within the circle.

    :param points: Array of points.
    :param R: Radius of the circle.
    :param dpi: DPI of the plot.
    :return: None.
    """

    # Initialize plot
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_title(f'N = {len(points)}')

    # Plot circle and points
    circle = Circle((0, 0), R, fill=False)
    ax.add_patch(circle)
    ax.scatter(points[:, 0], points[:, 1])

    # Set axes limits and aspect ratio
    ax.set_aspect('equal')
    ax.set_xlim(-1.1 * R, 1.1 * R)
    ax.set_ylim(-1.1 * R, 1.1 * R)

    plt.show()
