import numpy as np


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


def simulated_annealing(N, R, Temp_max, Temp_min, alpha, iter_num, step_length):
    """
    Simulated annealing algorithm to minimize the energy of the system, with charges within the circle.
    
    :param N: Number of points.
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :param step_length: Maximum length of each perturbation.
    :return: Final points, energy, and specific heat.
    """

    # Initialize current temperature, points, and energy
    current_temp = Temp_max
    current_points = generate_points(N, R)
    current_energy = calculate_energy(current_points)

    # Initialize array of specific heats
    C = []

    # Run simulated annealing
    while current_temp > Temp_min:
        # Initialize array of attempted energies
        E = []

        # Run iter_num iterations at current temperature
        for _ in range(iter_num):
            # Initialize array of perturbations
            perturbations = np.zeros_like(current_points)

            # Attempt to change each point
            for i in range(len(current_points)):
                # Generate a random perturbation to change the point
                new_points = current_points.copy()
                perturbation = step_length * np.random.normal(0, current_temp, size=2)
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
        C.append(np.var(E) / current_temp**2)

        # Decrease the temperature
        current_temp *= alpha

    return current_points, current_energy, C


def simulated_annealing_immediately(N, R, Temp_max, Temp_min, alpha, iter_num, step_length):
    """
    Simulated annealing algorithm to minimize the energy of the system, with charges within the circle. Update the points and energy immediately.
    
    :param N: Number of points.
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :param step_length: Maximum length of each perturbation.
    :return: Final points, energy, and specific heat.
    """

    # Initialize current temperature, points, and energy
    current_temp = Temp_max
    current_points = generate_points(N, R)
    current_energy = calculate_energy(current_points)

    # Initialize array of specific heats
    C = []

    # Run simulated annealing
    while current_temp > Temp_min:
        # Initialize array of attempted energies
        E = []

        # Run iter_num iterations at current temperature
        for _ in range(iter_num):
            # Attempt to change each point
            for i in range(len(current_points)):
                # Generate a random perturbation to change the point
                new_points = current_points.copy()
                new_points[i] += step_length * np.random.normal(0, current_temp, size=2)

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

                # Save the new energy
                E.append(new_energy)

        # Save the specific heat
        C.append(np.var(E) / current_temp**2)

        # Decrease the temperature
        current_temp *= alpha

    return current_points, current_energy, C


def simulated_annealing_together(N, R, Temp_max, Temp_min, alpha, iter_num, step_length):
    """
    Simulated annealing algorithm to minimize the energy of the system, with charges within the circle. Perturb all points together.
    
    :param N: Number of points.
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :param step_length: Maximum length of each perturbation.
    :return: Final points, energy, and specific heat.
    """

    # Initialize current temperature, points, and energy
    current_temp = Temp_max
    current_points = generate_points(N, R)
    current_energy = calculate_energy(current_points)

    # Initialize array of specific heats
    C = []

    # Run simulated annealing
    while current_temp > Temp_min:
        # Initialize array of attempted energies
        E = []

        # Run iter_num iterations at current temperature
        for _ in range(iter_num):
            # Generate random perturbations to change all points
            new_points = current_points.copy()
            new_points += step_length * np.random.normal(0, current_temp, size=new_points.shape)

            # If the new points are outside the circle, move them to the edge
            for i in range(len(new_points)):
                if np.linalg.norm(new_points[i]) > R:
                    new_points[i] *= R / np.linalg.norm(new_points[i])
            
            # Accept the new points if the new energy is lower or by a probability depending on the temperature
            new_energy = calculate_energy(new_points)
            energy_change = new_energy - current_energy
            if energy_change < 0 or np.exp(
                    -energy_change / current_temp) > np.random.rand():
                current_points = new_points
                current_energy = new_energy
            
            # Save the new energy
            E.append(new_energy)

        # Save the specific heat
        C.append(np.var(E) / current_temp**2)

        # Decrease the temperature
        current_temp *= alpha

    return current_points, current_energy, C


def optimal_configuration(N, R, Temp_max, Temp_min, alpha, iter_num, run_num, step_length=1.):
    """
    Run the simulated annealing algorithm several times to find the optimal configuration.

    :param N: Number of points.
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :param run_num: Number of times to run the algorithm.
    :param step_length: Maximum length of each perturbation.
    :return: Final points and energy.
    """

    # Initialize minimum energy
    min_energy = np.inf

    # Initialize list of energies and points history
    energies = []
    points_history = []

    # Run simulated annealing several times
    for _ in range(run_num):
        points, energy, _ = simulated_annealing_immediately(N, R, Temp_max, Temp_min, alpha, iter_num, step_length)
        energies.append(energy)
        points_history.append(points)

    # Find the best points and energy
    best_points = points_history[np.argmin(energies)]
    min_energy = np.min(energies)

    # Return the final points and energy
    return best_points, min_energy
