import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from numba import jit, prange


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


def save_points(points, R, filename):
    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_title(f'N = {len(points)}')

    # Plot circle and points
    circle = Circle((0, 0), R, fill=False)
    ax.add_patch(circle)
    ax.scatter(points[:, 0], points[:, 1])

    # Set axes limits and aspect ratio
    ax.set_aspect('equal')
    ax.set_xlim(-1.1 * R, 1.1 * R)
    ax.set_ylim(-1.1 * R, 1.1 * R)

    # Save figure
    plt.savefig(filename, dpi=300)


def test_configuration(points, R):
    """
    Test the configuration of the points.

    :param points: Array of points.
    :param R: Radius of the circle.
    :return: 1 if the configuration is valid, 0 otherwise.
    """

    inner_points = 0
    for point in points:
        # Count the number of points within the circle
        if (point ** 2).sum() < 0.99 * R ** 2:
            inner_points += 1

    magic_number = np.array([12, 16, 17, 21, 22])
    expect_inner_points = np.searchsorted(magic_number, len(points), side='right')
    if inner_points == expect_inner_points:
        return 1
    else:
        return 0


@jit(nopython=True, cache=True, parallel=True)
def multi_simulate(N, R, Temp_max, Temp_min, alpha, iter_num, run_num, func):
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
        p, e = func(N, R, Temp_max, Temp_min, alpha, iter_num)

        # Save the energy and positions
        E.append(e[-1])
        ps.append(p)

    return E, ps

@jit(nopython=True, parallel=True, cache=True)
def multi_simulate_particle(N, R, Temp_max, Temp_min, alpha, iter_num, func):
    """
    Run the simulated annealing algorithm for different particles

    :param N: List containing number of particles
    :param R: Radius of the circle.
    :param Temp_max: Maximum temperature.
    :param Temp_min: Minimum temperature.
    :param alpha: Temperature reduction factor.
    :param iter_num: Number of iterations at each temperature.
    :return: Energy and positions of particles at the end of each run.
    """

    # Positions
    positions = []

    for i in prange(len(N)):
        # Set n
        n = N[i]

        # Simulate
        p, E = func(n, R, Temp_max, Temp_min, alpha, iter_num)

        # Add to list
        positions.append(p)

    return positions



def optimal_configuration(N, R, Temp_max, Temp_min, alpha, iter_num, simulations, func):
    # Simulate annealing several times
    E, ps = multi_simulate(N, R, Temp_max, Temp_min, alpha, iter_num, simulations, func)

    # Find the index of the minimum energy
    index = np.where(E == np.min(E))[0][0]

    # Return the final points and energy
    return ps[index], E[index]
