import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


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
        if (point ** 2).sum() < 0.99 * R**2:
            inner_points += 1

    magic_number = np.array([12, 16, 17, 21, 22])
    expect_inner_points = np.searchsorted(magic_number, len(points), side='right')
    if inner_points == expect_inner_points:
        return 1
    else:
        return 0