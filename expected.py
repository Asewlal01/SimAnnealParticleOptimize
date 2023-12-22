import numpy as np
from scipy.optimize import brute


def energy(positions):
    # Energy of system
    energy = 0

    # Loop through each particle
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            energy += 1 / np.linalg.norm(positions[i] - positions[j])

    return energy


def getPositions(R, N):
    # Array storing positions
    positions = [np.array([]), np.array([])]

    # Loop through rings
    for r, n in zip(R, N):
        # Get angle for each particle in current ring
        angles = np.linspace(0, 2 * np.pi, n + 1)[1:]

        # Convert to cartesian
        x = r * np.cos(angles)
        y = r * np.sin(angles)

        # Add to list
        positions[0] = np.append(positions[0], x)
        positions[1] = np.append(positions[1], y)

    # Convert to array
    return np.array(positions).T


def system(params, *N):
    # Get the positions
    positions = getPositions(params, N)

    # Return energy
    return energy(positions)


def optimalSystem(N):
    # Number of rings
    rings = len(N)

    # Create bounds for radii and angles
    radii = [[0, 1] for _ in range(rings)]
    radii = tuple(radii)

    # Minimize
    res = brute(system, ranges=radii, args=tuple(N), Ns=20, finish=None, full_output=True)

    return res[0], res[1]