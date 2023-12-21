import numpy as np
from scipy.optimize import minimize

def energy(positions):
    # Energy of system
    energy = 0

    # Loop through each particle
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            energy += 1 / np.linalg.norm(positions[i] - positions[j])

    return energy

def getPositions(params, N):

    R = params[:len(N)]
    theta = params[len(N):]

    # Array storing positions
    positions = [np.array([]), np.array([])]

    # Loop through rings
    for r, n, a in zip(R, N, theta):
        # Get angle for each particle in current ring
        angles = np.linspace(0, 2 * np.pi, n + 1)[1:] + a

        # Convert to cartesian
        x = r * np.cos(angles)
        y = r * np.sin(angles)

        # Add to list
        positions[0] = np.append(positions[0], x)
        positions[1] = np.append(positions[1], y)

    # Convert to array
    return np.array(positions).T

def system(params, N):
    # Get the positions
    positions = getPositions(params, N)

    # Return energy
    return energy(positions)

def optimalSystem(N):
    # Number of rings
    rings = len(N)

    # Create bounds for radii and angles
    radii = [[0,1]] * rings
    angles = [[0, 2 * np.pi]] * rings

    # Create initial guess for r and theta
    r = np.linspace(1, 0.1, rings)
    theta = np.zeros(rings)

    # Create bounds and initial guess
    bounds = np.array(radii + angles)
    init = np.concatenate([r, theta])


    # Minimize
    res = minimize(system, x0=init, method='Nelder-Mead', bounds=bounds, args=N)

    return res.fun, res.x

N = [10, 1]
res = optimalSystem(N)
print(res[0])
