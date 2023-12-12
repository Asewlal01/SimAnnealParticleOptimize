
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
# Function to initialize N positions randomly inside a unit circle
def initialize(N):
    p = []
    while len(p) < N:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            p.append([x, y])
    return np.array(p)


@jit(nopython=True)
# Function to return a small perturbation
def perturb(sigma):
    k = np.random.normal(0, sigma)
    # if k > 0:
    #     return k
    # else:
    #     return k/10
    return  abs(np.random.normal(0, sigma))


@jit(nopython=True)
# Function to check whether particles are within a unit circle
def in_circle(p):
    return np.sum(p**2) <= 1


@jit(nopython=True)
# Function to calculate the energy of the system
def energy(p):
    E = 0
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            E += 1/np.linalg.norm(p[i] - p[j])
    return E


@jit(nopython=True)
# Function to calculate the force on a particle
def force(p, i):
    f = np.zeros(2)
    for j in range(len(p)):
        if j != i:
            f += (p[i] - p[j]) / np.linalg.norm(p[i] - p[j])**3
    return f


@jit(nopython=True)
# Function to induce one perturbation on the system
def perturb_one(p, i, sigma):
        dir = force(p, i)/np.linalg.norm(force(p, i)) # Calculate direction of force

        p[i] += dir*perturb(sigma)
        
        if not in_circle(p[i]): # Check if perturbation is valid
            p[i] *= 1/np.linalg.norm(p[i]) # If not, move particle back to the edge of the circle
        return p


@jit(nopython=True)
# Function to induce perturbations on the system
def perturb_system(p, sigma):
    for i in range(len(p)):
        p[i] += perturb_one(p, i, sigma)
    return p


@jit(nopython=True)
# Function to calculate the acceptance probability
def acceptance_probability(E_old, E_new, T):
    if E_new < E_old:
        return 1
    else:
        return np.exp(-(E_new - E_old) / T)



@jit(nopython=True)
# Function to find new position of partcles
def new_positions(p, E, T, sigma):
    p_new = p.copy()
    E_new = E
    
    # Perturb the system
    for i in range(len(p)):
        p_trial = perturb_one(p_new, i, sigma)
        E_trial = energy(p_trial)

        # Check if perturbation is accepted
        if acceptance_probability(E_new, E_trial, T) > np.random.uniform(0, 1):
            p_new = p_trial
            E_new = E_trial
            
    return p_new, E_new


# Function to plot the positions of the particles
def plot_positions(p):
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), 1, color='blue', fill=False)
    ax.add_artist(circle)
    ax.scatter(p[:,0], p[:,1], color='red', s=4)
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.axhline(y=0, color='blue')
    plt.axvline(x=0, color='blue')
    plt.title("Particle positions for N = " + str(len(p)))
    plt.show()


@jit(nopython=True)
# Function to run the annealing process
def annealing(N, T_max, T_min, cooling_schedule, no_iterations):
    # Initialize positions
    p = initialize(N)

    # Initialize energy
    E = energy(p)

    # Initialize temperature
    T = T_max

    # Annealing process
    while T > T_min:
        #print("T: ", T) # Print temperature

        # Initialize standard deviation of random normal perturbation
        sigma = T /T_max

        # Markov chain
        for _ in range(no_iterations):
            # Get new positions for each iteration
            p, E = new_positions(p, E, T, sigma)

        # Cool system
        T *= cooling_schedule

    return p, E   





