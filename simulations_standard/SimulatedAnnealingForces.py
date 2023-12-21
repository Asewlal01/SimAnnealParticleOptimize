import numpy as np



def initialize(N):
    """
    Function to initialize N positions randomly inside a unit circle
    
    :param N: Number of particles
    :return: Initial positions
    """
    p = []
    while len(p) < N:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            p.append([x, y])
    return np.array(p)


def perturb(sigma):
    """
    Function to perturb the position of one particle
    
    :param sigma: Standard deviation of the normal distribution 
    :return: Random number from normal distribution
    """
    return abs(np.random.normal(0, sigma))


def in_circle(p):
    """
    Function to check whether particles are within a unit circle
    
    :param p: Positions of the particle 
    :return: True or false depending on whether the particle is within the unit circle
    """
    return np.sum(p**2) <= 1


def energy(p):
    """
    Function to calculate the energy of the system
    
    :param p: Positions of the particles
    :return: Energy of the system
    """
    E = 0
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            E += 1/np.linalg.norm(p[i] - p[j])
    return E


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
            f += (p[i] - p[j]) / np.linalg.norm(p[i] - p[j])**4
    return f


def random_perturb(p, i, sigma):
    """
    Function to induce one random perturbation on the system

    :param p: Positions of the particles
    :param i: Particle index
    :param sigma: Normal distribution standard deviation
    :return: Perturbed positions of the particle
    """
    p[i] += np.random.normal(0, sigma, 2)
        
    if not in_circle(p[i]): # Check if perturbation is valid
        p[i] *= 1/np.linalg.norm(p[i]) # If not, move particle back into the circle
    return p


def forced_perturb(p, i, sigma):
    """
    Function to induce one perturbation on the system in the direction of force

    :param p: Positions of the particles
    :param i: Particle index
    :param sigma: Normal distribution standard deviation
    :return: Perturbed positions of the particle
    """
    dir = force(p, i)/np.linalg.norm(force(p, i)) # Calculate direction of force

    p[i] += dir*perturb(sigma)
        
    if not in_circle(p[i]): # Check if perturbation is valid
        p[i] *= 1/np.linalg.norm(p[i]) # If not, move particle back into the circle
    return p


def perturb_system(p, sigma):
    """
    Function to induce perturbations on the system
    
    :param p: Positions of the particles
    :param sigma: Normal distribution standard deviation
    :return: Perturbed positions of the particles
    """
    for i in range(len(p)):
        p[i] += random_new_positions(p, i, sigma)
    return p


def acceptance_probability(E_old, E_new, T):
    """
    Function to calculate the acceptance probability
    
    :param E_old: Energy in previous state
    :param E_new: Energy in new state
    :param T: Cooling temperature
    :return: Probability of accepting the new state
    """
    if E_new < E_old:
        return 1
    else:
        return np.exp(-(E_new - E_old) / T)


def random_new_positions(p, E, T, sigma):
    """
    Function to find new position of particles based on random perturbation

    :param p: Positions of the particles
    :param E: Current energy of the system
    :param T: Cooling temperature
    :param sigma: Standard deviation of the normal distribution
    :return: New positions of the particles and the corresponding energy
    """
    p_new = p.copy()
    E_new = E
    
    # Perturb the system
    for i in range(len(p)):
        p_trial = random_perturb(p_new, i, sigma)
        E_trial = energy(p_trial)

        # Check if perturbation is accepted
        if acceptance_probability(E_new, E_trial, T) > np.random.uniform(0, 1):
            p_new = p_trial
            E_new = E_trial
            
    return p_new, E_new


def forced_new_positions(p, E, T, sigma):
    """
    Function to find new position of particles
    
    :param p: Positions of the particles
    :param E: Current energy of the system
    :param T: Cooling temperature
    :param sigma: Standard deviation of the normal distribution
    :return: New positions of the particles and the corresponding energy
    """
    p_new = p.copy()
    E_new = E

    # Perturb the system
    for i in range(len(p)):
        p_trial = random_new_positions(p_new, i, sigma)
        E_trial = energy(p_trial)

        # Check if perturbation is accepted
        if acceptance_probability(E_new, E_trial, T) > np.random.uniform(0, 1):
            p_new = p_trial
            E_new = E_trial

    return p_new, E_new


def annealing(N, T_max, T_min, cooling_schedule, no_iterations):
    """
    Function to run the annealing process
    
    :param N: Number of particles
    :param T_max: Maximum temperature
    :param T_min: Minimum temperature
    :param cooling_schedule: Amount of cooling per iteration
    :param no_iterations: Number of iterations per temperature
    :return: Positions and energies of the system per temperature
    """
    # Initialize positions
    p = initialize(N)

    # Initialize energy
    E = energy(p)

    # Initialize temperature
    T = T_max

    # Initialize lists to store minima per temperature
    E_min_per_temp = [E]
    p_min_per_temp = [p]

    # Annealing process
    while T > T_min:

        # Initialize standard deviation of random normal perturbation
        sigma = T

        # Markov chain
        for k in range(no_iterations):
            # Inducing some random perturbations
            if k == no_iterations/10:
                p, E = random_new_positions(p, E, T, sigma)
            else:
            # Get new positions for each iteration
                p, E = forced_new_positions(p, E, T, sigma)

        # Cool system
        T *= cooling_schedule

    return p, E
