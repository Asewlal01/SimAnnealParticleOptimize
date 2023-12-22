# Simulated Annealing Particle Optimization
This repository contains Python code used to find the most optimal configuration of charged particles confined
within a circle. This configuration has been found by using simulated annealing

# Dependencies
The following dependencies have to be installed:
- NumPy
- Matplotlib
- Numba (Optional)
Numba has been used to greatly decrease the computation time of the simulations. In case that the Numba implementations
not work, or is has not been installed, a non-Numba implementation has been added to this repository. All notebooks use
the Numba implementation, so they cannot be used if the Numba implementations also don't work.

# Structure
- DataAcquirement.ipynb: This notebook contains all the code used to obtain data from simulating and save them
  txt files to be used for the other notebook.
- DataVisualization.ipynb: This notebook is used to visualize all data obtained from the DataAcquirement notebook
- Analysis.py: This file constains analysis functions used to plot positions, get the optimal configuration,
  run multiple simulations
- simulations_jit: This folder constains functions used to simulate a N particle system confined within some cirlce with radius R.
  All .py files within this folder use the Numba implementation
- simulations_standard: This folder contains the same function as the simulations_jit, but is does not use a Numba implementation.
  This makes the code run a lot slower, and thus has not been implemented.
- expected.py: This file 
  

Other .py and .ipynb files within this repository are not used anymore, as they were methods that did not show
expected behaviour.
