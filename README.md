# HOPSO: Harmonic Oscillator based Particle Swarm Optimization

This repository contains the implementation of Harmonic Oscillator based Particle Swarm Optimization (HOPSO), a novel optimization algorithm introduced in the paper "Harmonic Oscillator based Particle Swarm Optimization" by Yury Chernyak, Ijaz Ahamed Mohammad, Nikolas Masnicak, Matej Pivoluska, and Martin Plesch.

## Paper Reference

This code is supplementary material to the paper:

**Harmonic Oscillator based Particle Swarm Optimization**  
*Authors:* Yury Chernyak, Ijaz Ahamed Mohammad, Nikolas Masnicak, Matej Pivoluska, and Martin Plesch  
*Link:* [https://arxiv.org/html/2410.08043v1](https://arxiv.org/html/2410.08043v1)

## Algorithm Overview

HOPSO extends the traditional Particle Swarm Optimization (PSO) algorithm by introducing the principles of Harmonic Oscillators. This physics-based approach adds the concept of energy to enable smoother and more controlled convergence during the optimization process.

Key features of HOPSO include:
- Integration of harmonic oscillator mechanics with swarm intelligence
- Improved control over particle velocities to prevent uncontrolled movements
- Better convergence properties compared to traditional PSO
- Enhanced ability to escape local minima

## Implementation

The main function `hopso()` implements the HOPSO algorithm with the following parameters:

- `cost_fn`: The objective function to be minimized
- `hp`: Hyperparameters for the algorithm
- `num_particles`: Number of particles in the swarm
- `runs`: Number of independent optimization runs
- `dimension`: Dimensionality of the search space
- `max_cut`: Maximum amplitude cut-off factor
- `e_min`, `vectors`, `velocities`, `vel_mag`, `gbest`, `amps`, `pos`: Output parameters to store results
- `max_iterations`: Maximum number of iterations (default: 1000)

## HOPSO with Periodicity for Quantum Systems

For quantum systems and other inherently periodic problems, use `hopso_periodicity.py` instead of the standard implementation. This version includes:

- **Periodic boundary handling**: Properly manages periodic search spaces using modular arithmetic
- **Immediate particle updates**: Each particle updates its attractor/amplitude/theta immediately after finding a better personal best
- **Enhanced convergence**: Swarm-wide updates when global best changes, ensuring better coordination
- **Particle management**: Includes mechanisms to handle invalid particle states during optimization

**When to use the periodicity version:**
- Quantum system optimization (e.g., quantum circuit parameters, phase optimization)
- Any optimization problem with periodic variables (angles, phases, etc.)
- Problems where the search space has periodic boundary conditions

The periodicity version has the same function signature as the standard HOPSO but with enhanced handling for periodic variables:

```python
from hopso_periodicity import hopso

# Same usage as standard HOPSO, but optimized for periodic systems
hopso(cost_fn, hp, num_particles, runs, dimension, max_cut, 
      e_min, vectors, velocities, vel_mag, gbest, max_iterations=500)
```

## Usage

To use the HOPSO optimizer, you need to:

1. Define your cost function
2. Set up the hyperparameters
3. Initialize the output variables
4. Call the `hopso()` function (or `hopso_periodicity.hopso()` for quantum/periodic systems)

Example:
```python
import numpy as np
from hopso import hopso

# Define cost function (example: Sphere function)
def sphere(x):
    return np.sum(x**2)

# Set hyperparameters [c1, c2, time_step, lambda]
hp = [2.0, 2.0, 0.1, 0.1]

# Initialize output variables
e_min = []
vectors = []
velocities = []
vel_mag = []
gbest = []
amps = []
pos = []

# Run HOPSO
hopso(sphere, hp, num_particles=30, runs=1, dimension=10, 
      max_cut=0.5, e_min=e_min, vectors=vectors, velocities=velocities, 
      vel_mag=vel_mag, gbest=gbest, amps=amps, pos=pos)

# Get the best solution
best_solution = vectors[0]
best_value = e_min[0]
```

## Requirements

- NumPy
- tqdm

## License

Please refer to the original paper for citation guidelines and licensing information.

## Acknowledgments

This implementation is based on the research work by the QAA group at the Institute of Physics, Slovak Academy of Sciences. More of their works and research may be found at https://qaa.sav.sk/
