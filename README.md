MonteCarloXY.py
Authors: Liya Charlaganova, Charlotte Coone
Date: 01-05-2025

Required packages: ffmpeg, numba

Usage:
python MonteCarloXY.py
------------------------
This python file simulates an XY model which consists of a NxN grid of spins
with values between 0 and 2pi. The goal is to let the system evolve to its
thermal equilibrium and then measure certain properties of the system, such 
as correlation time, magnetics susceptibility and specific heat, among others.
The system is evolved with the Metropolis algorithm by sequentially flipping 
single spins and either accepting or refuising this state based on the 
energy difference between the original and candidate state (see report
for more details).

By default a simulation is run for a box of size N=20, at a temperature
of 1.6. The default quantities are calculated for this run and then printed.
These results are also appended to a file called summary.txt.

Other options (in order as they appear in the file):
    - animate()
        Make an animation of the simulation :)
    - batch_equilib()
        Create the magnetization and energy plots which show the system
        equlibrating
    - batch_corrtime_error()
        Calculate the correlation time (with error) of a single temperature
        by running 10 simulations and taking the average correlation time.
        This can take a while
    - plot_taus()
        Plot the correlation times privided in taus_summary.csv (see below)
    - batch_stat_quantities()
        Calculate the specific heat and magnetic susceptibility using
        correlation times stored in taus_summary.csv (see below)
        This takes a while for a simulation of size 50.
------------------------
OTHER FILES:
On taus_summary.csv:
    This file contains the correlation times (taus) calculated by us in this 
    project.
    Since we did not have time to make a nice dynamic solution. This file was
    created by hand (from our own simulations!!). It is used to create the
    correlation times plot, and to calculate specific heat and magnetic 
    susceptibility for different temperatures. For the latter, a simulation
    length is chosen which has 10 blocks (1 block = 16 * tau). This method
    saves time since we run simulations which are as short as reasonably
    possible.
