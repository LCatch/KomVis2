'''
MonteCarloXY.py
Authors: Liya Charlaganova, Charlotte Coone
Date: 01-05-2025

Required packages: ffmpeg, numba

Usage:
python MonteCarloXY.py

This code is used to simulate an XY spin model. It is evolved
using the Metropolis algorithm (Monte Carlo). The system is
allowed to reach thermal equlibrium, after which certain
quantities are calculated.

'''

import matplotlib.pyplot as plt
import numpy as np
# from math import sin, cos
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from cycler import cycler
import time
from numba import jit

plt.rcParams['image.cmap'] = 'hsv'

@jit(nopython=True)
def _Hamiltonian(s0, s1, s2, s3, s4):
    '''
    Calculate the totoal interaction energy between the spin s0, and its 
    neighbors(which need to be called explicitly), s1, s2, s3, s4

    Function outside the class to be used by jit, this increases the
    calculation speed by a factor ~2.
    '''
    summ = 0
    # -1 * np.cos(- self.spins[(x+1)%self.size, y])

    summ = summ - np.cos(s0 - s1)
    summ = summ - np.cos(s0 - s2)
    summ = summ - np.cos(s0 - s3)
    summ = summ - np.cos(s0 - s4)

    return summ

class Box():
    '''
    Class for simulating a two-dimensional XY spin model using a 
    Metropolis Monte Carlo method.
    '''
    def __init__(self, size=20, sweeps=5000, temp=1.5, seed=0, summary_file=None,
                 teq=500, forced_tau=None):
        '''
        Setup the simulation:
        N: size of the simulation (total number of spins: NxN)
        sweeps: length of simulation measured in sweeps 
            (1 sweep = NxN Metropolis steps)
        temp = temperature in natural units
        seed = initial state of simulation
            0 -> all spins aligned to the right
            anything else -> random configuration
        summary_file = file to save the run information to
        teq = equlibration time of simulation, this is set by hand and 
            not calculated during the run, so keep this in mind.
            teq = 1000 is enough for all simulations, it is advised to decrease
            teq for warm (temp>1.2) simulations
        forced_tau = only used for calculation of magnetic susceptibility and
            specific heat by using a shorter simulation, see batch_stat_quantities()
        '''
        self.size = size #grid size 1D
        self.sweeps = sweeps #timesweeps
        self.temp = temp #temperature

        if summary_file:
            with open(summary_file, "w") as file:
                file.write("")
            self.summary_file = summary_file
        else:
            self.summary_file = 'summary.txt'
        
        self.spins = np.zeros([size,size])
        self.X, self.Y = np.meshgrid(np.arange(0,self.size), np.arange(0, self.size))
        self.energies = np.zeros(sweeps+1)
        self.magnetizations = np.zeros([sweeps+1, 2])
        self.teq = teq
        self.forced_tau = forced_tau    # only used to calculate magnetic susceptibility
                                        # and specific heat

        self.set_init_conditions(seed=seed)
        self.energies[0] = self.total_energy()
        self.magnetizations[0] = self.total_magnetization()


    def set_init_conditions(self, rnd=True, seed=3823582):
        ''' 
        Set initial state of system, default to a state with all spins
        aligned along direction in 0.5*pi (necessary to have the animation
        work).
            '''
        if seed==0:
            self.spins.fill(0.5 * np.pi)
        if rnd:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=seed)
        self.spins = rng.uniform(-np.pi, np.pi, [self.size,self.size])

    def perform_sweep(self, ti):
        '''
        Perform a single sweep (1 time step) of the Metropolis algorithm.
        '''

        # Track the total change in energy and magnetization
        dE_tot = 0
        dM_tot = np.array([0.0,0.0])

        # Iterate over random spin positions (x,y), their new candidate
        # spin (spin_new) and a random value between 0 and 1 (rnd) 
        for (x,y), spin_new, rnd in zip(np.random.choice(self.size, size=[self.size**2, 2]), 
                                        np.random.uniform(-np.pi, np.pi, self.size**2),
                                        np.random.rand(self.size**2)):
            spin_curr = self.spins[x,y]

            # Calculate the interaction strenghts of the old and new spin
            H1 = _Hamiltonian(spin_curr, 
                            self.spins[(x+1)%self.size, y], 
                            self.spins[(x-1)%self.size, y],
                            self.spins[x, (y+1)%self.size],
                            self.spins[x, (y-1)%self.size])
        
            H2 = _Hamiltonian(spin_new, 
                            self.spins[(x+1)%self.size, y], 
                            self.spins[(x-1)%self.size, y],
                            self.spins[x, (y+1)%self.size],
                            self.spins[x, (y-1)%self.size])

            dE = H2 - H1
            p = np.exp(-1/self.temp * dE) 

            # Accept move if total energy decreases or if the chance of accepting
            # the new spin (p) is higher than the random value 'rnd'
            if (dE < 0) or (rnd < p):
                self.spins[x,y] = spin_new
                dM_tot[0] += np.cos(spin_new) - np.cos(spin_curr)
                dM_tot[1] += np.sin(spin_new) - np.sin(spin_curr)
                dE_tot += dE

        self.energies[ti+1] = self.energies[ti] + dE_tot
        self.magnetizations[ti+1] = self.magnetizations[ti] + dM_tot

    def autocorrelation(self, plot=False):
        '''
        Calculate the correlation time of a simulation based on the
        total absolute magnetization.
        'plot' decides whether to print the correlation function 
        (was used for debugging, and can be useful)
        '''

        # Set needed values to decrease self calls
        teq = self.teq
        tmax = self.sweeps - teq
        self.tmax = tmax
        abs_m_perspin = self.abs_m_perspin
    
        chi = np.zeros(tmax)

        # Slices needed to calculate the correlation function. These correspond
        # to the following in the report:
        # _1: m(t')
        # _2: m(t' + t)
        for t in range(0, tmax-1):
            _1 = abs_m_perspin[teq:tmax-t+teq]
            _2 = abs_m_perspin[teq+t:tmax+teq]

            chi[t] = (np.sum(_1 * _2) - np.sum(_1) * np.sum(_2)/(tmax - t)) / (tmax - t)
        tau = np.sum(chi[:np.argmax(chi<0)])/chi[0]

        # Separate file for saving the correlation times
        with open('taus.txt', "a") as file:
            file.write(f"T: {self.temp:.2f}, tau: {tau:.4f}\n")
        print(f'correlation time = {tau:.4f}')
        if plot:
            plt.figure()
            plt.plot(chi)
            plt.title(f'T = {self.temp}, sweeps = {self.sweeps}')
            plt.savefig('autocorrelation.png')

        self.tau = tau

    def total_magnetization(self):
        """
        Calculates total magnetization M of the system.
        """
        Mx = np.sum(np.cos(self.spins))
        My = np.sum(np.sin(self.spins))

        # M = np.sqrt(Mx*Mx + My*My)
        return Mx, My
    
    def total_energy(self):
        '''
        Calculate total energy E of the system by calculating the
        interaction energy for each spin with its neighbours. This leads to
        counting every interaction two times, so the total energy
        is halved at the end to arrive at the correct result.
        '''
        total_E= 0
        for i in range(self.size):
            for j in range(self.size):
                total_E += _Hamiltonian(self.spins[i,j],
                            self.spins[(i+1)%self.size, j], 
                            self.spins[(i-1)%self.size, j],
                            self.spins[i, (j+1)%self.size],
                            self.spins[i, (j-1)%self.size])
        # print(-1*total_E/2)
        return -1*total_E/2
    
    def error_of(self, quantity):
        '''
        Calculate the standard deviation of the mean for the array 'quantity'.
        '''
        return np.sqrt(2 * self.tau * (np.mean(quantity ** 2) - np.mean(quantity) ** 2) / self.tmax)


    def plot_magnetization(self, ax=None, label = ""):
        '''
        Plot the absolute magnetization (per spin) of a simulation over time. 
        By default a new plot is created, but if 'ax' is provided the data is 
        added to these axes.
        '''

        save=False

        if not ax:
            plt.figure()
            ax = plt.gca()
            ax.set_title(f'N = {self.size}, sweeps = {self.sweeps}, T = {self.temp}')
            save = True
        ax.plot(self.abs_m_perspin, label = label)
        ax.set_xlabel("Time [sweeps]")
        ax.set_ylabel("Magnetization per spin [n.u.]")
        ax.legend(title="Temperature [n.u.]", loc=1)
        
        if save:
            plt.savefig('magn_plot.png')

    def plot_energy(self, ax=None, label=""):
        '''
        Plot the energy (per spin) of a simulation over time (sweeps). 
        By default a new plot is created, but if 'ax' is provided the data is 
        added to these axes.
        '''

        save=False
        if not ax:
            plt.figure()
            ax = plt.gca()
            ax.set_title(f'N = {self.size}, sweeps = {self.sweeps}, T = {self.temp}')
            save = True
        
        ax.plot(self.energies / (self.size ** 2), label=label)
        ax.set_xlabel("Time [sweeps]")
        ax.set_ylabel("Energy per spin [n.u.]")
        ax.legend(title="Temperature [n.u.]", loc=1)
        
        if save:
            plt.savefig(f'energy_plot.png') 

    def state(self):
        '''
        Function that is used to create an animation, returns the state of 
        spins (U,V) and the positions (M)
        '''
        U = np.cos(self.spins)
        V = np.sin(self.spins)
        M = self.spins
        return U,V,M
    
    def angle_difference(self, a, b):
        '''
        Calculate angle difference.
        '''
        delta = a - b
        while delta > np.pi:
            delta -= 2*np.pi
        while delta < -np.pi:
            delta += 2*np.pi
        return delta

    def plot_state(self, filename="state.png"):
        '''
        Plot a the current state of the simulation
        
        '''

        X, Y = np.meshgrid(np.arange(0,self.size), np.arange(0, self.size))
        U = np.cos(self.spins)
        V = np.sin(self.spins)
        M = self.spins
        fig, ax = plt.subplots(figsize=[10,10])
        ax.quiver(X, Y, U, V, M, pivot='mid', scale=1, scale_units='xy',
                headaxislength=10, headlength=10, headwidth=6)
        plt.title(f'State of system, T={self.temp}')

        vortices = []
        anti_vort = []

        for i in range(self.size-1):
            for j in range(self.size-1):
                dtheta1 = self.angle_difference(self.spins[i+1, j], self.spins[i, j])
                dtheta2 = self.angle_difference(self.spins[i+1, j+1], self.spins[i+1, j])
                dtheta3 = self.angle_difference(self.spins[i, j+1], self.spins[i+1, j+1])
                dtheta4 = self.angle_difference(self.spins[i, j], self.spins[i, j+1])
                curl = dtheta1 + dtheta2 + dtheta3 + dtheta4

                if curl > np.pi:
                    vortices.append((i+0.5, j+0.5))
                elif curl < -np.pi:
                    anti_vort.append((i+0.5, j+0.5))

        # Plot circles around vortices and antivortices
        for (x, y) in vortices:
            circle = patches.Circle((x, y), radius=1.5, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)

        for (x, y) in anti_vort:
            circle = patches.Circle((x, y), radius=1.5, color='blue', fill=False, linewidth=2)
            ax.add_patch(circle)
        plt.savefig(filename)
        print(f"Amount of vortices per plotted state: {len(vortices)}; Amount of anti-vortices: {len(anti_vort)}")

        self.vortex_count = len(vortices)
        self.antivortex_count = len(anti_vort)

    def magnetic_sus(self):
        ''' Calculate the magnetic susceptibility of a simulation (haha sus)'''
        block_length = int(self.tau * 16)

        # Cut 'abs_m' array to have a whole number of blocks 'block_len' starting
        # from the equilibration time 'teq'
        M_ = self.abs_m[self.teq : ((self.sweeps-self.teq) // block_length)*block_length + self.teq]
        # Reshape to a 2D array
        M = np.reshape(M_, (-1, block_length))

        if len(M) == 0:
            print('Cannot calculate magnetic susceptibility, simulation too short')
            self.magn_sus_avg = 0
            self.magn_sus_std = 0
            return

        magn_sus = ((M ** 2).mean(axis=1) - M.mean(axis=1) ** 2) / (self.size**2 * self.temp)
        self.magn_sus_avg = np.mean(magn_sus)
        self.magn_sus_std = np.std(magn_sus)

        with open(self.summary_file, "a") as file:
            file.write(f'chi_m = {self.magn_sus_avg:.4f} +/- {self.magn_sus_std:.4f}\n')
        print(f'chi_m = {self.magn_sus_avg:.4f} +/- {self.magn_sus_std:.4f}')

    def specific_heat(self):
        ''' Calculate the specific heat of a simulation.'''
        block_length = int(self.tau * 16)

        # Cut 'energies' array to have a whole number of blocks 'block_len' starting
        # from the equilibration time 'teq'
        E_ = self.energies[self.teq : ((self.sweeps-self.teq) // block_length)*block_length + self.teq]
        # Reshape to 2D array
        E = np.reshape(E_, (-1, block_length))

        if len(E) == 0:
            print('Cannot calculate specific heat, simulation too short')
            self.spec_heat_avg = 0
            self.spec_heat_std = 0
            return

        spec_heat = ((E ** 2).mean(axis=1) - E.mean(axis=1) ** 2) / (self.size**2 * self.temp**2)
        self.spec_heat_avg = np.mean(spec_heat)
        self.spec_heat_std = np.std(spec_heat)

        with open(self.summary_file, "a") as file:
            file.write(f'no. Independent blocks = {len(E)}\n')
            file.write(f'specific heat C = {self.spec_heat_avg:.4f} +/- {self.spec_heat_std:.4f}\n')

        print(f'sped_heat = {self.spec_heat_avg:.4f} +/- {self.spec_heat_std:.4f}')

    def finalize(self):
        '''
        Function run after the simulation to calculate interesting values
        of the simulation and save these in the summary file:
        - correlation time 'tau'
        - energy per spin + error
        - absolute magnetization (per spin) + error(s)
        - magnetic susceptibility
        - specific heat
        '''

        self.autocorrelation(plot=False) # calculate correlation time of run

        self.abs_m_perspin_err = self.error_of(self.abs_m_perspin)
        self.e_perspin_err = self.error_of(self.e_perspin)

        self.e_perspin_avg = np.average(self.e_perspin)
        self.abs_m_perspin_avg = np.average(self.abs_m_perspin)

        with open(self.summary_file, "a") as file:
            file.write(f'\nT = {self.temp}\n')
            file.write(f'calculated tau = {self.tau}\n')
            if self.forced_tau:
                file.write(f'forced tau: {self.forced_tau}\n')
            file.write(f'no. Independent blocks = {(self.sweeps-self.teq)//int(self.tau*16)}\n')
            file.write(f'abs m = {self.abs_m_perspin_avg:.4f} +/- {self.abs_m_perspin_err:.4f}\n')
            file.write(f'e = {self.e_perspin_avg:.4f} +/- {self.e_perspin_err:.4f}\n')

        print(f'abs m = {self.abs_m_perspin_avg:.4f} +/- {self.abs_m_perspin_err:.4f}')
        print(f'e = {self.e_perspin_avg:.4f} +/- {self.e_perspin_err:.4f}')

        # If a 'tau' is provided in the definition of the instance, overwrite the 
        # calculated tau with this value (only used for calculation of 
        # magnetic susceptibility and specific heat).
        if self.forced_tau:
            self.tau = self.forced_tau
        self.magnetic_sus()
        self.specific_heat()

    def run_simulation(self, finalize=True, plot_states=False):
        '''
        Perform the simulation and calculate the absolute magnetization
        and energy (per spin). These can be plotted afterwards.
        If 'finalize' then further of the simulation is run. 
        If 'plot_states', the state of the simulation is saved every
        500 steps.
        '''

        print(f'\033[91mrunning... T={self.temp:.4f}\033[00m')

        start = time.perf_counter()
        for time_step in range(0, self.sweeps):
            self.perform_sweep(time_step)
            if ((time_step)%500==0):
                print(f'{time_step} sweeps done')
                if plot_states:
                    self.plot_state()
        end = time.perf_counter()
        
        print(f'\033[91mdone! runtime = {(end-start):.0f} s\033[00m')

        self.abs_m = np.sqrt(np.sum(self.magnetizations ** 2, axis=1))
        self.abs_m_perspin = self.abs_m / (self.size ** 2)
        self.e_perspin = self.energies / (self.size ** 2)

        if finalize:
            self.finalize()


def animated(box):
    '''
    Run a simulation of the Box() instance 'box' and create an animation.
    Warning: this takes longer than simply running the simulation.
    '''
    fig = plt.figure(figsize=[6,6])
    plt.title(f'T = {box.temp}')
    ax = plt.axes(xlim =(0, box.size), ylim =(0, box.size))
    quiv = ax.quiver([], [], [], [], [])
    a,c,d = box.state()
    quiv = ax.quiver(box.X, box.Y, a,c,d, pivot='mid', scale=1, scale_units='xy',
                headaxislength=10, headlength=10, headwidth=6) 

    def animate(ti):
        box.perform_sweep(ti)
        a, c, d = box.state()
        quiv.set_UVC(a,c,d) 
        
        return quiv,

    # update_every=5
    anim = FuncAnimation(fig, animate, 
                     frames=box.sweeps, interval=20, blit = True) 
    
    anim.save('simulation_animation.mp4', writer='ffmpeg')

    box.plot_state()

def basic_analysis(box):
    ''' 
    Run a simulation of the Box() instance 'box' and calculate the 
    basic results:
    - average magnetization and energy per spin
    - correlation time
    - magnetic susceptibility
    - specific heat
    Additionally, plot the final state of the system
    '''
    box.run_simulation(plot_states=True)
    box.plot_state()

def batch_stat_quantities(file='taus_summary.csv', size=50):
    '''
    Calculate two additional quantities of the simulation: specific heat
    and magnetic susceptibility using the correlation times stored in
    'taus_summary.csv'. These pre-calculated correlation times are used
    to run simulations that have a lenght of 10 blocks, with each block
    having 16*tau sweeps. This method is used in order to decrease the
    runtime of the function. 

    '''

    loaded_array = np.loadtxt(file, skiprows=1, delimiter=',')

    fig_magn_sus, ax_magn_sus = plt.subplots(1,1)
    fig_spec_heat, ax_spec_heat = plt.subplots(1,1)
    
    for temp, tau, _, _ in loaded_array:
        sweeps = int(10 * 16 * tau + 1000)
        print(sweeps)
        box = Box(size, sweeps, temp, forced_tau=tau)
        box.run_simulation()

        ax_spec_heat.errorbar(temp, box.spec_heat_avg, box.spec_heat_std,
                              c='steelblue', linestyle='', marker='o',
                              label=sweeps)
        ax_magn_sus.errorbar(temp, box.magn_sus_avg, box.magn_sus_std,
                              c='steelblue', linestyle='', marker='o',
                              label=sweeps)
        
    ax_spec_heat.set_title(f'Specific heat of {size}x{size} system')
    ax_spec_heat.set_xlabel('Temperature [n.u.]')
    ax_spec_heat.set_ylabel('Specific heat [n.u.]')
    fig_spec_heat.savefig('Specific_heat.png')
    
    ax_magn_sus.set_title(f'Magnetic susceptibility of {size}x{size} system')
    ax_magn_sus.set_xlabel('Temperature [n.u.]')
    ax_magn_sus.set_ylabel('Magnetic susceptibility [n.u.]')
    fig_magn_sus.savefig('Magnetic_susceptibility.png')


def batch_equilib(size=50):
    '''
    Run simulation for temperatures between T=0.5 and T=2.5 in steps of 0.2
    and plot the magnetization and energy per spin. Also calculate the
    averages.
    '''
    sweeps = 10000

    temps = np.arange(0.5, 2.7, 0.2)

    fig_E, axs_E = plt.subplots(1, 1, figsize=[8,5])
    fig_M, axs_M = plt.subplots(1, 1, figsize=[8,5])

    with open('summary.txt', "w") as file:
        file.write(f"New batch, N={size}\n")

    colormap = plt.get_cmap('plasma')
    axs_E.set_prop_cycle(cycler(color=[colormap(i) for i in np.linspace(0,1,len(temps))]))
    axs_M.set_prop_cycle(cycler(color=[colormap(i) for i in np.linspace(0,1,len(temps))]))
    
    for T in temps:
        box = Box(size=size, sweeps=sweeps, seed=0, T=T)
        box.run_simulation()

        box.plot_energy(axs_E, label=f'{T:.2f}')
        box.plot_magnetization(axs_M, label=f'{T:.2f}')
    
    axs_E.set_title(f'Energy per spin for a {size}x{size} grid')
    axs_M.set_title(f'Magnetization per spin for a {size}x{size} grid')

    fig_E.savefig(f'energy_batch.png')
    fig_M.savefig(f'magn_batch.png')

def plot_m_e_vsT():
    '''
    Calculate and plot the average magnetization and energy, per spin,
    with errors. A simulation is run for each temperature, and the 
    results are additionally saved in in absm_e_T2.txt.
    '''
    N = 50
    sweeps = 10000
    Ts = np.arange(0.5, 2.7, 0.2)

    avg_energies = []
    avg_magnetizations = []
    mag_err = []
    e_err=[]

    with open('absm_e_T2.txt', "w") as file:
        file.write(f"New batch, N={N}, runs per temperature = 1\n")

    for T in Ts:
        E_list = []
        M_list = []
        E_err_list =[]
        M_err_list = []

        teq = 1400
        if T > 1.1:
            teq = 400
        box = Box(size=N, sweeps=sweeps, temp=T, teq = teq)  # Use different seeds for variety
        box.run_simulation(plot_states=False)

        E_list.append(box.e_perspin_avg)
        M_list.append(box.abs_m_perspin_avg)
        E_err_list.append(box.e_perspin_err)
        M_err_list.append(box.abs_m_perspin_err)

        E_mean = np.mean(E_list)
        M_mean = np.mean(M_list)
        E_err = np.sqrt(np.mean(np.square(E_err_list)))
        M_err = np.sqrt(np.mean(np.square(M_err_list)))

        avg_energies.append(E_mean)
        avg_magnetizations.append(M_mean)
        mag_err.append(M_err)
        e_err.append(E_err)

        with open('absm_e_T2.txt', "a") as file:
            file.write(f"T={T:.2f} | E_avg={E_mean:.4f} | M_avg={M_mean:.4f}\n")

    # Plotting average energy vs temperature
    plt.figure(figsize=(8, 5))
    plt.errorbar(Ts, avg_energies, yerr = e_err, marker='o')
    plt.xlabel('Temperature [n.u.]')
    plt.ylabel('Energy per Spin [n.u.]')
    plt.title(f'Avg. Energy vs Temperature ({N}x{N})')
    #plt.grid(True)
    plt.savefig('energy_vs_temp2.png')

    # Plotting average magnetization vs temperature
    plt.figure(figsize=(8, 5))
    plt.errorbar(Ts, avg_magnetizations, yerr = mag_err, marker='o')
    plt.xlabel('Temperature [n.u.]')
    plt.ylabel('Magnetization per Spin [n.u.]')
    plt.title(f'Avg. Magnetization vs Temperature ({N}x{N})')
    #plt.grid(True)
    plt.savefig('magnetization_vs_temp2.png')


def plot_numbervortices(sweeps=20000):
    '''
    Calculate the average number of vortices ( = vortex + anitvortex) for
    a range of temperatures. 
    '''

    N = 50
    sweeps = sweeps
    Ts = np.arange(0.5, 2.5, 0.2)
    vortex_counts = []

    for T in Ts:
        vortex_total =[]
        teq = 1400
        if T > 1.1:
            teq = 300
        box = Box(size = N, sweeps = sweeps, temp = T, teq=teq)
        box.run_simulation(plot_states=True)

        vortex_total.append(box.vortex_count + box.antivortex_count)
    
    avg_vortices = np.mean(vortex_total)
    vortex_counts.append(avg_vortices)

    plt.plot(Ts, avg_vortices, label = f'{T:.2f}')
    plt.xlabel("Temperature [n.u.]")
    plt.ylabel("Number of (anti) vortices")
    plt.title("Total amount of (anti)vortices as a function of temperature")
    plt.savefig("vorticesvsT.png")

def batch_corrtime_error(temp, sweeps=10000, size=50):
    '''
    Run 10 simulations of length 'sweeps' at temperature T and
    calculate the average correlation time and error.
    '''

    Nsims = 10
    taus = []

    for i in range(Nsims):
        teq = 1000
        if temp > 1.2:
            teq = 200
        box = Box(size=size, sweeps=sweeps, temp=temp, teq=teq)
        box.run_simulation(finalize=False)
        box.autocorrelation()
        taus.append(box.tau)

    print(taus)

    print(np.mean(taus))
    print(np.std(taus))

def plot_taus(file='taus_summary.csv'):
    '''
    Plot the correlation times stored in 'taus_summary.csv' by default, 
    which has the following values in the 4 columns:
        Temperature of simulation T
        Correlation time tau
        Error of tau tau_std
        (Lenght of simulation sweeps)

    This .csv file has to be created manually, as writing a function that
    dynamically creates such a file without deleting old data and overwriting
    an old simulation with new values would take time that we don't have, sadly.
    '''

    loaded_array = np.loadtxt(file, skiprows=1, delimiter=',')
    temp, tau, tau_std, _ = loaded_array.T

    plt.figure()
    plt.errorbar(temp, tau, tau_std, linestyle='',
                 marker='o')
    plt.title('Correlation times of 50x50 system')
    plt.xlabel('T [n.u.]')
    plt.ylabel('Correlation time [sweeps]')
    plt.savefig('Correlation_times.png')


def main():
    box = Box(size=20, temp=1.7, sweeps=2500)

# Run default analysis of the defined box
    basic_analysis(box)

# Create an animation of the defined box
    # animated(box)

# Create an equilibration plot for a lattice of N=20 for temperatures
# between 0.5 and 2.5
    # batch_equilib(20)

# Calculate the correlation time using 10 runs for a temperature of 2.1
# and a simulation of length 5000 (sweeps)
    # batch_corrtime_error(2.1, sweeps=2000, size=10)

# Create the correlation times plot using the file 'taus_summary.csv'
# This file is pre-defined and needs to be created by hand by running
# calculating the correlation times. 
    # plot_taus()

# Calculate and plot absolute magnetization and energy per spin vs temperature, 
# averaged over the amount of simulations ran. Saved in a .txt file.
#    plot_m_e_vsT()

# Calculate vortices, and create average vortices plot
    # plot_numbervortices(sweeps=10000)

# Calculate two additional quantities of the simulation: specific heat
# and magnetic susceptibility using the correlation times stored in
# 'taus_summary.csv'. 
    # batch_stat_quantities(size=20)


if __name__ == "__main__":
    main()