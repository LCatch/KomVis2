'''
Usage:
python MonteCarloXY.py

'''

import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos
from matplotlib.animation import FuncAnimation
from cycler import cycler
import time
from numba import jit, float64

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
    # -1 * np.cos(- self.spins[(x+1)%self.N, y])

    summ = summ - np.cos(s0 - s1)
    summ = summ - np.cos(s0 - s2)
    summ = summ - np.cos(s0 - s3)
    summ = summ - np.cos(s0 - s4)

    return summ

class Box():
    def __init__(self,N=10, sweeps=100, T=1, seed=3823582, summary_file=None, save=True,
                 teq=1000):
        self.N = N #grid size 1D
        self.sweeps = sweeps #timesweeps
        self.T = T #temperature
        self.save = save

        if summary_file:
            with open(summary_file, "w") as file:
                file.write("")
            self.summary_file = summary_file
        else:
            self.summary_file = 'summary.txt'
        
        self.spins = np.zeros([N,N])
        self.X, self.Y = np.meshgrid(np.arange(0,self.N), np.arange(0, self.N))
        self.energies = np.zeros(sweeps+1)
        self.magnetizations = np.zeros([sweeps+1, 2])
        self.teq = teq
        # self.ti = 0     # index of last accepted step

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
        self.spins = rng.uniform(-np.pi, np.pi, [self.N,self.N])

    # def Hamiltonian(self, x,y, val):
    #     """
    #     Computes the local energy sum of interactions of the 4 nearest
    #     neighbouring spins, using periodic boundary conditions. 
    #     """
    #     summ = 0
    #     # # -1 * np.cos(- self.spins[(x+1)%self.N, y])

    #     summ -= 1 * np.cos(val - self.spins[(x+1)%self.N, y])
    #     summ -= 1 * np.cos(val - self.spins[(x-1)%self.N, y])
    #     summ -= 1 * np.cos(val - self.spins[x, (y+1)%self.N])
    #     summ -= 1 * np.cos(val - self.spins[x, (y-1)%self.N])
    #     return summ

    def perform_sweep(self, ti):
        dE_tot = 0
        dM_tot = np.array([0.0,0.0])

        for (x,y), spin_new, rnd in zip(np.random.choice(self.N, size=[self.N**2, 2]), 
                                        np.random.uniform(-np.pi, np.pi, self.N**2),
                                        np.random.rand(self.N**2)):
            spin_curr = self.spins[x,y]

            H1 = _Hamiltonian(spin_curr, 
                            self.spins[(x+1)%self.N, y], 
                            self.spins[(x-1)%self.N, y],
                            self.spins[x, (y+1)%self.N],
                            self.spins[x, (y-1)%self.N])
        
            H2 = _Hamiltonian(spin_new, 
                            self.spins[(x+1)%self.N, y], 
                            self.spins[(x-1)%self.N, y],
                            self.spins[x, (y+1)%self.N],
                            self.spins[x, (y-1)%self.N])

            dE = H2 - H1
            p = np.exp(-1/self.T * dE) 
            if (dE < 0) or (rnd < p):
                self.spins[x,y] = spin_new
                dM_tot[0] += np.cos(spin_new) - np.cos(spin_curr)
                dM_tot[1] += np.sin(spin_new) - np.sin(spin_curr)
                dE_tot += dE

        self.energies[ti+1] = self.energies[ti] + dE_tot
        self.magnetizations[ti+1] = self.magnetizations[ti] + dM_tot

    def autocorrelation(self, plot=False):
        teq = self.teq
        tmax = self.sweeps - teq
        self.tmax = tmax
        abs_m_perspin = self.abs_m_perspin
    
        chi = np.zeros(tmax)

        for t in range(0, tmax-1):
            _1 = abs_m_perspin[teq:tmax-t+teq]
            _2 = abs_m_perspin[teq+t:tmax+teq]

            chi[t] = (np.sum(_1 * _2) - np.sum(_1) * np.sum(_2)/(tmax - t)) / (tmax - t)
        tau = np.sum(chi[:np.argmax(chi<0)])/chi[0]

        with open('taus.txt', "a") as file:
            file.write(f"T: {self.T:.2f}, tau: {tau:.4f}\n")
        if plot:
            plt.figure()
            plt.plot(chi)
            plt.title(f'T = {self.T}, sweeps = {self.sweeps}')
            # plt.xlim(0,1000)
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
        total_E= 0
        for i in range(self.N):
            for j in range(self.N):
                total_E += _Hamiltonian(self.spins[i,j],
                            self.spins[(i+1)%self.N, j], 
                            self.spins[(i-1)%self.N, j],
                            self.spins[i, (j+1)%self.N],
                            self.spins[i, (j-1)%self.N])
        # print(-1*total_E/2)
        return -1*total_E/2
    
    def error_of(self, quantity):
        return np.sqrt(2 * self.tau * (np.mean(quantity ** 2) - np.mean(quantity) ** 2) / self.tmax)


    def plot_magnetization(self, ax=None, label = ""):
        save=False
        # abs_m_perspin = np.sqrt(np.sum(self.magnetizations ** 2, axis=1)) / (self.N ** 2)
        
        if not ax:
            plt.figure()
            ax = plt.gca()
            ax.set_title(f'N = {self.N}, sweeps = {self.sweeps}, T = {self.T}')
            save = True
        ax.plot(self.abs_m_perspin, label = label)
        ax.set_xlabel("Time [sweeps]")
        ax.set_ylabel("Magnetization per spin [n.u.]")
        ax.legend(title="Temperature [n.u.]", loc=1)
        
        if save:
            plt.savefig('magn_plot.png')

    def plot_energy(self, ax=None, label=""):
        save=False
        if not ax:
            plt.figure()
            ax = plt.gca()
            ax.set_title(f'N = {self.N}, sweeps = {self.sweeps}, T = {self.T}')
            save = True
        
        ax.plot(self.energies / (self.N ** 2), label=label)
        ax.set_xlabel("Time [sweeps]")
        ax.set_ylabel("Energy per spin [n.u.]")
        ax.legend(title="Temperature [n.u.]", loc=1)
        
        if save:
            plt.savefig(f'energy_plot.png') 

    def state(self):
        U = np.cos(self.spins)
        V = np.sin(self.spins)
        M = self.spins
        return U,V,M

    def plot_state(self, ti='max'):
        X, Y = np.meshgrid(np.arange(0,self.N), np.arange(0, self.N))
        U = np.cos(self.spins)
        V = np.sin(self.spins)
        M = self.spins
        plt.figure(figsize=[10,10])
        plt.title(f'System state, T={self.T}, time={ti} sweeps')
        # plt.gca().set_facecolor('black')
        plt.quiver(X, Y, U, V, M, pivot='mid', scale=1, scale_units='xy',
                headaxislength=10, headlength=10, headwidth=6)
        plt.savefig(f'states/state_{ti}.png')


    def magnetic_sussy(self):
        block_length = int(self.tau * 16)
        M_ = self.abs_m[self.teq : ((self.sweeps-self.teq) // block_length)*block_length + self.teq]
        M = np.reshape(M_, (-1, block_length))
        mag_sus = ((M ** 2).mean(axis=1) - M.mean(axis=1) ** 2) / (self.N**2 * self.T)
        
        with open(self.summary_file, "a") as file:
            file.write(f'no. Independent blocks = {len(M)}\n')
            file.write(f'chi_m = {np.mean(mag_sus):.4f} +/- {np.std(mag_sus):.4f}\n')
        
        print(rf'chi_m = {np.mean(mag_sus):.4f} +/- {np.std(mag_sus):.4f}')

    def specific_heat(self):
        block_length = int(self.tau * 16)
        E_ = self.energies[self.teq : ((self.sweeps-self.teq) // block_length)*block_length + self.teq]
        E = np.reshape(E_, (-1, block_length))
        spec_heat = ((E ** 2).mean(axis=1) - E.mean(axis=1) ** 2) / (self.N**2 * self.T**2)

        with open(self.summary_file, "a") as file:
            file.write(f'no. Independent blocks = {len(E)}\n')
            file.write(f'specific heat C = {np.mean(spec_heat):.4f} +/- {np.std(spec_heat):.4f}\n')
        
        print(rf'chi_m = {np.mean(spec_heat):.4f} +/- {np.std(spec_heat):.4f}')

    def finalize(self):

        self.autocorrelation(plot=False) #sets tau, necessary.

        self.abs_m_perspin_err = self.error_of(self.abs_m_perspin)
        self.e_perspin_err = self.error_of(self.e_perspin)

        self.e_perspin_avg = np.average(self.e_perspin)
        self.abs_m_perspin_avg = np.average(self.abs_m_perspin)

        with open(self.summary_file, "a") as file:
            file.write(f'\nT = {self.T}\n')
            file.write(f'tau = {self.tau}\n')
            file.write(f'abs m = {self.abs_m_perspin_avg:.4f} +/- {self.abs_m_perspin_err:.4f}\n')
            file.write(f'e = {self.e_perspin_avg:.4f} +/- {self.e_perspin_err:.4f}\n')

        print(f'abs m = {self.abs_m_perspin_avg:.4f} +/- {self.abs_m_perspin_err:.4f}')
        print(f'e = {self.e_perspin_avg:.4f} +/- {self.e_perspin_err:.4f}')

        self.magnetic_sussy()
        self.specific_heat()

    def save_run(self):
        '''
        Save the absolute magnetization and energy of a run in a txt file
        Default directory: saved_runs/
        Default naming convention: {T}_{N}_{sweeps}.txt
        '''
        dict = 'saved_runs'
        np.savetxt(f'{dict}/{self.T}_{self.N}_{self.sweeps}', np.vstack([self.abs_m, self.energies]))

    def load_run(self, T, N, sweeps):
        ''' Load file, need to specify everything '''
        self.abs_m, self.energies = np.loadtxt(f'{T}_{N}_{sweeps}')

    def run(self, finalize=True, plot_states=False):
        print(f'\033[91mrunning... T={self.T:.4f}\033[00m')

        start = time.perf_counter()
        for time_step in range(0, self.sweeps):
            self.perform_sweep(time_step)
            if ((time_step)%500==0):
                print(f'{time_step} sweeps done')
                if plot_states:
                    self.plot_state(ti=time_step)
        end = time.perf_counter()
        
        print(f'\033[91mdone! runtime = {(end-start):.0f} s\033[00m')

        self.abs_m = np.sqrt(np.sum(self.magnetizations ** 2, axis=1))
        self.abs_m_perspin = self.abs_m / (self.N ** 2)
        self.e_perspin = self.energies / (self.N ** 2)

        if self.save:
            self.save_run()

        if finalize:
            self.finalize()


def animated(box):
    fig = plt.figure(figsize=[6,6])
    plt.title(f'T = {box.T}')
    ax = plt.axes(xlim =(0, box.N), ylim =(0, box.N))
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
    
    anim.save('test.mp4', writer='ffmpeg')

    box.plot_state()
    box.plot_energy()
    box.plot_magnetization()

def normal(box):
    box.run()
    box.plot_state()
    box.plot_energy()
    box.plot_magnetization()
    box.autocorrelation()

def batch():
    N = 20
    sweeps = 500
    T = 0.5
    Nsims = 2

    # fig_E, axs_E = plt.subplots(Nsims, 1)
    # fig_M, axs_M = plt.subplots(Nsims, 1)


    for i in range(Nsims):
        box = Box(N=N, sweeps=sweeps, seed=975, T=T)
        box.run()

        # box.plot_energy(axs_E[i])
        # axs_E[i].set_title(f'Energy for simul {i}')
        # box.plot_magnetization(axs_M[i])
        # axs_M[i].set_title(f'Magnetization for simul {i}')

    # fig_E.savefig(f'energy_{T}_multi.png')
    # fig_M.savefig(f'magn_{T}_multi.png')

def batch_equilib():
    N = 50
    sweeps = 10000
    # Nsims = 2

    Ts = np.arange(0.5, 2.7, 0.2)

    fig_E, axs_E = plt.subplots(1, 1, figsize=[8,5])
    fig_M, axs_M = plt.subplots(1, 1, figsize=[8,5])

    with open('summary.txt', "w") as file:
        file.write(f"New batch, N={N}\n")

    colormap = plt.get_cmap('plasma')
    axs_E.set_prop_cycle(cycler(color=[colormap(i) for i in np.linspace(0,1,len(Ts))]))
    axs_M.set_prop_cycle(cycler(color=[colormap(i) for i in np.linspace(0,1,len(Ts))]))
    
    for T in Ts:
        box = Box(N=N, sweeps=sweeps, seed=0, T=T)
        box.run()

        box.plot_energy(axs_E, label=f'{T:.2f}')
        box.plot_magnetization(axs_M, label=f'{T:.2f}')
    
    axs_E.set_title(f'Energy per spin for a {N}x{N} grid')
    axs_M.set_title(f'Magnetization per spin for a {N}x{N} grid')

    fig_E.savefig(f'energy_batch.png')
    fig_M.savefig(f'magn_batch.png')

def batch_corrtime_error(T, sweeps=10000):
    N = 50
    sweeps = sweeps
    Nsims = 1

    # Ts = np.arange(0.5, 2.7, 0.2)
    # with open('taus.txt', "w") as file:
    #     file.write(f'sweeps = {sweeps}, N={N}\n')

    # fig_E, axs_E = plt.subplots(len(Ts), 1, figsize=[4,len(Ts)*4])
    # fig_M, axs_M = plt.subplots(len(Ts), 1, figsize=[4,len(Ts)*4])
    
    taus = []
    energies_perspin = []
    magn_perspin = []

    for i in range(Nsims):
        teq = 1000
        if T > 1.2:
            teq = 200
        box = Box(N=N, sweeps=sweeps, T=T, teq=teq)
        box.run(finalize=False)
        box.autocorrelation()
        taus.append(box.tau)

        # box.plot_energy(axs_E[i], label = f'Simul {j+1}')
        # axs_E[i].set_title(f'T = {T}')
        # axs_E[i].legend()          
        # box.plot_magnetization(axs_M[i], label = f'Simul {j+1}')
        # axs_M[i].set_title(f'T = {T}')
        # axs_M[i].legend()

    # fig_E.savefig(f'energymulti.png')
    # fig_M.savefig(f'magnmulti.png')

    print(taus)

    print(np.mean(taus))
    print(np.std(taus))

    # plt.figure()
    # plt.plot(Ts, taus)
    # plt.ylabel(r'Correlation time $\tau$')
    # plt.xlabel('Temperature')
    # plt.savefig('taus.png')

    # fig_E.savefig(f'energy.png')
    # fig_M.savefig(f'magn.png')

def main():
    # N = 20
    # sweeps = 5000
    # box = Box(N=N, sweeps=sweeps, seed=0, T=0.881, summary_file='summary.txt',
                # save=False, teq=500)
    # box.run(finalize=False, plot_states=True)
    # box.autocorrelation(plot=True)
    # print('tau = ', box.tau)
    # box.plot_state()
    # box.plot_magnetization()
    # box.plot_energy()
# 
    # batch_equilib()
    batch_corrtime_error(T=0.5, sweeps=30000)
    # animated(box)

    # box.total_magnetization()
    # box.total_energy()
    # box.plot_state()
    # box.plot_magnetization()
    # box.plot_energy()



    #box.plot_state
    #box.plot_energy


if __name__ == "__main__":
    main()