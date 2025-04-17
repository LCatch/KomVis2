'''
Usage:
python MonteCarloXY.py

'''

import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos
from matplotlib.animation import FuncAnimation 

plt.rcParams['image.cmap'] = 'hsv'

class Box():
    def __init__(self,N=10, steps=100, T=1, seed=3823582):
        self.N = N #grid size 1D
        self.steps = steps #timesteps
        self.T = T #temperature

        
        self.spins = np.zeros([N,N])
        self.X, self.Y = np.meshgrid(np.arange(0,self.N), np.arange(0, self.N))
        self.energies = np.zeros(steps+1)
        self.magnetizations = np.zeros([steps+1, 2])
        # self.ti = 0     # index of last accepted step

        self.set_init_conditions(seed=seed)
        self.energies[0] = self.total_energy()
        self.magnetizations[0] = self.total_magnetization()


    def set_init_conditions(self, rnd=True, seed=3823582):
        # TODO: decide logical way to create init state, options:
            # random random seed
            # random set seed
            # absolute 0
        if seed==0:
            return
        if rnd:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=seed)
        self.spins = rng.uniform(-np.pi, np.pi, [self.N,self.N])


    # def inner_prod(self, si, sj):
    #     """
    #     Computes the energy contribution between two neighbouring spins, 
    #     i.e. the interaction energy.
    #     """
    #     return -1 * (np.cos(si)*np.cos(sj) + np.sin(si)*np.sin(sj))
    #     return -1 * np.cos(si-sj)


    def Hamiltonian(self, x,y, val):
        """
        Computes the local energy sum of interactions of the 4 nearest
        neighbouring spins, using periodic boundary conditions. 
        """
        summ = 0
        # -1 * np.cos(- self.spins[(x+1)%self.N, y])

        summ -= 1 * np.cos(val - self.spins[(x+1)%self.N, y])
        summ -= 1 * np.cos(val - self.spins[(x-1)%self.N, y])
        summ -= 1 * np.cos(val - self.spins[x, (y+1)%self.N])
        summ -= 1 * np.cos(val - self.spins[x, (y-1)%self.N])

        # summ += self.inner_prod(val, self.spins[(x+1)%self.N, y])
        # summ += self.inner_prod(val, self.spins[(x-1)%self.N, y])
        # summ += self.inner_prod(val, self.spins[x, (y+1)%self.N])
        # summ += self.inner_prod(val, self.spins[x, (y-1)%self.N])

        return summ

    def try_new_state(self):
        ''' Try 1 new position '''
        x,y = np.random.choice(self.N, size=2)
        spin_curr = self.spins[x,y]
        spin_new = np.random.uniform(-np.pi, np.pi, 1)
        H1 = self.Hamiltonian(x,y,spin_curr)
        H2 = self.Hamiltonian(x,y,spin_new)

        dE = H2 - H1
        p = np.exp(-1/self.T * dE) 

        if (H2 < H1) or (np.random.rand() < p):
            self.spins[x,y] = spin_new
            dM = [cos(spin_new) - cos(spin_curr), sin(spin_new) - sin(spin_curr)]
            # dM = [0,0]
            return dE, dM
        else:
            return 0, [0,0]

    def sweep(self, ti): 
        """
        Performs NxN Metropolis steps, so for every site in the grid.
        """
        dE_tot = 0
        dM_tot = np.array([0.0,0.0])
        for i in range(self.N ** 2):
            dE, dM = self.try_new_state()
            dE_tot += dE
            dM_tot += dM

        self.energies[ti+1] = self.energies[ti] + dE_tot
        self.magnetizations[ti+1] = self.magnetizations[ti] + dM_tot

    def autocorrelation(self, plot=False):
        teq = 40
        tmax = self.steps - teq
        self.tmax = tmax
        abs_m_perspin = self.abs_m_perspin
    
        chi = np.zeros(tmax)

        # abs_m_perspin = np.array([22, 24, 25, 25, 28, 29, 34, 37, 40, 44, 51, 48, 47, 50, 51])
        # teq=0
        # tmax=15

        for t in range(0, tmax-1):
            # print(teq, t, tmax)
            _1 = abs_m_perspin[teq:tmax-t+teq]
            _2 = abs_m_perspin[teq+t:tmax+teq]

            # print(_1, _2)
            # print(len(_1))
            # print(tmax - t)

            chi[t] = (np.sum(_1 * _2) - np.sum(_1) * np.sum(_2)/(tmax - t)) / (tmax - t)
            # chi[t] = (np.sum(_1 * _2) ) / (tmax - t)
            # print((_1 * _2).shape)

        tau = np.sum(chi[:np.argmax(chi<0)])/chi[0]
        # print(self.T, tau)
        # return tau

        with open('taus.txt', "a") as file:
            file.write(f"T: {self.T:.2f}, tau: {tau:.4f}\n")
        if plot:
            plt.figure()
            plt.plot(chi)
            plt.xlim(0,200)
            plt.savefig('autocorrelation.png')

        # return tau
        self.tau = tau
        # np.sum(_1 * _2)


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
                total_E+= self.Hamiltonian(i, j, self.spins[i,j])
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
            save = True
        ax.plot(self.abs_m_perspin, label = label)
        
        if save:
            plt.savefig('magn_plot.png')

    def plot_energy(self, ax=None, label=""):
        save=False
        if not ax:
            plt.figure()
            ax = plt.gca()
            save = True
        
        ax.plot(self.energies / (self.N ** 2), label = label)
        
        if save:
            plt.savefig(f'energy_plot.png') 

    def state(self):
        U = np.cos(self.spins)
        V = np.sin(self.spins)
        M = self.spins
        return U,V,M

    def plot_state(self):
        X, Y = np.meshgrid(np.arange(0,self.N), np.arange(0, self.N))
        U = np.cos(self.spins)
        V = np.sin(self.spins)
        M = self.spins
        plt.figure(figsize=[10,10])
        # plt.gca().set_facecolor('black')
        plt.quiver(X, Y, U, V, M, pivot='mid', scale=1, scale_units='xy',
                headaxislength=10, headlength=10, headwidth=6)
        plt.savefig('state.png')

    def finalize(self):
        self.abs_m = np.sqrt(np.sum(self.magnetizations ** 2, axis=1))
        self.abs_m_perspin = self.abs_m / (self.N ** 2)
        self.autocorrelation(plot=False) #sets tau, necessary.
        self.e_perspin = self.energies / (self.N ** 2)

        self.abs_m_perspin_err = self.error_of(self.abs_m_perspin)
        self.e_perspin_err = self.error_of(self.e_perspin)

        self.e_perspin_avg = np.average(self.e_perspin)
        self.abs_m_perspin_avg = np.average(self.abs_m_perspin)
        
        print(f'abs m = {self.abs_m_perspin_avg:.4f} +/- {self.abs_m_perspin_err:.4f}')
        print(f'e = {self.e_perspin_avg:.4f} +/- {self.e_perspin_err:.4f}')



    def run(self):
        print(f'\033[91mrunning... T={self.T}\033[00m')
        # while self.ti+1 < self.steps:
        for time_step in range(0, self.steps):
            self.sweep(time_step)
            if ((time_step)%50==0):
                print(f'{time_step} steps done')
            # self.state()
        print('\033[91mdone!\033[00m')
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
        box.sweep(ti)
        a, c, d = box.state()
        quiv.set_UVC(a,c,d) 
        
        return quiv,

    # update_every=5
    anim = FuncAnimation(fig, animate, 
                     frames=box.steps, interval=20, blit = True) 
    
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
    steps = 500
    T = 0.5
    Nsims = 2

    # fig_E, axs_E = plt.subplots(Nsims, 1)
    # fig_M, axs_M = plt.subplots(Nsims, 1)


    for i in range(Nsims):
        box = Box(N=N, steps=steps, seed=975, T=T)
        box.run()

        # box.plot_energy(axs_E[i])
        # axs_E[i].set_title(f'Energy for simul {i}')
        # box.plot_magnetization(axs_M[i])
        # axs_M[i].set_title(f'Magnetization for simul {i}')

    # fig_E.savefig(f'energy_{T}_multi.png')
    # fig_M.savefig(f'magn_{T}_multi.png')

def batch2():
    N = 20
    steps = 1000
    Nsims = 1

    Ts = np.arange(0.5, 2.5, 0.2)
    with open('taus.txt', "w") as file:
        file.write(f'steps = {steps}, N={N}\n')

    # fig_E, axs_E = plt.subplots(len(Ts), 1, figsize=[4,len(Ts)*4])
    # fig_M, axs_M = plt.subplots(len(Ts), 1, figsize=[4,len(Ts)*4])
    
    taus = []

    for j in range(Nsims):
        for i, T in enumerate(Ts):
            box = Box(N=N, steps=steps, T=T)
            box.run()
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

    plt.figure()
    plt.plot(Ts, taus)
    plt.ylabel(r'Correlation time $\tau$')
    plt.xlabel('Temperature')
    plt.savefig('taus.png')


    # fig_E.savefig(f'energy.png')
    # fig_M.savefig(f'magn.png')

def main():
    N = 20
    steps = 1000
    box = Box(N=N, steps=steps, seed=1, T=1.5)

    box.run()
    # tau = box.autocorrelation(plot=True)
    # print(tau)
    # box.plot_state()
    # box.plot_magnetization()
    # box.plot_energy()
# 
    # batch2()
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