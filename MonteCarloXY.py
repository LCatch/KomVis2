'''
Usage:
python MonteCarloXY.py

'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation 

class Box():
    def __init__(self,N=10, steps=100, T=1, seed=3823582):
        self.N = N #grid size 1D
        self.steps = steps #timesteps
        self.T = T #temperature

        
        # self.spins = np.zeros([N,N])
        self.X, self.Y = np.meshgrid(np.arange(0,self.N), np.arange(0, self.N)) #2D grid lattice NxN
        self.energies = np.zeros(steps) #to store energies of the system at each t-step
        self.ti = 0     # index of last accepted step

        self.set_init_conditions(seed=seed)


    def set_init_conditions(self, rnd=True, abs_zero=False, seed=3823582):
        # TODO: decide logical way to create init state, options:
            # random random seed
            # random set seed
            # absolute 0
        if abs_zero:
            return
        if rnd:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=seed)
        self.spins = rng.uniform(-np.pi, np.pi, [self.N,self.N])


    def inner_prod(self, si, sj):
        """
        Computes the energy contribution between two neighbouring spins, 
        i.e. the interaction energy.
        """
        return -1 * (np.cos(si)*np.cos(sj) + np.sin(si)*np.sin(sj))

    def Hamiltonian(self, x,y, val):
        """
        Computes the local energy sum of interactions of the 4 nearest
        neighbouring spins, using periodic boundary conditions. 
        """
        summ = 0
        summ += self.inner_prod(val, self.spins[(x+1)%self.N, y])
        summ += self.inner_prod(val, self.spins[(x-1)%self.N, y])
        summ += self.inner_prod(val, self.spins[x, (y+1)%self.N])
        summ += self.inner_prod(val, self.spins[x, (y-1)%self.N])

        return summ

    def try_new_state(self):
        ''' Try 1 new position '''
        x,y = np.random.choice(self.N, size=2)
        curr = self.spins[x,y]
        new = np.random.uniform(-np.pi, np.pi, 1)
        H1 = self.Hamiltonian(x,y,curr)
        H2 = self.Hamiltonian(x,y,new)

        dE = H2 - H1
        p = np.exp(-1/self.T * dE) 

        if (H2 < H1) or (np.random.rand() < p):
            return new, (x,y), dE
        else:
            return False, False, False

    def step_metropolis(self):
        '''
        Perform 1 Metropolis step: try new states until one satifsies 
        requirements.
        '''
        ti = self.ti

        new_state = False
        while not new_state:
            new_state, pos, dE = self.try_new_state()

        self.spins[pos] = new_state
        self.energies[ti+1] = self.energies[ti] + dE
        self.ti += 1

    def magnetization(self):
        pass

    def total_energy(self):
        pass

    def sweep(self): 
        """
        Performs NxN Metropolis steps, so for every site in the grid.
        """
        for i in range(self.N ** 2):
            self.step_metropolis()

    def plot_energy(self):
        plt.figure()
        plt.plot(self.energies)
        plt.savefig('energy.png')

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
        plt.figure()
        plt.quiver(X, Y, U, V, M, pivot='mid')
        plt.savefig('state.png')

    def run(self):
        print('running...')
        # while self.ti+1 < self.steps:
        for time_step in range(1, self.steps):
            self.step_metropolis()
            self.state()

def animated(box):
    fig = plt.figure()
    ax = plt.axes(xlim =(0, box.N), ylim =(0, box.N))
    quiv = ax.quiver([], [], [], [], [])
    a,c,d = box.state()
    quiv = ax.quiver(box.X, box.Y, a,c,d, pivot='mid') 

    def animate(i):
        box.sweep
        update_every = 5

        if box.ti+1 >= box.steps:
            pass
        else:
            box.step_metropolis()
            a, c, d = box.state()
            quiv.set_UVC(a,c,d) 
        
        return quiv,

    # update_every=5
    anim = FuncAnimation(fig, animate, 
                     frames=box.steps, interval=10, blit = True) 
    
    anim.save('test2.mp4', writer='ffmpeg')

    box.plot_state()
    box.plot_energy()

def normal(box):
    box.run()
    box.plot_state()
    box.plot_energy()

def main():
    N = 20
    steps = 5000
    box = Box(N=N, steps=steps, T=2.5, seed=12)

    #animated(box)
    normal(box)


if __name__ == "__main__":
    main()