'''
Usage:
python MonteCarloXY.py

'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation 

class Box():
    def __init__(self,N=10, steps=100, T=1):
        self.N = N #grid size 1D
        self.steps = steps #timesteps
        self.T = T #temperature

        self.spins = np.random.uniform(-np.pi, np.pi, [N,N]) #initializing spin states on grid
        # self.spins = np.zeros([N,N])
        self.X, self.Y = np.meshgrid(np.arange(0,self.N), np.arange(0, self.N)) #2D grid lattice NxN
        self.energies = np.zeros(steps) #to store energies of the system at each t-step
        self.ti = 0     # index of last accepted step

    def inner_prod(self, si, sj):
        """
        Computes the energy contribution between two neighbouring spins, 
        i.e. the interaction energy.
        """
        return -1 * (np.cos(si)*np.cos(sj) + np.sin(si)*np.sin(sj))

    def Hamiltonian(self, x,y,val):
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

    # def get_state_or_smt


    def metropolis_1(self):
        """
        Computes the Metropolis algorithm for one position on the grid. 
        """
        ti = self.ti
        x,y = np.random.choice(self.N, size=2) #picks random x,y (site) from grid
        curr = self.spins[x,y] #spins on the random grid point
        new = np.random.uniform(-np.pi, np.pi, 1) #new spin state
        H1 = self.Hamiltonian(x,y,curr) #Hamiltonians of current and new spin states
        H2 = self.Hamiltonian(x,y,new)

        dE = H2 - H1 #Difference in the energies of 2 systems 
        # print(dE)

        # add acceptance logic
        p = np.exp(-1/self.T * dE) 
        # print(p)
        # p = 2

        if H2 < H1:
            # print('New E is lower, accept the move!')
            self.spins[x,y] = new
            self.energies[ti+1] = self.energies[ti] + dE
            self.ti += 1 
            
        elif np.random.rand() < p: 
            # print('New E is higher, accept with certain probability!')
            self.spins[x,y] = new
            self.energies[ti+1] = self.energies[ti] + dE
            self.ti += 1 

    def total_magnetization(self):
        """
        Calculates total magnetization M of the system.
        """
        Mx = np.sum(np.cos(self.spins))
        My = np.sum(np.sin(self.spins))

        M = np.sqrt(Mx**2 + My**2)
        return M
    
    def plot_magnetization(self):
        m = self.total_magnetization()/(self.N **2) #magnetization per spin

        plt.figure()
        #plt.plot(, self.M)


    def total_energy(self):
        total_E= 0
        for i in range(self.N):
            for j in range(self.N):
                total_E+= self.Hamiltonian(i, j, self.spins[i,j])
        print(-1*total_E/2)
        return -1*total_E/2

    def sweep(self): 
        """
        Performs NxN Metropolis steps, so for every site in the grid.
        """
        for i in range(self.N ** 2):
            self.metropolis_1()

    def plot_energy(self):
        plt.figure()
        plt.plot(self.energies)
        plt.savefig('energy2.png')

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
        while self.ti+1 < self.steps:
        # for time_step in range(1, self.steps):
            self.metropolis_1()
            self.state()

def animated(box):
    fig = plt.figure()
    ax = plt.axes(xlim =(0, box.N), ylim =(0, box.N))
    quiv = ax.quiver([], [], [], [], [])
    a,c,d = box.state()
    quiv = ax.quiver(box.X, box.Y, a,c,d, pivot='mid') 

    def animate(i):
        # print(i)
        box.sweep
        update_every = 5
        

        old_ti = box.ti
        # print(i)
        if box.ti+1 >= box.steps:
            return quiv,
        while box.ti == old_ti:
            box.metropolis_1()
            # box.state()
            # print('cur ti: ', box.ti)
            # print('old ti: ', old_ti)

        # print('exit while')

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
    steps = 1000
    box = Box(N=N, steps=steps)
    
    normal(box)

    box.total_magnetization()
    box.total_energy()
    box.plot_state()


    #box.plot_state
    #box.plot_energy
    #animated(box)
    #normal(box)


if __name__ == "__main__":
    main()