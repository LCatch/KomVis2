'''
Usage:
python MonteCarloXY.py

'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation 

class Box():
    def __init__(self,N=10, steps=100, T=1):
        self.N = N
        self.steps = steps
        self.T = T

        self.spins = np.random.uniform(-np.pi, np.pi, [N,N])
        # self.spins = np.zeros([N,N])
        self.X, self.Y = np.meshgrid(np.arange(0,self.N), np.arange(0, self.N))
        self.energies = np.zeros(steps)

    def inner_prod(self, si, sj):
        return -1 * np.cos(si)*np.cos(sj) + np.sin(si)*np.sin(sj)

    def Hamiltonian(self, x,y,val):
        summ = 0
        summ += self.inner_prod(val, self.spins[(x+1)%self.N, y])
        summ += self.inner_prod(val, self.spins[(x-1)%self.N, y])
        summ += self.inner_prod(val, self.spins[x, (y+1)%self.N])
        summ += self.inner_prod(val, self.spins[x, (y-1)%self.N])

        return summ

    # def get_state_or_smt

    def bleh(self, ti):
        x,y = np.random.choice(self.N, size=2)
        curr = self.spins[x,y]
        new = np.random.uniform(-np.pi, np.pi, 1)
        H1 = self.Hamiltonian(x,y,curr)
        H2 = self.Hamiltonian(x,y,new)

        dE = H2 - H1
        self.energies[ti] = self.energies[ti-1] + dE

        # add acceptance logic
        p = np.exp(-1/self.T * dE)
        # p = 2

        if H2 < H1:
            self.spins[x,y] = new
        elif np.random.rand() > p:
            self.spins[x,y] = new

    def sweep(self):
        for i in range(self.N ** 2):
            self.bleh()

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
        plt.quiver(X, Y, U, V, M)
        plt.savefig('state.png')

    def run(self):
        for ti in range(1, self.steps):
            self.bleh(ti)
            self.state

def animated(box):
    fig = plt.figure()
    ax = plt.axes(xlim =(0, box.N), ylim =(0, box.N))
    quiv = ax.quiver([], [], [], [], [])
    a,c,d = box.state()
    quiv = ax.quiver(box.X, box.Y, a,c,d, pivot='mid') 

    def animate(i):
        # print(i)
        box.sweep()
        for j in range(update_every):
            box.bleh(i*update_every+j)
        # print(np.shape(box.state()))
        a, c, d = box.state()
        quiv.set_UVC(a,c,d) 
        
        return quiv,

    anim = FuncAnimation(fig, animate, 
                     frames=box.steps//update_every, interval=20, blit = True) 
    
    anim.save('test.mp4', writer='ffmpeg')

    box.plot_state()
    box.plot_energy()

def normal(box):
    box.run()
    box.plot_state()
    box.plot_energy()

def main():
    N = 10
    steps = 10000
    box = Box(N=N, steps=steps)

    animated(box)
    # normal(box)


if __name__ == "__main__":
    main()