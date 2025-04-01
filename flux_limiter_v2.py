''' Define flux limiting function for advection problems
    This code is a simple implementation of a flux limiter for a one-dimensional advection problem.
    It uses the Van Leer flux limiter to compute the flux at the cell interfaces.
    The code also includes a grid class to handle the grid properties and boundary conditions.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

class Grid(object):

    def __init__(self, nx, nGhost, xmin=0.0, xmax=1.0):

        self.nGhost = nGhost
        self.nx = nx

        self.xmin = xmin
        self.xmax = xmax

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = nGhost
        self.ihi = nGhost+nx-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.xCentres = xmin + (np.arange(nx+2*nGhost)-nGhost+0.5)*self.dx
        self.xL = xmin + (np.arange(nx+2*nGhost)-nGhost)*self.dx
        self.xR = xmin + (np.arange(nx+2*nGhost)+1.0)*self.dx

        # storage for the solution
        self.phi = np.zeros(nx+2*nGhost)


    def create_array(self):
        """ return a scratch array dimensioned for our grid """
        return np.zeros(self.nx+2*self.nGhost)


    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions """

        for n in range(self.nGhost):
            # left boundary
            self.phi[self.ilo-1-n] = self.phi[self.ihi-n]

            # right boundary
            self.phi[self.ihi+1+n] = self.phi[self.ilo+n]
    
    def check_grid_values(self):
        """ check the grid values """
        print("Grid values:")
        print("xL:", self.xL)
        print("xR:", self.xR)
        print("xCentres:", self.xCentres)
        print("dx:", self.dx)
        print("ilo:", self.ilo)
        print("ihi:", self.ihi)
        print("starting phi values", self.phi)


class Simulation(object):

    def __init__(self, grid, u, slope_type):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.u = u   # the constant advective velocity
        self.slope_type = slope_type # slope type for the limiter


    def init_cond(self, type="tophat"):
        """ initialize the data """
        if type == "tophat":
            self.grid.phi[:] = 0.0
            self.grid.phi[np.logical_and(self.grid.xCentres >= 0.333,
                                       self.grid.xCentres <= 0.666)] = 1.0

        elif type == "sine":
            self.grid.phi[:] = np.sin(2.0*np.pi*self.grid.xCentres/(self.grid.xmax-self.grid.xmin))

        elif type == "gaussian":
            phi_L = 1.0 + np.exp(-60.0*(self.grid.xL - 0.5)**2)
            phi_R = 1.0 + np.exp(-60.0*(self.grid.xR - 0.5)**2)
            phi_c = 1.0 + np.exp(-60.0*(self.grid.xCentres - 0.5)**2)
            
            self.grid.phi[:] = (1./6.)*(phi_L + 4*phi_c + phi_R)


    def timestep(self):
        """ return the advective timestep """
        dt = 0.01
        return dt


    def period(self):
        """ return the period for advection with velocity u """
        return (self.grid.xmax - self.grid.xmin)/self.u


    def interface_values(self, dt):
        """ compute the left and right interface states """

        # compute the piecewise linear slopes
        g = self.grid
        slope = g.create_array()

        if self.slope_type == "centered":

            for i in range(g.ilo-1, g.ihi+2):
                slope[i] = 0.5*(g.phi[i+1] - g.phi[i-1])/g.dx

        elif self.slope_type == "vanleer":
            for i in range(g.ilo-1, g.ihi+2):
                # van Leer limiter
                # returns the " limited slowp = van leer function * slope"
                epsilon = 1.0e-10
                r = ((g.phi[i+1] - g.phi[i])+ epsilon)/((g.phi[i]-g.phi[i-1])+ epsilon)
                slope [i] = (r + abs(r))/(1.0 + abs(r)) * (g.phi[i]-g.phi[i-1])/g.dx


        # loop over all the interfaces.  Here, i refers to the left
        # interface of the zone.  Note that there are 1 more interfaces
        # than zones
        phi_L = g.create_array()
        phi_R = g.create_array()

        for i in range(g.ilo, g.ihi+2):

            # left state on the current interface comes from zone i-1
            phi_L[i] = g.phi[i-1] + 0.5*g.dx*(1.0 - self.u*dt/g.dx)*slope[i-1] #
            # right state on the current interface comes from zone i
            phi_R[i] = g.phi[i] - 0.5*g.dx*(1.0 + self.u*dt/g.dx)*slope[i] # 

        return phi_L, phi_R


    def advection(self, phi_L, phi_R):
        """
        Riemann problem for advection -- this is simply upwinding,
        but we return the flux
        """

        if self.u > 0.0:
            return self.u*phi_L
        else:
            return self.u*phi_R


    def update(self, dt, flux):
        """ conservative update """

        g = self.grid

        phi_new = g.create_array()

        phi_new[g.ilo:g.ihi+1] = g.phi[g.ilo:g.ihi+1] + \
            dt/g.dx * (flux[g.ilo:g.ihi+1] - flux[g.ilo+1:g.ihi+2])

        return phi_new


    def run_simulation(self, num_periods=1):
        """ evolve the linear advection equation """
        self.t = 0.0
        g = self.grid

        tmax = num_periods*self.period()

        # main evolution loop
        while self.t < tmax:

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep()

            if self.t + dt > tmax:
                dt = tmax - self.t

            # get the interface states
            phi_L, phi_R = self.interface_values(dt)

            # solve the Riemann problem at all interfaces
            flux = self.advection(phi_L, phi_R)

            # do the conservative update
            phi_new = self.update(dt, flux)

            g.phi[:] = phi_new[:]

            self.t += dt





    #-------------------------------------------------------------------------
    # compare limiting and no-limiting

xmin = 0.0
xmax = 1.0
nx = 50
nGhost = 2

g = Grid(nx, nGhost, xmin=xmin, xmax=xmax)
    #g.check_grid_values()

u = 1.0

s = Simulation(g, u, slope_type="centered")
s.init_cond("tophat")
phi_init = s.grid.phi.copy()
s.run_simulation(num_periods=1)

plt.plot(g.xCentres[g.ilo:g.ihi+1], phi_init[g.ilo:g.ihi+1],
             ls=":", label="exact")

plt.plot(g.xCentres[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1], label="centered")

s = Simulation(g, u, slope_type="vanleer")
s.init_cond("tophat")
s.run_simulation(num_periods=1)

plt.plot(g.xCentres[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1],
             label="vanLeer")

plt.legend(frameon=False, loc="best")

plt.xlabel(r"$x$")
plt.ylabel(r"$phi$")
plt.show()