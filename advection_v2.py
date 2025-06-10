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
        self.xR = xmin + (np.arange(nx+2*nGhost)-nGhost+1.0)*self.dx

        # storage for the solution
        self.phi = np.zeros(nx+2*nGhost)
        self.phi_L = np.zeros(nx+2*nGhost)
        self.phi_R = np.zeros(nx+2*nGhost)


    def create_array(self):
        """ return a scratch array dimensioned for our grid """
        return np.zeros(self.nx+2*self.nGhost)


    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions """

        for n in range(self.nGhost):
            # left boundary
            self.phi[self.ilo-1-n] = 0 #self.phi[self.ihi-n]

            # right boundary
            self.phi[self.ihi+1+n] = 0  #self.phi[self.ilo+n]
    
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

class CalculateInterfaceValues(object):
    def __init__(self, grid, slope_type):
        self.grid = grid
        self.slope_type = slope_type

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

    def interface_values(self):
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
        g.phi_L = g.create_array()
        g.phi_R = g.create_array()

        for i in range(g.ilo, g.ihi+2):

            # left state on the current interface comes from zone i-1
            g.phi_L[i] = g.phi[i-1] + 0.5*g.dx*slope[i-1] #
            # right state on the current interface comes from zone i
            g.phi_R[i] = g.phi[i] + 0.5*g.dx*slope[i] # 

        return g.phi_L, g.phi_R
    

#-----------------------------------------------------------------------------

xmin = 0.0
xmax = 1.0
nx = 50
nGhost = 2

g = Grid(nx, nGhost, xmin=xmin, xmax=xmax)
g.check_grid_values()

calc = CalculateInterfaceValues(g, slope_type="centered")
calc.init_cond("tophat")
phi_init = calc.grid.phi.copy()
phi_L, phi_R = calc.interface_values()

plt.plot(g.xCentres[g.ilo:g.ihi+1], phi_init[g.ilo:g.ihi+1],
             ls=":", label="Exact")
print(g.xCentres[g.ilo:g.ihi+1])
print(phi_init[g.ilo:g.ihi+1])

plt.plot(g.xL[g.ilo:g.ihi+2], g.phi_L[g.ilo:g.ihi+2], label="Centered")
print(g.xL[g.ilo:g.ihi+2])
print(g.phi_L[g.ilo:g.ihi+2])

plt.plot(g.xR[g.ilo-1:g.ihi+1], g.phi_R[g.ilo-1:g.ihi+1], label="Centered")
print(g.xR[g.ilo-1:g.ihi+1])
print(g.phi_R[g.ilo-1:g.ihi+1])

calc = CalculateInterfaceValues(g, slope_type="vanleer")
calc.init_cond("tophat")
phi_L, phi_R = calc.interface_values()

plt.plot(g.xL[g.ilo:g.ihi+2], g.phi_L[g.ilo:g.ihi+2], label="Van Leer")
print(g.xL[g.ilo:g.ihi+2])
print(g.phi_L[g.ilo:g.ihi+2])

plt.plot(g.xR[g.ilo-1:g.ihi+1], g.phi_R[g.ilo-1:g.ihi+1], label="Van Leer")
print(g.xR[g.ilo-1:g.ihi+1])
print(g.phi_R[g.ilo:g.ihi+1])


plt.legend(frameon=False, loc="best")

plt.xlabel(r"$x$")
plt.ylabel(r"$phi$")
plt.show()