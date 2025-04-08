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
        self.dx_wide = 2* (xmax - xmin) / (2*nx - 9)
        #print("dx_wide = ", self.dx_wide)
        self.dx_small = self.dx_wide / 4
        #print(self.dx_small)
        xGhost_start = xmin - self.dx_small * (np.flip(np.arange(nGhost))+0.5)
        xFirst = xmin + (np.arange(3)+0.5)*self.dx_small
        xCentral = xmin + xFirst[-1] + (np.arange(nx-6)+1/8+1/2)*self.dx_wide
        xEnd = xmin + xCentral[-1] + (1/8+1/2)*self.dx_wide + self.dx_small*(np.arange(3))
        xGhost_end = xmax + self.dx_small * (np.arange(nGhost)+0.5)
        self.xCentres = np.concatenate((xGhost_start, xFirst, xCentral, xEnd, xGhost_end))
        
        for n in range(1, nGhost +4):
            xWalls = self.xCentres[:n] - 0.5*self.dx_small
        for n in range(nGhost + 4, nx-3+nGhost):
            xWalls = self.xCentres[:n] - 0.5*self.dx_wide
        for n in range (nx-3, nx+2*nGhost+2):
            xWalls = self.xCentres[:n] - 0.5*self.dx_small

        xWalls = np.append(xWalls, self.xCentres[nx+2*nGhost-1] + 0.5*self.dx_small)
        #print("xWalls = ", xWalls)

        self.xL = xWalls[:nx+2*nGhost]
        self.xR = xWalls[1:nx+2*nGhost+1]
        self.diffX = self.xR - self.xL
        #print(self.xCentres[1:nx+2*nGhost+1])
        #print(self.xCentres[:nx+2*nGhost-1])
        print(self.diffX)

        # storage for the solution
        self.phi = np.zeros(nx+2*nGhost)
        self.phi_L = np.zeros(nx+1)
        self.phi_R = np.zeros(nx+1)


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
        #print("Grid values:")
        print("xL:", self.xL)
        print("xR:", self.xR)
        print("xCentres:", self.xCentres)
        print("Differences", self.xR - self.xL)
        print("ilo:", self.ilo)
        print("ihi:", self.ihi)
        #print("starting phi values", self.phi)

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
        flux_limiter = g.create_array()
        #R = g.create_array()

        if self.slope_type == "centered":

            for i in range(g.ilo-1, g.ihi+2):
                #modified[i] = 0.5*(g.phi[i+1] + g.phi[i])
                pass

        elif self.slope_type == "vanleer":
            for i in range(g.ilo-2, g.ihi+1):
                # van Leer limiter
                # returns the " limited slowp = van leer function * slope"
                epsilon = 1.0e-10
                R_l = (g.diffX[i] + g.diffX[i-1]) / g.diffX[i-1]
                r_l = ((g.phi[i-1] - g.phi[i-2]) + epsilon)/((g.phi[i]-g.phi[i-1])+ epsilon)*(g.diffX[i]+g.diffX[i-1])/(g.diffX[i-1] + g.diffX[i-2])
                modified_van_leer = (0.5 * R_l * r_l + 0.5 * R_l * abs(r_l))/(R_l + r_l - 1 )
                flux_limiter[i] = modified_van_leer / R_l
                print("flux_limiter = ", flux_limiter[i])
        # loop over all the interfaces.  Here, i refers to the left
        # interface of the zone.  Note that there are 1 more interfaces
        # than zones
        g.phi_L = g.create_array()
        g.phi_R = g.create_array()

        for i in range(g.ilo, g.ihi+1):

            # left state on the current interface comes from zone i-1
            g.phi_L[i] = g.phi[i-1] + flux_limiter[i] * (g.phi[i]-g.phi[i-1]) #
            # right state on the current interface comes from zone i
            g.phi_R[i] = g.phi[i] + flux_limiter[i+1] * (g.phi[i+1]-g.phi[i]) # 


        print("phi_L = ", g.phi_L)
        print("xL = ", g.xL)
        print("phi_R = ", g.phi_R)
        print("xR = ", g.xR)

        return g.phi_L, g.phi_R
        
#-----------------------------------------------------------------------------

xmin = 0.0
xmax = 1.0
nx = 20
nGhost = 2

g = Grid(nx, nGhost, xmin=xmin, xmax=xmax)
#g.check_grid_values()

calc = CalculateInterfaceValues(g, slope_type="vanleer")
calc.init_cond("tophat")
phi_init = calc.grid.phi.copy()
phi_L, phi_R = calc.interface_values()

plt.plot(g.xCentres[g.ilo:g.ihi+1], phi_init[g.ilo:g.ihi+1],
             ls=":", label="Exact")
#print(g.xCentres[g.ilo:g.ihi+1])
#print(phi_init[g.ilo:g.ihi+1])

#plt.plot(g.xL[g.ilo:g.ihi+2], g.phi_L[g.ilo:g.ihi+2], label="Centered")
#print(g.xL[g.ilo:g.ihi+2])
#print(g.phi_L[g.ilo:g.ihi+2])

#plt.plot(g.xR[g.ilo-1:g.ihi+1], g.phi_R[g.ilo-1:g.ihi+1], label="Centered")
#print(g.xR[g.ilo-1:g.ihi+1])
#print(g.phi_R[g.ilo-1:g.ihi+1])

calc = CalculateInterfaceValues(g, slope_type="vanleer")
calc.init_cond("tophat")
phi_L, phi_R = calc.interface_values()

plt.plot(g.xL[g.ilo:g.ihi+1], g.phi_L[g.ilo:g.ihi+1], label="Van Leer L")
#print(g.xL[g.ilo:g.ihi+1])
#print(g.phi_L[g.ilo:g.ihi+1])

plt.plot(g.xR[g.ilo-1:g.ihi], g.phi_R[g.ilo-1:g.ihi], label="Van Leer R")
#print(g.xR[g.ilo:g.ihi+1])
#print(g.phi_R[g.ilo:g.ihi+1])


plt.legend(frameon=False, loc="best")

plt.xlabel(r"$x$")
plt.ylabel(r"$phi$")
plt.show()