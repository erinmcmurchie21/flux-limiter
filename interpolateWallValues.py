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

        self.dx_wide = 2* (xmax - xmin) / (2*nx - 9)
        self.dx_small = self.dx_wide / 4
        
        #Added by DD
        xWalls_s = np.arange(-2,4)*self.dx_small
        xWalls_m = xWalls_s[5] + np.arange(1,nx-5)*self.dx_wide
        xWalls_e = xWalls_m[nx-7] + np.arange(1,6)*self.dx_small
        xWalls = np.concatenate((xWalls_s, xWalls_m, xWalls_e))

        self.xL = xWalls[:nx+2*nGhost]
        self.xR = xWalls[1:nx+2*nGhost+1]
        self.diffX = self.xR - self.xL
        print(self.diffX)

        # storage for the solution
        self.phi = np.zeros(nx+2*nGhost)
        self.phi_R = np.zeros(nx+2*nGhost)


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
            self.grid.phi[np.logical_and(self.grid.xCentres >0.33,
                                       self.grid.xCentres <0.66)] = 1.0

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

        if self.slope_type == "centered":

            for i in range(g.ilo-1, g.ihi+2):
                #modified[i] = 0.5*(g.phi[i+1] + g.phi[i])
                pass

        elif self.slope_type == "vanleer":
            for i in range(g.ilo-2, g.ihi+2):
                # van Leer limiter
                # returns the " limited slowp = van leer function * slope"
                epsilon = 1.0e-10
                R_r = (g.diffX[i+1] + g.diffX[i]) / g.diffX[i]
                r_r = ((g.phi[i] - g.phi[i-1]) + epsilon)/((g.phi[i+1]-g.phi[i])+ epsilon)*(g.diffX[i+1]+g.diffX[i])/(g.diffX[i] + g.diffX[i-1])
                modified_van_leer = (0.5 * R_r * r_r + 0.5 * R_r * abs(r_r))/(R_r + r_r - 1 )
                flux_limiter[i] = modified_van_leer / R_r
                #print("flux_limiter = ", flux_limiter[i])
        # loop over all the interfaces.  Here, i refers to the left
        # interface of the zone.  Note that there are 1 more interfaces
        # than zones
        g.phi_R = g.create_array()

        for i in range(g.ilo, g.ihi+1):

            g.phi_R[i] = g.phi[i] + flux_limiter[i] * (g.phi[i+1]-g.phi[i]) #
            # right state on the current interface comes from zone i

        return g.phi_R
        
#-----------------------------------------------------------------------------

xmin = 0.0
xmax = 1.0
nx = 10
nGhost = 2
testcase = "tophat"
fluxlimiter = "vanleer"

g = Grid(nx, nGhost, xmin=xmin, xmax=xmax)
#g.check_grid_values()

calc = CalculateInterfaceValues(g, slope_type=fluxlimiter)
calc.init_cond(testcase)
phi_init = calc.grid.phi.copy()
phi_R = calc.interface_values()

plt.plot(g.xCentres[g.ilo:g.ihi+1], phi_init[g.ilo:g.ihi+1],
             'bo', markersize=3, label="Exact")

#plt.plot(g.xR[g.ilo-1:g.ihi+1], g.phi_R[g.ilo-1:g.ihi+1], label="Centered")
#print(g.xR[g.ilo-1:g.ihi+1])
#print(g.phi_R[g.ilo-1:g.ihi+1])

calc = CalculateInterfaceValues(g, slope_type=fluxlimiter)
calc.init_cond(testcase)
phi_R = calc.interface_values()

plt.plot(g.xR[g.ilo:g.ihi+1], g.phi_R[g.ilo:g.ihi+1], 'g+', label="Van Leer R")
print(g.xR[:])
print(g.phi_R[:])


plt.legend(frameon=False, loc="best")

plt.xlabel(r"$x$")
plt.ylabel(r"$phi$")
plt.show()