""" 1D non-uniform grid for calculating the interior wall values
        Flux at volume walls constructed from
         Zhang, W., Przybycien, T., Schmölder, J., Leweke, S. and von Lieres, E., 2024. 
         Solving crystallization/precipitation population balance models in CADET, part I: 
         Nucleation growth and growth rate dispersion in batch and continuous modes on 
         nonuniform grids. Computers & Chemical Engineering, 183, p.108612. 
         https://www.sciencedirect.com/science/article/pii/S0098135424000309#sec4

"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


#grid object: takes inputs for number of cells, number of ghost cells, and min/max x values
# and returns values for the locations of the cell centres, the cell walls, and the cell widths
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
        xWalls_s = np.arange(-nGhost,4)*self.dx_small
        xWalls_m = xWalls_s[-1] + np.arange(1,nx-5)*self.dx_wide
        xWalls_e = xWalls_m[-1] + np.arange(1,3+nGhost+1)*self.dx_small
        self.xWalls = np.concatenate((xWalls_s, xWalls_m, xWalls_e))

        self.diffX = self.xWalls[1:nx+2*nGhost+1] - self.xWalls[:nx+2*nGhost]
    
    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions. 
        This should be amended once boundary conditions have been decided """

        for n in range(self.nGhost):
            # left boundary
            self.phi[self.ilo-1-n] = self.phi[self.ihi-n]

            # right boundary
            self.phi[self.ihi+1+n] = self.phi[self.ilo+n]

    def get_grid_values(self):
        """ Get the locations and value of the gri """
        return self.xCentres, self.diffX, self.ilo, self.ihi, self.nGhost, self.xWalls

class CalculateInterfaceValues(object):
    def __init__(self, xCentres, diffX, ilo, ihi, nGhost, xWalls, slope_type):
        
        #setting up the grid structure
        self.xCentres = xCentres
        self.diffX = diffX
        self.slope_type = slope_type
        self.ilo = ilo
        self.ihi = ihi
        self.nGhost = nGhost
        self.xWalls = xWalls

        #creating arrays to store the values of the cell centres and wall
        self.phi = np.zeros(len(xCentres))
        self.phi_R = np.zeros(len(xCentres))

    def init_cond(self, type="tophat"):
        
        """ initialize the data with a test function. This can be removed later """
        if type == "tophat":
            self.phi[:] = 0.0
            self.phi[np.logical_and(self.xCentres >0.33,
                                       self.xCentres <0.66)] = 1.0

        elif type == "sine":
            #self.phi[:] = np.sin(2.0*np.pi*self.xCentres/(self.xmax-self.xmin))
            pass

        elif type == "gaussian":
            phi_L = 1.0 + np.exp(-60.0*(self.xWalls[:-1] - 0.5)**2)
            phi_R = 1.0 + np.exp(-60.0*(self.xWalls[1:] - 0.5)**2)
            phi_c = 1.0 + np.exp(-60.0*(self.xCentres - 0.5)**2)
            
            self.phi[:] = (1./6.)*(phi_L + 4*phi_c + phi_R)
            
    def interface_values(self):
        """ compute the left and right interface states """

        # compute the piecewise linear slopes
        flux_limiter = np.zeros(len(xCentres))

        if self.slope_type == "vanleer":
            for i in range(self.ilo-self.nGhost, self.ihi+self.nGhost):
                # van Leer limiter
                # returns the " limited slowp = van leer function * slope"
                epsilon = 1.0e-10
                # R_{i+1/2} = \frac{\Delta x_{i+1}+\Delta x_i}{\Delta x_i}
                R_r = (self.diffX[i+1] + self.diffX[i]) / self.diffX[i]
                # r_{i+1/2} = \frac{\phi_{i+1}-\phi_i}{\phi_i-\phi_{i-1}} \cdot \frac{\Delta x_{i+1}+\Delta x_i}{\Delta x_i + \Delta x_{i-1}}
                r_r = ((self.phi[i] - self.phi[i-1]) + epsilon)/((self.phi[i+1]-self.phi[i])+ epsilon)*(self.diffX[i+1]+self.diffX[i])/(self.diffX[i] + self.diffX[i-1])
                #\psi_{i+1/2} =  \frac{\frac{1}{2} R_{i+1/2} r_{i+1/2} + \frac{1}{2} R_{i+1/2}|r_{i+1/2}|}{R_{i+1/2} + r_{i+1/2} - 1}
                modified_van_leer = (0.5 * R_r * r_r + 0.5 * R_r * abs(r_r))/(R_r + r_r - 1 )
                
                flux_limiter[i] = modified_van_leer / R_r
                
        # loop over all the interfaces.  Here, i refers to the left
        # interface of the zone.  Note that there are 1 more interfaces
        # than zones

        self.phi_R = np.zeros(len(xCentres))

        for i in range(self.ilo, self.ihi):
            
            # \phi_{i+1/2} = \phi_i + \frac{\psi_{i+1/2}}{R_{i+1/2}} * (\phi_{i+1}-\phi_i)
            self.phi_R[i] = self.phi[i] + flux_limiter[i] * (self.phi[i+1]-self.phi[i]) #
            # right state on the current interface comes from zone i

        return self.phi_R
        
#-----------------------------------------------------------------------------

xmin = 0.0
xmax = 1.0
nx = 10
nGhost = 1
testcase = "tophat"
fluxlimiter = "vanleer"

g = Grid(nx, nGhost, xmin=xmin, xmax=xmax)
xCentres, diffX, ilo, ihi, nGhost, xWalls = g.get_grid_values()

calc = CalculateInterfaceValues(xCentres, diffX, ilo, ihi, nGhost, xWalls, slope_type=fluxlimiter)
calc.init_cond(testcase)
phi_init = calc.phi.copy()
phi_R = calc.interface_values()

plt.plot(xCentres[ilo:ihi+1], phi_init[ilo:ihi+1],
             'bo', markersize=3, label="Exact")

calc = CalculateInterfaceValues(xCentres, diffX, ilo, ihi, nGhost, xWalls, slope_type=fluxlimiter)
calc.init_cond(testcase)
phi_R = calc.interface_values()

plt.plot(xWalls[ilo+1:ihi+1], phi_R[ilo:ihi], 'g+', label="Van Leer")
print("xWalls = ", xWalls[ilo+1:ihi+1])
print("phi_R = ", phi_R[:])


# Figure settings
plt.legend(frameon=False, loc="best")
plt.xlabel(r"$x$")
plt.ylabel(r"$phi$")
plt.show()