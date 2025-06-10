"""This is a first draft of a column model for the 1D case.
"""
#Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# Define model inputs

#define extra functions
def quadratic_extrapolation(x0, y0, x1, y1, x2, y2, x):
    """Quadratic extrapolation for ghost cells"""
    # Calculate the coefficients of the quadratic polynomial
    #Y = y0*L_0(x) + y1*L_1(x) + y2*L_2(x)
    L_0 = (x-x1)*(x-x2)/((x0-x1)*(x0-x2))
    L_1 = (x-x0)*(x-x2)/((x1-x0)*(x1-x2))
    L_2 = (x-x0)*(x-x1)/((x2-x0)*(x2-x1))
    Y = y0*L_0 + y1*L_1 + y2*L_2
    return Y
    # Calculate the value of the polynomial at the ghost cell location

def quadratic_extrapolation_derivative(x0, y0, x1, y1, x2, y2):
    # We want to solve for value y2, given that dy/dx = 0 at x2
    y2 = (y1*(x0-x2)**2-y0(x1-x2)**2)/((x0-x1)*(x0+x1-2*x2))
    return y2

def adsorption_isotherm_1(pressure, mole_fraction):
    K_1 = 1.0e-3 # adsorption equilibrium constant
    R = 8.314 # J/(mol*K)
    T = 298.15 # K
    equilibrium_loading = - K_1 * pressure * mole_fraction/ (R*T)
    return equilibrium_loading

def adsorption_isotherm_2(pressure):
    K_2 = 1.0e-3 # adsorption equilibrium constant
    R = 8.314 # J/(mol*K)
    T = 298.15 # K
    equilibrium_loading = - K_2 * pressure / (R*T)
    return equilibrium_loading

class AdsorptionColumn(object):
    """ Adsorption column object
    - Takes inputs for locations of cell centres, cell walls and cell widths
    - Takes inputs for state variables P, y, n1, n2, at each cell centre from 
        previous time step and F for inlet and outlet molar flows, 
        e.g. mega vector 4N + 4 values long {P[N], y[N], n1[N], n2[N], F[4]}
        Then:
    - Calculates boundary conditions and ghost cells at next time step
    - Calculates ODEs for the next time step
    - Exports results to ODE solver
    """
    def __init__(self,  num_cells, nGhost, zCentres, zWalls, deltaZ, results_vector,bed_properties, feed_properties):
        # Set grid structure
        self.num_cells = num_cells # number of values at cell centres, excluding ghost cells
        self.nGhost = nGhost # number of ghost cells on each side of the column

        self.xCentres = zCentres # includes ghost cell centre locations
        self.xWalls = zWalls = # includes ghost cell wall locations
        self.deltaZ = deltaZ # inlcudes ghost cell widths

        # set values at cell centres
        self.results_vector = results_vector
        
        # create empty array for wall values
        self.phi_walls = np.zeros(len(zWalls))

    def set_bed_properties(self, bed_properties):

        """ Set bed properties for the column model """
        self.bed_voidage = bed_properties['bed_voidage']
        self.particle_diameter = bed_properties['particle_diameter']
        self.column_area = bed_properties['column_area']
        self.R = bed_properties['R']
        self.T_column = bed_properties['T_column']
        self.rho_b = bed_properties['rho_b']

        return self.bed_voidage, self.particle_diameter, self.column_area, self.R, self.T_column, self.rho_b
    
    def set_feed_properties(self, feed_properties):
        """ Set feed properties for the column model """
        self.y_feed_value = feed_properties['y_feed_value']
        self.P_feed_value = feed_properties['P_feed_value']
        self.rho_feed = feed_properties['rho_feed']
        self.mu = feed_properties['mu']
        self.feed_mass_flow = feed_properties['feed_mass_flow']

        return self.y_feed_value, self.P_feed_value, self.rho_feed, self.mu, self.feed_mass_flow

    def data_prep(self):
        # Split up mega-vector into the different components of the model
        # P, y, n1, n2, F
        N_tot = self.num_cells
        # P = pressure, y = mole fraction component 1, n1 = adsorbed phase component 1,
        #  adsorbed phase component 2, F = inlet/outlet flow rates
        self.P = self.results_vector[:N_tot]
        self.y = self.results_vector[N_tot:2*N_tot]
        self.n1 = self.results_vector[2*N_tot:3*N_tot]
        self.n2 = self.results_vector[3*N_tot:4*N_tot]
        self.F = self.results_vector[4*N_tot:4*N_tot+4]
        
        
        return self.P, self.y, self.n1, self.n2, self.F

    def inlet_BC(self, inlet_type):
        # Implement inlet boundary conditions

        if inlet_type == "massflow":
            # Mass flow inlet boundary condition

             #Convert from mass flowrate to volumetric flowrate
            vol_flow = feed_mass_flow / self.rho_feed
            #find velocity
            v_inlet_ = vol_flow / self.column_area
            #use velocity to find dPdZ from ergun equation
            dPdz_inlet = 1.75 * (1- self.bed_voidage) * self.rho_feed * v_inlet**2 / (self.bed_voidage * self.particle_diameter) + 12 * self.mu * (1-self.bed_voidage)**2 *v_inlet / (self.bed_voidage**3 * self.particle_diameter**2)
            #use dPdZ to find P_inlet_
            P_inlet_ = self.P[0] + dPdz_inlet * (self.xWalls[self.nGhost] - self.xCentres[0])

            """"""
            #can P_inlet_ be found from linear interpolation of P values at cell centres?
            P_inlet_ = (self.xWalls[self.nGhost] - self.xCentres[2]) * (self.P[2] - self.P[1]) / (self.xCentres[2] - self.xCentres[1])
            #y can be defined as inlet feed value if axial dispersion not included
            y_inlet_ = y_feed_value

        elif inlet_type == "closed":
            # from lagrangian derivative function
            P_inlet_ = quadratic_extrapolation_derivative(self.xCentres[self.nGhost], self.P[0], self.xCentres[self.nGhost+1], self.P[1],
                                               self.xCentres[self.nGhost+2], self.P[2], self.xWalls[self.nGhost])
            y_inlet_ = quadratic_extrapolation_derivative(self.xCentres[self.nGhost], self.y[0], self.xCentres[self.nGhost+1], self.y[1],
                                               self.xCentres[self.nGhost+2], self.y[2], self.xWalls[self.nGhost])
            n1_inlet_ = quadratic_extrapolation_derivative(self.xCentres[self.nGhost], self.n1[0], self.xCentres[self.nGhost+1], self.n1[1],
                                               self.xCentres[self.nGhost+2], self.n1[2], self.xWalls[self.nGhost]) 
            n2_inlet_ = quadratic_extrapolation_derivative(self.xCentres[self.nGhost], self.n2[0], self.xCentres[self.nGhost+1], self.n2[1],
                                               self.xCentres[self.nGhost+2], self.n2[2], self.xWalls[self.nGhost])
            v_inlet_ = 0
        
        return P_inlet_, y_inlet_, v_inlet_

    def outlet_BC(self, outlet_type):
        # Implement outlet boundary conditions

        if outlet_type == "pressure":
            # Mass flow inlet boundary condition
            outlet_boundary_value = 0 #not zero
        
        elif outlet_type == "massflow":
            # Mass flow inlet boundary condition
            outlet_boundary_value = 0

        elif outlet_type == "closed":
            #uses langrangian derivate function
            outlet_boundary_value = 0 #not zero
        
        
        return P_outlet_, y_outlet_, v_outlet_

    def ghost_cell_calcs(self):
        # Extrapolate to find ghost cell values for P, y, n1, n2
        #Lagragian quadratic extrapolation for ghost cells
        N = self.num_cells
        

        # Left ghost cell, i = -1, using values at i = 1, 2, 3
        P_ghost_start = (self.xCentres[0] - self.xCentres[2]) * (self.P[2] - self.P[1]) / (self.xCentres[2] - self.xCentres[1])
        # Right ghost cell, i = N+1, using values at i = N-2, N-1, N
        # P_ghost_end = (self.xCentres[N+1] - self.xCentres[N]) * (self.P[N] - self.P[N-1]) / (self.xCentres[N] - self.xCentres[N-1]) 

        P_centres_and_ghost_cells = np.concatenate((P_ghost_start, self.P))
        
        # Left ghost cell, i = -1, using values at i = 1, 2, 3
        y_ghost_start = quadratic_extrapolation(self.xCentres[0], self.y[0], self.xCentres[1], self.y[1],
                                                self.xCentres[2], self.y[2], self.xGhostcell_left)
        # Right ghost cell, i = N+1, using values at i = N-2, N-1, N
        #y_ghost_end = quadratic_extrapolation(self.xCentres[N-2], self.y[N-2], self.xCentres[N-1], self.y[N-1],
                                               # self.xCentres[N], self.y[N], self.xGhostcell_right) 

        y_centres_and_ghost_cells = np.concatenate((y_ghost_start, self.y))                       
    
        return y_centres_and_ghost_cells, P_centres_and_ghost_cells # should have N + 2 values for each
    
    def calculate_internal_wall_values(self):
        """ Calculate wall values for P, y, n1, n2 at cell walls
         Value at cell walls for z=0 and z=1 are from boundary conditions.
         Internal wall values for P, y, n1, n2 are calculated
         Use van Leer flux limiter function for P and y
         n1 and n2 wall values are not required because dn1/dt and dn2/dt can be calulated directly
         """
        epsilon = 1.0e-10
        y_centres_and_ghost_cells, P_centres_and_ghost_cells = self.ghost_cell_calcs()
        #calculate dP/dz at internal cell walls by linear interpolation
        #should this include the values at 0 and 1? 
        dPdz_walls = (self.P[1:]- self.P[:-1]) / (self.xCentres[1:] - self.xCentres[:-1])

        #calculate velocity vector at cell walls, from Ergun equation
        a = 1.75 * (1- self.epsilon) * self.rho_feed / (self.epsilon * self.particle_diameter)
        b = 12 * self.mu * (1-self.epsilon)**2 / (self.epsilon**3 * self.particle_diameter**2)
        c = dPdz_walls
        v_walls = -b + np.sqrt(b**2 - 4*a*c) / (2*a) # should have N + 1 values

        #calculate P at cell walls by van Leer flux limiter function
        R_r = (self.deltaZ[2:] + self.deltaZ[1:-1]) / self.deltaZ[1:-1]  # R_{i+1/2} = (deltaZ_{i+1} + deltaZ_i) / deltaZ_i
        r_r = ((P_centres_and_ghost_cells[2:] - P_centres_and_ghost_cells[:-2]) + epsilon)/((P_centres_and_ghost_cells[2:]-P_centres_and_ghost_cells[1:-1])+ epsilon)*(self.deltaZ[2:]+self.deltaZ[1:-1])/(self. deltaZ[1:-1] + self.deltaZ[0:-2])
        modified_van_leer = (0.5 * R_r * r_r + 0.5 * R_r * abs(r_r))/(R_r + r_r - 1 )
        flux_limiter = modified_van_leer / R_r
        P_walls = P_centres_and_ghost_cells[1:-1] + flux_limiter[1:-1] * (P_centres_and_ghost_cells[2:]-P_centres_and_ghost_cells[1:-1])

        #y vector at cell walls, from van leer flux limiter function
        R_r = (self.deltaZ[2:] + self.deltaZ[1:-1]) / self.deltaZ[1:-1]
        r_r = ((y_centres_and_ghost_cells[1:-1] - y_centres_and_ghost_cells[:-2]) + epsilon)/((y_centres_and_ghost_cells[2:]-y_centres_and_ghost_cells[1:-1])+ epsilon)*(self.deltaZ[2:]+self.deltaZ[1:-1])/(self. deltaZ[1:-1] + self.deltaZ[0:-2])
        modified_van_leer = (0.5 * R_r * r_r + 0.5 * R_r * abs(r_r))/(R_r + r_r - 1 )
        flux_limiter = modified_van_leer / R_r
        y_walls = y_centres_and_ghost_cells[1:-1] + flux_limiter[1:-1] * (self.y[i+1]-self.y[i])

        return P_walls, y_walls, v_walls
    
    def get_wall_values(self, inlet_type, outlet_type):
        """ Compile wall values for P, y, n1, n2 at cell walls
         Value at cell walls for z=0 and z=1 are from boundary conditions.
         Internal wall values for P, y, n1, n2 are calculated
         Use van Leer flux limiter function for P and y
         n1 and n2 wall values are not required because dn1/dt and dn2/dt can be calulated directly
         """
        
        P_inlet_, y_inlet_, v_inlet_ = self.inlet_BC(inlet_type=inlet_type)
        P_outlet_, y_outlet_, v_outlet = self.outlet_BC(outlet_type=outlet_type)
        
        #calculate internal wall values
        P_walls, y_walls, v_walls = self.calculate_internal_wall_values()

        #add boundary condition values for z=0 and z=1
        P_walls = np.concatenate((P_inlet_, P_walls, P_outlet_))  # add inlet and outlet values
        y_walls = np.concatenate((y_inlet_, y_walls, y_outlet_))  # add inlet and outlet values
        v_walls = np.concatenate((v_inlet_, v_walls, v_outlet))  # add inlet and outlet values

        return P_walls, y_walls, v_walls

    def ODE_calculations(self):
        # Calculate differential term at cell centres for the next time step
        # using wall values calculated in previous step

        P_walls, y_walls, v_walls = self.get_wall_values(inlet_type=inlet_type, outlet_type=outlet_type)


        #n1 vector at cell centres, from van leer flux limiter function
        k1 = 1.0e-3 # linear driving force mass transfer constant
        dn1dt = k1*(adsorption_isotherm_1(self.P, self.y)-self.n1)
            
        #n2 vector at cell walls, from van leer flux limiter function
        #updating central values
        k2 = 1.0e-3 # linear driving force mass transfer constant
        dn2dt = k2*(adsorption_isotherm_2(self.P, 1-self.y)-self.n2)


        #pressure differential at cell centres
        
        dPdt = 1/ self.deltaZ[1:-1] * (v_walls[:-1]*P_walls[:-1] - v_walls[1:]*P_walls[1:]) - self.rho_b*self.R*self.T_column/self.bed_voidage * (dn1dt[:] + dn2dt[:]) # dP/dz at cell centres

        #mole fraction differential at cell centres
        dydt = 1/ (self.deltaZ[1:-1] * self.P[:]) * (v_walls[:-1]*P_walls[:-1]*y_walls[:-1] - v_walls[1:]*P_walls[1:](y_walls[1:])) - self.rho_b*self.R*self.T_column/self.bed_voidage * dn1dt[:] - self.y[:]/self.P[:] * dPdt[:]

        dF1dt = self.bed_voidage * self.column_area / (self. R * self.T_column) * v_walls[0]*P_walls[0] * y_walls[0]
        dF2dt = self.bed_voidage * self.column_area / (self. R * self.T_column) * v_walls[0]*P_walls[0] * (1 - y_walls[0])
        dF3dt = self.bed_voidage * self.column_area / (self. R * self.T_column) * v_walls[-1]*P_walls[-1] * y_walls[-1]
        dF4dt = self.bed_voidage * self.column_area / (self. R * self.T_column) * v_walls[-1]*P_walls[-1] * (1 - y_walls[-1])
        dFdt = np.array([dF1dt, dF2dt, dF3dt, dF4dt])  # inlet and outlet flow rates

        
        return dPdt, dydt, dn1dt, dn2dt, dFdt

    def return_mega_vector(self):
        # Return the mega-vector for the ODE solver: combine P, y, n1, n2, F vectors
        
        dPdt, dydt, dn1dt, dn2dt, dFdt = self.ODE_calculations()
        results_vector = np.concatenate((dPdt, dydt, dn1dt, dn2dt, dFdt))
        
        
        return results_vector



