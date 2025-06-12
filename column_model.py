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
    K_1 = 6.077 / 10**5 # adsorption equilibrium constant
    R = 8.314 # J/(mol*K)
    T = 308.15 # K
    equilibrium_loading = K_1 * pressure * mole_fraction/ (R*T)
    return equilibrium_loading

def adsorption_isotherm_2(pressure, mole_fraction):
    K_2 = 0 / 10**5 # adsorption equilibrium constant
    R = 8.314 # J/(mol*K)
    T = 308.15 # K
    equilibrium_loading = K_2 * pressure * mole_fraction/ (R*T)
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
    def __init__(self, column_grid, bed_properties, inlet_values, outlet_values):
        
        """ Set grid structure from column_grid, which contains the following keys:
            "num_cells", "nGhost", "zCentres", "zWalls", "deltaZ" """
        self.num_cells = int(column_grid["num_cells"]) # number of values at cell centres, excluding ghost cells
        self.nGhost = int(column_grid["nGhost"]) # number of ghost cells on each side of the column
        self.xCentres = column_grid["zCentres"] # includes ghost cell centre locations
        self.xWalls = column_grid["zWalls"] # includes ghost cell wall locations
        self.deltaZ = column_grid["deltaZ"] # inlcudes ghost cell widths
        

        """ Set bed properties for the column model """
        self.bed_voidage = float(bed_properties['bed_voidage'])
        self.particle_diameter = float(bed_properties['particle_diameter'])
        self.column_area = float(bed_properties['column_area'])
        self.R = float(bed_properties['R'])
        self.T_column = float(bed_properties['T_column'])
        self.rho_b = float(bed_properties['rho_b'])
        self.MW = 1/10**3 * np.array([28.0134, 32.01])  # Molecular weights of components in kg/mol, e.g. nitrogen and oxygen

        """Import boundary conditions for inlet and outlet"""
        self.inlet_type = inlet_values["inlet_type"]  # e.g. "massflow", "pressure", "closed"
        self.outlet_type = outlet_values["outlet_type"]
        self.inlet_values = inlet_values
        self.outlet_values = outlet_values
        self.mu = float(inlet_values["mu"])  # assuming mu is the same for inlet and outlet

    def data_prep(self,results_vector):
        # Split up mega-vector into the different components of the model
        # P, y, n1, n2, F
        N_tot = self.num_cells
        # P = pressure, y = mole fraction component 1, n1 = adsorbed phase component 1,
        #  adsorbed phase component 2, F = inlet/outlet flow rates
        P = results_vector[0:N_tot]
        y = results_vector[N_tot:2*N_tot]
        n1 = results_vector[2*N_tot:3*N_tot]
        n2 = results_vector[3*N_tot:4*N_tot]
        F = results_vector[4*N_tot:4*N_tot+4]
        
        return P, y, n1, n2, F

    def inlet_BC(self, results_vector):
        # Implement inlet boundary conditions

        P, y, n1, n2, F = self.data_prep(results_vector)
        P = np.array(P)
    
        if self.inlet_type == "mass_flow":
            # Mass flow inlet boundary condition
            
            inlet_values = self.inlet_values
            y_inlet_ = float(inlet_values["y_feed_value"])
             #Convert from mass flowrate to volumetric flowrate
            vol_flow_inlet_ = float(inlet_values["feed_mass_flow"]) / float(inlet_values["rho_feed"])
            #avg_density_inlet_ = inlet_values["rho_feed"] * ( self.MW[0] * y_inlet_ + self.MW[1]*(1-y_inlet_))
            #find velocity
            v_inlet_ = vol_flow_inlet_  / (self.column_area)
            #use velocity to find dPdZ from ergun equation
            dPdz_inlet_ = 1.75 * (1- self.bed_voidage) * float(inlet_values["rho_feed"]) * v_inlet_**2 / (self.bed_voidage**3 * self.particle_diameter) + 150 * inlet_values["mu"] * (1-self.bed_voidage)**2 *v_inlet_ / (self.bed_voidage**3 * self.particle_diameter**2)
            #use dPdZ to find P_inlet_
            P_inlet_ = P[0] - dPdz_inlet_ * (self.xWalls[int(self.nGhost)] - self.xCentres[int(self.nGhost)])
           

        elif self.inlet_type == "closed":
            # from lagrangian derivative function
            dPdz_inlet_ = 0
            P_inlet_ = P[self.nGhost]
            y_inlet_ = quadratic_extrapolation_derivative(self.xCentres[self.nGhost], y[0], self.xCentres[self.nGhost+1], y[1],
                                               self.xCentres[self.nGhost+2], y[2], self.xWalls[self.nGhost])
            v_inlet_ = 0
        
        return P_inlet_, y_inlet_, v_inlet_, dPdz_inlet_, P, y

    def outlet_BC(self, results_vector):
        # Implement outlet boundary conditions
        outlet_values = self.outlet_values
        P, y, n1, n2, F = self.data_prep(results_vector)

        if self.outlet_type == "pressure":
            # Mass flow inlet boundary condition
            y_outlet_ = quadratic_extrapolation(self.xCentres[-(self.nGhost+1)], y[-1], self.xCentres[-(self.nGhost+2)], y[-2],
                                               self.xCentres[-(self.nGhost+3)], y[-3], self.xWalls[-(self.nGhost+1)])
            
            P_outlet_ = outlet_values["outlet_pressure"]
            rho_g_outlet_ = P_outlet_ / (self.R * self.T_column)  # density at cell walls
            avg_density_outlet_ = rho_g_outlet_ * (self.MW[0] * y_outlet_ + self.MW[1]*(1-y_outlet_))
            dPdz_outlet_ = (P_outlet_ - P[-1]) / (self.xWalls[-1] - self.xCentres[-1])
            #### how do I calculate the desnity in the ergun equation at the outlet??
            a = 1.75 * (1- self.bed_voidage) * avg_density_outlet_ / (self.bed_voidage**3 * self.particle_diameter)
            b = 150 * outlet_values["mu"] * (1-self.bed_voidage)**2 / (self.bed_voidage**3 * self.particle_diameter**2)
            c = np.abs(dPdz_outlet_)
            v_outlet_ = - np.sign(dPdz_outlet_) * ((-b + np.sqrt(b**2 + 4*a*c)) / (2*a))
            
            

        elif self.outlet_type == "massflow":
            # Mass flow inlet boundary condition
            # need to calcaulte density from ideal gas law using a while loop
            vol_flow_outlet_ = outlet_values["outlet_mass_flow"] / outlet_values["outlet_density"] # would i calcualte this with idea gas law?
            v_outlet_ = vol_flow_outlet_ / self.column_area
            dPdz_outlet_ = 1.75 * (1- self.bed_voidage) * outlet_values["outlet_density"] * v_outlet_**2 / (self.bed_voidage * self.particle_diameter) + 150 * outlet_values["outlet_mu"] * (1-self.bed_voidage)**2 *v_outlet_ / (self.bed_voidage**3 * self.particle_diameter**2)
            P_outlet_ = P[-1] - dPdz_outlet_ * (self.xWalls[-1] - self.xCentres[-1])
            # how do I calculate y_outlet_ ????

        elif self.outlet_type == "closed":
            #uses langrangian derivate function
            P_outlet_ = P[-(self.nGhost+1)]
            y_outlet_ = quadratic_extrapolation_derivative(self.xCentres[-(self.nGhost+1)], y[-1], self.xCentres[-(self.nGhost+2)], y[-2],
                                               self.xCentres[-(self.nGhost+3)], y[-3], self.xWalls[-(self.nGhost+1)])
            v_outlet_= 0
            dPdz_outlet_= 0
        
        return P_outlet_, y_outlet_, v_outlet_, dPdz_outlet_, P, y

    def ghost_cell_calcs(self, results_vector):
        # Extrapolate to find ghost cell values for P, y
    
        P_inlet_, y_inlet_, v_inlet_, dPdz_inlet_, P, y = self.inlet_BC(results_vector)
        P_outlet_, y_outlet_, v_outlet_, dPdz_outlet_, P, y = self.outlet_BC(results_vector)


        # Left ghost cell, i = -1, using values at i = 1/2, 1, 2
        P_ghost_start = P[0] + (self.xCentres[0] - self.xCentres[1]) * (P[0] - P_inlet_) / (self.xCentres[self.nGhost] - self.xWalls[self.nGhost])
        P_ghost_end = P[-1] - (self.xCentres[-1] - self.xCentres[-self.nGhost-1]) * (P[-1] - P_outlet_) / (self.xCentres[-(self.nGhost+1)] - self.xWalls[-(self.nGhost+1)]) 
        P_centres_and_ghost_cells = np.concatenate((np.array([P_ghost_start]), P, np.array([P_ghost_end])))
        
        # Left ghost cell, i = -1, using values at i = 1/2, 1, 2 # if there is an issue, use linear extrapolation
        y_ghost_start = quadratic_extrapolation(self.xWalls[self.nGhost], y_inlet_, self.xCentres[self.nGhost], y[0], self.xCentres[self.nGhost+1], y[1], self.xCentres[0])
        y_ghost_end = quadratic_extrapolation(self.xWalls[-(self.nGhost+1)], y_outlet_, self.xCentres[-(self.nGhost+1)], y[-1], self.xCentres[-(self.nGhost+2)], y[-2], self.xCentres[-1])
        y_centres_and_ghost_cells = np.concatenate((np.array([y_ghost_start]), y,np.array([y_ghost_end])))                       
    
        return y_centres_and_ghost_cells, P_centres_and_ghost_cells, P_inlet_, y_inlet_, v_inlet_, dPdz_inlet_, P_outlet_, y_outlet_, v_outlet_, dPdz_outlet_ # should have N + 1 values for each
    
    def calculate_wall_values(self, results_vector):
        """ Calculate wall values for P, y, n1, n2 at cell walls
         Value at cell walls for z=0 and z=1 are from boundary conditions.
         Internal wall values for P, y, n1, n2 are calculated
         Use van Leer flux limiter function for y
         n1 and n2 wall values are not required because dn1/dt and dn2/dt can be calulated directly
         """
        Nx = int(self.num_cells)
        epsilon = 1.0e-10
        y_centres_and_ghost_cells, P_centres_and_ghost_cells, P_inlet_, y_inlet_, v_inlet_, dPdz_inlet_, P_outlet_, y_outlet_, v_outlet_, dPdz_outlet_ = self.ghost_cell_calcs(results_vector)
        

        #y vector at cell walls, from van leer flux limiter function
        R_r = (self.deltaZ[2:Nx+2] + self.deltaZ[1:Nx+1]) / self.deltaZ[1:Nx+1]
        r_r = ((y_centres_and_ghost_cells[1:Nx+1] - y_centres_and_ghost_cells[:Nx]) + epsilon)/((y_centres_and_ghost_cells[2:Nx+2]-y_centres_and_ghost_cells[1:Nx+1])+ epsilon)*(self.deltaZ[2:Nx+2]+self.deltaZ[1:Nx+1])/(self.deltaZ[1:Nx+1] + self.deltaZ[0:Nx])
        modified_van_leer = (0.5 * R_r * r_r + 0.5 * R_r * abs(r_r))/(R_r + r_r - 1 )
        flux_limiter = modified_van_leer / R_r
        y_walls = y_centres_and_ghost_cells[1:Nx+1] + flux_limiter * (y_centres_and_ghost_cells[2:Nx+2]-y_centres_and_ghost_cells[1:Nx+1])
        y_walls[-1] = y_outlet_
        y_walls = np.concatenate((np.array([y_inlet_]), y_walls))

        #calculate dP/dz at internal cell walls by linear interpolation
        dPdz_walls = np.array((P_centres_and_ghost_cells[1:Nx+2]- P_centres_and_ghost_cells[0:Nx+1]) / (self.xCentres[1:Nx+2] - self.xCentres[0:Nx+1]))
        dPdz_walls[0] = dPdz_inlet_  # set inlet pressure gradient
        dPdz_walls[-1] = dPdz_outlet_

        #calculate P at cell walls by interpolation
        P_walls = np.array(P_centres_and_ghost_cells[0:Nx+1] + dPdz_walls * (self.deltaZ[0:Nx+1]/2))
        P_walls[0] = P_inlet_  # set inlet pressure
        P_walls[-1] = P_outlet_  # set outlet pressure

        #calculate velocity vector at cell walls, from Ergun equation
        rho_g_walls = P_walls / (self.R * self.T_column)  # density at cell walls
        avg_density_walls = rho_g_walls * (self.MW[0] * y_walls + self.MW[1]*(1-y_walls))
        a = 1.75 * (1- self.bed_voidage) * avg_density_walls / (self.bed_voidage**3 * self.particle_diameter)
        b = 150 * self.mu * (1-self.bed_voidage)**2 / (self.bed_voidage**3 * self.particle_diameter**2)
        c = np.abs(dPdz_walls[:])
        dominant = b**2+4*a*c
        
        if np.any(dominant < 0):
            raise ValueError("Negative value under square root in velocity calculation. Check your inputs and boundary conditions.")
        v_walls = np.array((-b + np.sqrt(dominant)) / (2*a)) # should have N + 1 values 
        v_walls = np.multiply(-np.sign(dPdz_walls),v_walls)  # make sure velocity is in the correct direction
        v_walls[0] = v_inlet_  # set inlet velocity
        v_walls[-1] = v_outlet_  # set outlet velocity

        return P_walls, y_walls, v_walls


    def ODE_calculations(self, t, results_vector):
        # Calculate differential term at cell centres for the next time step
        # using wall values calculated in previous step

        Nx = int(self.num_cells)

        P_walls, y_walls, v_walls = self.calculate_wall_values(results_vector)
        P, y, n1, n2, F = self.data_prep(results_vector)


        k1 = 30 # s-1 linear driving force mass transfer constant
        dn1dt = k1*(adsorption_isotherm_1(P, y)-n1)
            
        #updating central values
        k2 = 30 # s-1 linear driving force mass transfer constant
        dn2dt = k2*(adsorption_isotherm_2(P, 1-y)-n2)

        """ q* and q need to be in concentration units i.e. mol / m3 ----- double check for future"""
        #pressure differential at cell centres
        
        dPdt = -(1/ self.deltaZ[1:-1]) * (v_walls[1:]*P_walls[1:] - v_walls[:Nx]*P_walls[:Nx]) - self.rho_b*self.R*self.T_column/self.bed_voidage * (dn1dt[:] + dn2dt[:]) # dP/dz at cell centres
        #dPdt =  - self.rho_b*self.R*self.T_column/self.bed_voidage * (dn1dt[:] + dn2dt[:])  # Initialize dPdt with zeros
        #mole fraction differential at cell centres
        dydt = -(1/ (self.deltaZ[1:-1] * P[:])) * (v_walls[1:]*P_walls[1:]*y_walls[1:] - v_walls[:Nx]*P_walls[:Nx]*y_walls[:Nx]) - self.rho_b*self.R*self.T_column/(self.bed_voidage*P[:]) * dn1dt[:] - y[:]/P[:] * dPdt[:]
        

        # these are my boundary values!!!!
        dF1dt = self.bed_voidage * self.column_area / (self.R * self.T_column) * v_walls[0]*P_walls[0] * y_walls[0]
        dF2dt = self.bed_voidage * self.column_area / (self.R * self.T_column) * v_walls[0]*P_walls[0] * (1 - y_walls[0])
        dF3dt = self.bed_voidage * self.column_area / (self.R * self.T_column) * v_walls[-1]*P_walls[-1] * y_walls[-1]
        dF4dt = self.bed_voidage * self.column_area / (self.R * self.T_column) * v_walls[-1]*P_walls[-1] * (1 - y_walls[-1])
        dFdt = np.array([dF1dt, dF2dt, dF3dt, dF4dt])  # inlet and outlet flow rates

        # Return the derivatives for the ODE solver
        # Return the mega-vector for the ODE solver: combine P, y, n1, n2, F vectors
        results_vector = np.concatenate((dPdt, dydt, dn1dt, dn2dt, dFdt))
        
        return results_vector


