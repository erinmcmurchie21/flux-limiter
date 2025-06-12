
import numpy as np
from column_model import AdsorptionColumn, adsorption_isotherm_1, adsorption_isotherm_2
from scipy.integrate import solve_ivp
from non_uniform_grid import create_non_uniform_grid

bed_properties = {
    "bed_voidage" : 0.4,  # Example value for bed voidage
    "particle_diameter" : 0.0003,  # Example value for particle diameter in meters
    "bed_diameter" : 0.1,  # Example value for bed diameter in meters
    "column_area" : 0.1**2 * np.pi / 4 , # Cross-sectional area of the column
    "R" : 8.314,  # Universal gas constant in J/(mol*K)
    "T_column" : 308,  # Column temperature in Kelvin
    "rho_b" : 1000, # Example value for bed density in kg/m^3
    "bed_length" : 1.8,  # Example value for bed length in meters
}

column_grid = create_non_uniform_grid()

inlet_values = {
    "inlet_type" : "mass_flow", 
    "feed_mass_flow" : 0.001,  # Example value for feed mass flow in kg/s
    "feed_temperature" : 308,  # Example value for feed temperature in Kelvin
    "feed_pressure" : 101325,  # Example value for feed pressure in Pa
    "rho_feed" : 1.2,  # Example value for feed density in kg/m^3
    "mu" : 1.8e-5,  # Example value for feed viscosity in Pa.s
    "y_feed_value" : 1.0,  # Example value for feed mole fraction
}

outlet_values = {
    "outlet_type" : "pressure",
    "outlet_pressure" : 101325,  # Example value for outlet pressure in Pa
    "outlet_density" : 1.2,  # Example value for outlet density in kg/m^3
    "mu" : 1.8e-5,  # Example value for outlet viscosity in Pa.s
    }
P = np.ones(column_grid["num_cells"]) * 101325  # Example pressure vector in Pa
y = np.ones(column_grid["num_cells"]) * 1e-6
n1 = adsorption_isotherm_1(P, y) # Example concentration vector in mol/m^3
n2 = adsorption_isotherm_2(P, 1-y) # Example concentration vector in mol/m^3
F = np.ones(4) * 0
initial_conditions = np.concatenate([P,y,n1,n2,F])
# Create an instance of the AdsorptionColumn class
column_model = AdsorptionColumn(column_grid, bed_properties, inlet_values, outlet_values)

# Implement solver
t_span = [0, 10]  # Time span for the ODE solver
rtol = 1e-6
atol_P = 1e-2 * np.ones(len(P))
atol_y = 1e-6 * np.ones(len(y))
atol_n1 = 1e-3 * np.ones(len(n1))
atol_n2 = 1e-3 * np.ones(len(n2))
atol_F = 1e-4 * np.ones(len(F))
atol_array = np.concatenate([atol_P, atol_y, atol_n1, atol_n2, atol_F])
output_matrix = solve_ivp(column_model.ODE_calculations, t_span, initial_conditions, method='BDF', rtol=rtol, atol=atol_array)

    
print(output_matrix)