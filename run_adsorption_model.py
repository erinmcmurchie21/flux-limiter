
import numpy as np
from column_model import AdsorptionColumn, adsorption_isotherm_1, adsorption_isotherm_2
from scipy.integrate import solve_ivp
from non_uniform_grid import create_non_uniform_grid
import matplotlib.pyplot as plt

bed_properties = {
    "bed_voidage" : 0.4,  # Example value for bed voidage
    "particle_diameter" : 0.0002,  # Example value for particle diameter in meters
    "bed_diameter" : 0.1,  # Example value for bed diameter in meters
    "column_area" : 0.1**2 * np.pi / 4 , # Cross-sectional area of the column
    "R" : 8.314,  # Universal gas constant in J/(mol*K)
    "T_column" : 298,  # Column temperature in Kelvin
    "rho_b" : 1000, # Example value for bed density in kg/m^3
    "bed_length" : 1.8,  # Example value for bed length in meters
}

column_grid = create_non_uniform_grid()

inlet_values = {
    "inlet_type" : "mass_flow", 
    "velocity" : 1.0,  # Example value for superficial velocity in m/s
    "feed_mass_flow" : (0.1*bed_properties["column_area"] * 1.2),  # Example value for feed mass flow in kg/s
    "feed_temperature" : 298,  # Example value for feed temperature in Kelvin
    "feed_pressure" : 101325,  # Example value for feed pressure in Pa
    "rho_feed" : 1.2,  # Example value for feed density in kg/m^3
    "mu" : 1.8e-5,  # Example value for feed viscosity in Pa.s
    "y_feed_value" : 0.05,  # Example value for feed mole fraction
}

outlet_values = {
    "outlet_type" : "pressure",
    "outlet_pressure" : 101325,  # Example value for outlet pressure in Pa
    "outlet_density" : 1.2,  # Example value for outlet density in kg/m^3
    "mu" : 1.8e-5,  # Example value for outlet viscosity in Pa.s
    }
P = np.ones(column_grid["num_cells"]) * 101325  # Example pressure vector in Pa
y = np.ones(column_grid["num_cells"]) * 1e-6
n1 = np.ones(column_grid["num_cells"]) * adsorption_isotherm_1(P, y) # Example concentration vector in mol/m^3
n2 = np.ones(column_grid["num_cells"]) * adsorption_isotherm_2(P, 1-y) # Example concentration vector in mol/m^3
F = np.ones(4) * 0
initial_conditions = np.concatenate([P,y,n1,n2,F])
# Create an instance of the AdsorptionColumn class
column_model = AdsorptionColumn(column_grid, bed_properties, inlet_values, outlet_values)

# Implement solver
t_span = [0, 10]  # Time span for the ODE solver
rtol = 1e-6
atol_P = 1e-2 * np.ones(len(P))
atol_y = 1e-9 * np.ones(len(y))
atol_n1 = 1e-3 * np.ones(len(n1))
atol_n2 = 1e-3 * np.ones(len(n2))
atol_F = 1e-4 * np.ones(len(F))
atol_array = np.concatenate([atol_P, atol_y, atol_n1, atol_n2, atol_F])
output_matrix = solve_ivp(column_model.ODE_calculations, t_span, initial_conditions, method='BDF', rtol=rtol, atol=atol_array)

    
print(output_matrix)

P_result = output_matrix.y[0:column_grid["num_cells"]]
y_result = output_matrix.y[column_grid["num_cells"]:2*column_grid["num_cells"]]
n1_result = output_matrix.y[2*column_grid["num_cells"]:3*column_grid["num_cells"]]
n2_result = output_matrix.y[3*column_grid["num_cells"]:4*column_grid["num_cells"]]
F_result = output_matrix.y[4*column_grid["num_cells"]:]

P_plot = [P_result[0], P_result[14], P_result[29]]
y_plot = [y_result[0], y_result[14], y_result[29]]
n1_plot = [n1_result[0], n1_result[14], n1_result[29]]
n2_plot = [n2_result[0], n2_result[14], n2_result[29]]
print("Time points =", len(output_matrix.t))

# Create the plot
plt.figure(figsize=(6, 4))
data1 = P_result[0]
data2 = P_result[14]
data3 = P_result[29]

# Plot each dataset
plt.plot(output_matrix.t, data1, label='First node', linewidth=2, marker='o', markersize=3)
plt.plot(output_matrix.t, data2, label='Central node', linewidth=2, marker='s', markersize=3)
plt.plot(output_matrix.t, data3, label='Final node', linewidth=2, marker='^', markersize=3)

# Customize the plot
plt.title('Pressure against time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Pressure', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Show the plot
plt.show()

# Create the plot
plt.figure(figsize=(6, 4))
data1 = y_result[0]
data2 = y_result[14]
data3 = y_result[29]

# Plot each dataset
plt.plot(output_matrix.t, data1, label='First node', linewidth=2, marker='o', markersize=3)
plt.plot(output_matrix.t, data2, label='Central node', linewidth=2, marker='s', markersize=3)
plt.plot(output_matrix.t, data3, label='Final node', linewidth=2, marker='^', markersize=3)

# Customize the plot
plt.xlabel('Time', fontsize=12)
plt.ylabel('Gas phase mole fraction', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Show the plot
plt.show()
