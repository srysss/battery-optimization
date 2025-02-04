import os
import pandas as pd
import py_dss_interface
import csv
import matplotlib.pyplot as plt

# Initialize the DSS interface
dss = py_dss_interface.DSS()

# Compile the IEEE 13 Node Test Feeder
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")
print(f"✅ Current working directory: {script_path}")
dss.text(f"compile [{dss_file}]")
print(f"✅ IEEE 13 Node Test Feeder compiled: {dss_file}")




# Load new PV and load data
try:
    pv_data = pd.read_excel(os.path.join(script_path, "real_solar_radiation.xlsx"))
    load_data = pd.read_excel(os.path.join(script_path, "real_load_profile.xlsx"))
    print("✅ Solar Radiation data loaded successfully")
    print("✅ Load Profile data loaded successfully")
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    exit(1)

# Define the configuration of PV and energy storage systems
pv_bess_config = [
    {"pv_name": "PV1", "bess_name": "Battery1", "kVA_rated": 10, "bus_name": "671"},
    {"pv_name": "PV2", "bess_name": "Battery2", "kVA_rated": 10, "bus_name": "632"},
    {"pv_name": "PV3", "bess_name": "Battery3", "kVA_rated": 10, "bus_name": "633"},
]

# Add PV systems to OpenDSS
for config in pv_bess_config:
    dss.text(f"New PVSystem.{config['pv_name']} phases=3 bus1={config['bus_name']} kV=0.4 kVA={config['kVA_rated']} irrad=1 Pmpp={config['kVA_rated']}")

# Add three loads to OpenDSS
for i in range(3):
    bus_name = pv_bess_config[i]['bus_name']
    dss.text(f"New Load.Load{i+1} phases=3 bus1={bus_name} kV=0.4 kW=0")

# Define battery capacity and maximum discharge power
battery_capacity_kWh = 10  # Tesla Powerwall 2 capacity 13.5kW
max_discharge_power = 3.0  # Maximum discharge power is 3 kW

# Add storage systems to OpenDSS
for config in pv_bess_config:
    dss.text(
        f"New Storage.{config['bess_name']} "
        f"Phases=3 Bus1={config['bus_name']} "
        f"kV=0.4 kWRated={max_discharge_power} kWhRated={battery_capacity_kWh} "
        f"%EffCharge=95 %EffDischarge=95 dispmode=EXTERNAL"
    )

dss.text("solve")
#Get the exact location of the added element and plot in opendss
dss.text("set markpvsystems=yes")
dss.text("set markStorage=yes")
dss.text("plot circuit Power max=2000 y y labels=yes buswidth=2 C1=$FF0000 C2=$00FF00 markerpvs=3 markerstorage=2 markersizepvs=5 markersizestorage=8 offset=1")

# set and define
# Set voltage limits for the grid
v_min_limit = 0.94
v_max_limit = 1.1
# Define battery capacity and time step
battery_capacity_kWh = 10.0  # Total battery capacity is 10 kWh
time_step_hours = 0.5  # Time step is half an hour
# Initialize the initial SOC (State of Charge) of batteries
battery_soc = {"Battery1": 50.0, "Battery2": 50.0, "Battery3": 50.0}  # Initial SOC is 50%
# Set the maximum charge and discharge power limits
max_discharge_power = 3.0  # Maximum discharge power is 3 kW
max_charge_power = 3.0  # Maximum charge power is 3 kW



# Specify the path for the output file
output_file = os.path.join(script_path, "simulation_results.csv")
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Time", "PV_Name", "PV_Power_kW", "BESS_Name", "BESS_Power_kW", "SOC", "Load_Demand_kW", "Grid_Support_kW", "Voltage_pu"
    ])

    ### For plot
    time_series = []
    soc_series = {"Battery1": [], "Battery2": [], "Battery3": []}
    pv_power_series = {"PV1": [], "PV2": [], "PV3": []}
    load_series = []
    voltage_series = {bus: [] for bus in dss.circuit.buses_names}



    # Run simulation for each half-hour time step
    for idx, row in load_data.iterrows():   #Data update at each time step
        time = row['tstp']
        time_series.append(time)
        total_load = 0

        for config in pv_bess_config:
            pv_name = config["pv_name"]
            bess_name = config["bess_name"]
            bus_name = config["bus_name"]

            # Get the solar radiation data for the current half-hour
            solar_radiation = pv_data.loc[idx, "Solar Radiation"]

            # Assume a rated power of 10 kW for PV systems and calculate the PV generation power
            pv_power_kW = 10 * (solar_radiation / 1000)  # Scale accordingly
            pv_power_series[pv_name].append(pv_power_kW)

            # Update the irradiance for the PV system in OpenDSS
            dss.text(f"Edit PVSystem.{pv_name} irradiance={pv_power_kW / config['kVA_rated']}")

            # Get the load data for the current time step
            load_kw = row[f"Load{pv_bess_config.index(config) + 1}(kWh/hh)"] * 2  # Convert half-hour data to kW
            total_load += load_kw

            # Update the load power in OpenDSS
            dss.text(f"Edit Load.Load{idx + 1} kW={load_kw}")

            # Calculate net power
            net_power = pv_power_kW - load_kw

            # Solve the power flow
            dss.text("solve")

            # Get voltage values for all buses
            all_bus_voltages = dss.circuit.buses_vmag_pu

            # Default battery state is IDLE
            battery_state = "IDLE"
            battery_power = 0.0
            grid_support = 0.0

            time_step_hours = 0.5  # half hour

            # Calculate charging or discharging strategy for each battery
            # Calculate charging or discharging strategy for each battery
            if net_power > 0:
                # Excess PV generation, charge the battery
                if battery_soc[bess_name] < 80.0:
                    charge_power = min(net_power, max_charge_power)
                    soc_increment = (charge_power * time_step_hours) / battery_capacity_kWh * 100  # Calculate SOC increment
                    battery_soc[bess_name] = min(battery_soc[bess_name] + soc_increment, 100.0)
                    battery_state = "CHARGING"
                    battery_power = charge_power
                    dss.text(f"Edit Storage.{bess_name} kW={battery_power}")  # Update OpenDSS
            elif net_power < 0:
                # Insufficient PV generation, discharge the battery
                if battery_soc[bess_name] > 20.0:
                    discharge_power = min(abs(net_power), max_discharge_power)
                    max_possible_discharge = (battery_soc[bess_name] - 20.0) / 100 * battery_capacity_kWh / time_step_hours
                    discharge_power = min(discharge_power, max_possible_discharge)
                    soc_decrement = (discharge_power * time_step_hours) / battery_capacity_kWh * 100  # Calculate SOC decrement
                    battery_soc[bess_name] = max(battery_soc[bess_name] - soc_decrement, 20.0)
                    battery_state = "DISCHARGING"
                    battery_power = discharge_power
                    grid_support = abs(net_power) - discharge_power
                else:
                    # When the battery reaches minimum SOC, the grid supports the entire load
                    grid_support = abs(net_power)
                    battery_power = 0  # Ensure battery is not providing power

                dss.text(f"Edit Storage.{bess_name} kW={battery_power}")  # Update OpenDSS
            else:
                # No net power, battery remains idle
                battery_state = "IDLE"
                battery_power = 0
                grid_support = 0

            # Record SOC for each step
            soc_series[bess_name].append(battery_soc[bess_name])

            # Solve the OpenDSS system
            dss.text("solve")

            # Write results to the CSV file
            writer.writerow([
                time, pv_name, pv_power_kW, bess_name, battery_power, battery_soc[bess_name], load_kw, grid_support, all_bus_voltages[dss.circuit.buses_names.index(bus_name)]
            ])

            # Print simulation information
            print(
                f"{time}: {pv_name} Generation = {pv_power_kW:.2f} kW, Voltage = {all_bus_voltages[dss.circuit.buses_names.index(bus_name)]:.4f}, Load Demand = {load_kw:.2f} kW, Grid Support = {grid_support:.2f} kW"
            )
            if battery_state == "DISCHARGING":
                print(f"{time}: {bess_name} is discharging {battery_power:.2f} kW, SOC = {battery_soc[bess_name]:.2f}%")
            elif battery_state == "CHARGING":
                print(f"{time}: {bess_name} is charging {battery_power:.2f} kW, SOC = {battery_soc[bess_name]:.2f}%")
            else:
                print(f"{time}: {bess_name} is idle, SOC = {battery_soc[bess_name]:.2f}%")

        load_series.append(total_load)
        for bus in dss.circuit.buses_names:
            voltage_series[bus].append(all_bus_voltages[dss.circuit.buses_names.index(bus)])

# Plot SOC over time
plt.figure()
for battery, soc in soc_series.items():
    plt.plot(time_series, soc, label=f"{battery} SOC")
plt.title("Battery SOC Over Time")
plt.xlabel("Time")
plt.ylabel("SOC (%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot PV power generation over time
plt.figure()
for pv, power in pv_power_series.items():
    plt.plot(time_series, power, label=f"{pv} Power")
plt.title("PV Power Over Time")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot load demand over time
plt.figure()
plt.plot(time_series, load_series, label="Total Load Demand")
plt.title("Load Demand Over Time")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot bus voltages over time
plt.figure()
for bus, voltage in voltage_series.items():
    plt.plot(time_series, voltage, label=f"Bus {bus} Voltage")
plt.title("Bus Voltages Over Time")
plt.xlabel("Time")
plt.ylabel("Voltage (pu)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
