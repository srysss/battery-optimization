import os
import pandas as pd
import py_dss_interface

# Initialize the DSS interface
dss = py_dss_interface.DSS()

# Compile the IEEE 13 Node Test Feeder
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")
print(f"✅ Current working directory: {script_path}")
dss.text(f"compile [{dss_file}]")
print(f"✅ IEEE 13 Node Test Feeder compiled: {dss_file}")

# Define file paths
pv_file = os.path.join(script_path, "pv_data", "PV_OutputPower_in_PU.txt")
load1_file = os.path.join(script_path, "load_data", "Load1_OpenDSS_kW_NoNormalization.txt")
load2_file = os.path.join(script_path, "load_data", "Load2_OpenDSS_kW_NoNormalization.txt")
load3_file = os.path.join(script_path, "load_data", "Load3_OpenDSS_kW_NoNormalization.txt")

# Read files
pv_data = pd.read_csv(pv_file, header=None, names=["pv"])
load1_data = pd.read_csv(load1_file, header=None, names=["LOAD1"])
load2_data = pd.read_csv(load2_file, header=None, names=["LOAD2"])
load3_data = pd.read_csv(load3_file, header=None, names=["LOAD3"])

# Combine load data into a single DataFrame
load_data = pd.concat([load1_data, load2_data, load3_data], axis=1)

# Define configuration for PV systems, batteries, and loads
pv_bess_config = [
    {"pv_name": "pv1", "bess_name": "Battery1", "load_name": "LOAD1", "bus_name": "632", "kV": 4.16, "pv_kVA": 10,
     "load_kW": 1, "pf": 0.95, "bess_kWRated": 10, "bess_kWhRated": 13.5},
    {"pv_name": "pv2", "bess_name": "Battery2", "load_name": "LOAD2", "bus_name": "671", "kV": 4.16, "pv_kVA": 10,
     "load_kW": 1, "pf": 0.95, "bess_kWRated": 10, "bess_kWhRated": 13.5},
    {"pv_name": "pv3", "bess_name": "Battery3", "load_name": "LOAD3", "bus_name": "633", "kV": 4.16, "pv_kVA": 10,
     "load_kW": 1, "pf": 0.95, "bess_kWRated": 10, "bess_kWhRated": 13.5}
]


# Add components to the DSS
def add_components_to_dss():
    for config in pv_bess_config:
        #pv_loadshape_file = os.path.join(script_path, "pv_data", "PV_OutputPower_in_PU.txt")
        dss.text(
            f"New Loadshape.{config['pv_name']} npts=48 minterval=30 mult=(file={pv_file})"
        )

        # Define PVSystem
        dss.text(
            f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} kVA={config['pv_kVA']} "
            f"Pmpp={config['pv_kVA']} irradiance=1 %cutin=0.1 %cutout=0.1 daily={config['pv_name']}"
        )

        # Add Load
        loadshape_file = os.path.join(script_path, "load_data",
                                      f"Load{config['load_name'][-1]}_OpenDSS_kW_NoNormalization.txt")
        dss.text(
            f"New Loadshape.{config['load_name']} npts=48 minterval=30 mult=(file={loadshape_file})"
        )

        dss.text(
            f"New Load.{config['load_name']} Phases=1 Bus1={config['bus_name']} kV={config['kV']} kW={config['load_kW']} PF={config['pf']} daily={config['load_name']}"
        )

        # Add Battery Storage
        dss.text(f"New Storage.{config['bess_name']} Phases=1 Bus1={config['bus_name']} kV={config['kV']} "
                 f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
                 f"kWhStored={config['bess_kWhRated'] * 0.5} %EffCharge=95 %EffDischarge=95 dispmode=EXTERNAL")

        # Add Monitors for PV, Storage, and Load
        dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 Mode=1")

        dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 Mode=1")

        # Monitor for SOC and other storage variables
        dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")


    dss.text("Solve")

add_components_to_dss()

print("✅ Components added to OpenDSS")
# Set simulation parameters for time step and daily mode
dss.text("set mode=daily")         # Set mode to daily simulation
dss.text("set stepsize=0.5h")      # Set time step size to 30 minutes (0.5 hour)
dss.text("set number=48")          # Set number of time steps to simulate (48 for one day with 30 min steps)

print("✅ Simulation parameters set: mode=daily, stepsize=0.5h, number=48")
# Battery control logic
battery_soc = {config['bess_name']: 50.0 for config in pv_bess_config}  # Initial SOC at 50%
time_step_hours = 0.5  # 30 minutes in hours

for i in range(len(pv_data)):
    print(f"\nTime step {i + 1}:")
    for config in pv_bess_config:
        bess_name = config['bess_name']
        bus_name = config['bus_name']
        pv_power_kW = pv_data["pv"][i] * config['pv_kVA']  # Convert PV p.u. value to kW
        load_kw = load_data[config['load_name']][i] * config['load_kW']  # Convert Load p.u. value to kW
        net_power = pv_power_kW - load_kw

        max_charge_power = config['bess_kWRated']
        max_discharge_power = config['bess_kWRated']
        battery_capacity_kWh = config['bess_kWhRated']

        print(f"BUS {bus_name} - PV Power: {pv_power_kW} kW, Load: {load_kw} kW, Net Power: {net_power} kW")
        print(f"Initial SOC: {battery_soc[bess_name]}%")

        if net_power > 0:
            if battery_soc[bess_name] < 80.0:
                charge_power = min(net_power, max_charge_power)
                soc_increment = (charge_power * time_step_hours) / battery_capacity_kWh * 100
                battery_soc[bess_name] = min(battery_soc[bess_name] + soc_increment, 100.0)
                dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={charge_power}")
                print(f"Charging: {charge_power} kW, New SOC: {battery_soc[bess_name]}%")
                dss.text("Solve")
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                print("Battery SOC too high, idling.")
                dss.text("Solve")
        elif net_power < 0:
            if battery_soc[bess_name] > 20.0:
                discharge_power = min(abs(net_power), max_discharge_power)
                max_possible_discharge = (battery_soc[bess_name] - 20.0) / 100 * battery_capacity_kWh / time_step_hours
                discharge_power = min(discharge_power, max_possible_discharge)
                soc_decrement = (discharge_power * time_step_hours) / battery_capacity_kWh * 100
                battery_soc[bess_name] = max(battery_soc[bess_name] - soc_decrement, 20.0)
                dss.text(f"Edit Storage.{bess_name} State=DISCHARGING kW={discharge_power}")
                print(f"Discharging: {discharge_power} kW, New SOC: {battery_soc[bess_name]}%")
                dss.text("Solve")
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                print("Battery SOC too low, idling.")
                dss.text("Solve")
        else:
            dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
            print("Net power is zero, idling.")
            dss.text("Solve")

    # Export Monitor data for the current step
    for config in pv_bess_config:
        dss.text("Solve")
        dss.text(f"Export Monitors {config['pv_name']}_mon")
        dss.text(f"Export Monitors {config['load_name']}_mon")
        dss.text(f"Export Monitors {config['bess_name']}_mon")

monitor_dir = os.path.abspath(os.path.join(script_path, "../feeders/13bus/"))
# Read exported data into pandas DataFrame for analysis
# Define Monitor file paths
pv_monitor_file = os.path.join(monitor_dir, f"IEEE13Nodeckt_Mon_{config['pv_name']}_mon_1.csv")
load_monitor_file = os.path.join(monitor_dir, f"IEEE13Nodeckt_Mon_{config['load_name']}_mon_1.csv")
storage_monitor_file = os.path.join(monitor_dir, f"IEEE13Nodeckt_Mon_{config['bess_name']}_mon_1.csv")

# Use Pandas to read the files
pv_monitor_data = pd.read_csv(pv_monitor_file)
load_monitor_data = pd.read_csv(load_monitor_file)
storage_monitor_data = pd.read_csv(storage_monitor_file)


dss.loadshapes.name = "LOAD1"
dss.text("plot loadshape object=LOAD1")
dss.loadshapes.name = "LOAD2"
dss.text("plot loadshape object=LOAD2")
dss.loadshapes.name = "LOAD3"
dss.text("plot loadshape object=LOAD3")

dss.loadshapes.name = "pv1"
dss.text("plot loadshape object=pv1")

dss.loadshapes.name = "pv2"
dss.text("plot loadshape object=pv2")

dss.loadshapes.name = "pv3"
dss.text("plot loadshape object=pv3")

# Print the first few rows of Monitor data
print(pv_monitor_data.head())
print(load_monitor_data.head())
print(storage_monitor_data.head())
