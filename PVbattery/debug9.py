import os
import pandas as pd
import py_dss_interface
import matplotlib.pyplot as plt

# Initialize the DSS interface
dss = py_dss_interface.DSS()

# Compile the IEEE 13 Node Test Feeder
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")
dss.text(f"compile [{dss_file}]")

# Define file paths
load1_file = os.path.join(script_path, "load_data", "Load1_OpenDSS_kW_NoNormalization.txt")

# Read files
load1_data = pd.read_csv(load1_file, header=None, names=["LOAD1"])

# Print the first few rows of load1_data to verify
print("Loadshape Data (first few rows):")
print(load1_data.head())

# Define Loadshape and Load
loadshape_file = os.path.join(script_path, "load_data", "Load1_OpenDSS_kW_NoNormalization.txt")
dss.text(f"New Loadshape.LOAD1 npts=48 minterval=30 mult=(file={loadshape_file}) action=normalize")
dss.text(f"New Load.LOAD1 Phases=1 Bus1=632 kV=4.16 kW=1 PF=0.95 daily=LOAD1")

# Inspect available methods for LoadShapes object
print("Available methods for LoadShapes object:", dir(dss.loadshapes))

# Access the names of all load shapes
print("Defined Loadshape:", dss.loadshapes.names)

# Print Load properties to verify
dss.loads.name = "LOAD1"
print("Load properties:")
print("Name:", dss.loads.name)
print("Bus1:", dss.loads.bus1)
print("kV:", dss.loads.kv)
print("kW:", dss.loads.kw)
print("PF:", dss.loads.pf)
print("Daily:", dss.loads.daily)

# Define Monitor
dss.text("New Monitor.LOAD1_mon Element=Load.LOAD1 Terminal=1 ppolar=no Mode=1")

# Set simulation parameters
dss.text("set mode=daily")
dss.text("set stepsize=0.5h")
dss.text("set number=48")
dss.text("Solve")

# Export Monitor data
dss.text("Export monitors all")

# Read Monitor data
monitor_dir = os.path.abspath(os.path.join(script_path, "../feeders/13bus/"))
load_monitor_file = os.path.join(monitor_dir, "IEEE13Nodeckt_Mon_LOAD1_mon_1.csv")
load_monitor_data = pd.read_csv(load_monitor_file)

# Print columns to check for extra spaces or mismatches
print("Columns in load_monitor_data:", load_monitor_data.columns)

# Strip extra spaces from column names
load_monitor_data.columns = load_monitor_data.columns.str.strip()

# Plot Loadshape and Monitor data for comparison
plt.figure(figsize=(10, 5))
plt.plot(load1_data["LOAD1"], label="Input Loadshape", marker='o')
plt.plot(load_monitor_data["P1 (kW)"], label="Monitor Load Output", marker='x')
plt.xlabel("Time Step")
plt.ylabel("Load (kW)")
plt.title("Comparison of Input Loadshape and Monitor Load Output")
plt.legend()
plt.grid(True)
plt.show()

# Print statistics to check for consistency
print("Loadshape Data Statistics:")
print(load1_data.describe())

print("\nMonitor Data Statistics:")
print(load_monitor_data["P1 (kW)"].describe())