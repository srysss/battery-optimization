import os
import pandas as pd
import py_dss_interface
import matplotlib.pyplot as plt

# ========== 1. Initialization and previous setup ==========

# Initialize the DSS interface
dss = py_dss_interface.DSS()

# Compile the IEEE 13 Node Test Feeder
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")

dss.text(f"compile [{dss_file}]")

# ========== 2. Configure 10 sets of PV+Load+Storage ==========

# First 3 sets: use your originally defined configurations
pv_bess_config = [
    {
        "pv_name": "pv1",  "bess_name": "Battery1",
        "load_name": "LOAD1", "bus_name": "632.1",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    },
    {
        "pv_name": "pv2",  "bess_name": "Battery2",
        "load_name": "LOAD2", "bus_name": "671.1",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    },
    {
        "pv_name": "pv3",  "bess_name": "Battery3",
        "load_name": "LOAD3", "bus_name": "633.1",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    }
]

# Additional 7 sets to reach a total of 10 sets
additional_7 = [
    {
        "pv_name": "pv4",  "bess_name": "Battery4",
        "load_name": "LOAD4", "bus_name": "675.1",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    },
    {
        "pv_name": "pv5",  "bess_name": "Battery5",
        "load_name": "LOAD5", "bus_name": "645.2",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    },
    {
        "pv_name": "pv6",  "bess_name": "Battery6",
        "load_name": "LOAD6", "bus_name": "646.2",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    },
    {
        "pv_name": "pv7",  "bess_name": "Battery7",
        "load_name": "LOAD7", "bus_name": "652.1",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    },
    {
        "pv_name": "pv8",  "bess_name": "Battery8",
        "load_name": "LOAD8", "bus_name": "670.1",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    },
    {
        "pv_name": "pv9",  "bess_name": "Battery9",
        "load_name": "LOAD9", "bus_name": "634.1",
        "kV": 0.277, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    },
    {
        "pv_name": "pv10", "bess_name": "Battery10",
        "load_name": "LOAD10", "bus_name": "680.1",
        "kV": 2.4, "pv_kVA": 10, "load_kW": 1, "pf": 0.95,
        "bess_kWRated": 10, "bess_kWhRated": 13.5
    }
]

# Merge: original 3 sets + additional 7 sets = 10 sets total
pv_bess_config.extend(additional_7)
for config in pv_bess_config:
    pv_name = config["pv_name"]
    load_name = config["load_name"]

    # Construct PV file path, e.g. "pv_data/PV_OpenDSS_kW.txt"
    pv_filename = os.path.join(script_path, "pv_data", f"PV_OpenDSS_kW.txt")
    # Construct Load file path, e.g. "load_data/LOAD1_OpenDSS_kW_NoNormalization.txt"
    load_filename = os.path.join(
        script_path, "load_data",
        f"{load_name}_OpenDSS_kW.txt"
    )

    # Read PV file: assume a single column -> "pv"
    df_pv = pd.read_csv(pv_filename, header=None, names=["pv"])
    # Read Load file: assume a single column -> "load"
    df_load = pd.read_csv(load_filename, header=None, names=["load"])

    # Store back into config, so later you can directly use config["pv_data"] / config["load_data"]
    config["pv_data"] = df_pv
    config["load_data"] = df_load

def add_components_to_dss():
    """
    Iterate over pv_bess_config (total 10 sets), and for each set create:
     - PVSystem with its corresponding Loadshape
     - Load with its corresponding Loadshape
     - Storage
     - Monitor
    """
    for config in pv_bess_config:
        # --- 1) Loadshape for PV ---
        pv_file_txt = os.path.join(script_path, "pv_data", f"PV_OpenDSS_kW.txt")
        load_file_txt = os.path.join(script_path, "load_data", f"{config['load_name']}_OpenDSS_kW.txt")
        dss.text(
            f"New Loadshape.{config['pv_name']} npts=48 minterval=30 mult=(file={pv_file_txt}) useactual=yes"
        )

        # --- 2) Define PVSystem ---
        dss.text(
            f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} "
            f"kV={config['kV']} kVA={config['pv_kVA']} Pmpp={config['pv_kVA']} "
            f"irradiance=1 %cutin=0.1 %cutout=0.1 daily={config['pv_name']}"
        )


        dss.text(
            f"New Loadshape.{config['load_name']} npts=48 minterval=30 mult=(file={load_file_txt}) useactual=yes"
        )

        # --- 4) Create Load ---
        dss.text(
            f"New Load.{config['load_name']} Phases=1 Bus1={config['bus_name']} "
            f"kV={config['kV']} kW={config['load_kW']} PF={config['pf']} "
            f"daily={config['load_name']}"
        )

        # --- 5) Create Storage ---
        dss.text(
            f"New Storage.{config['bess_name']} Phases=1 Bus1={config['bus_name']} kV={config['kV']} "
            f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
            f"kWhStored={config['bess_kWhRated'] * 0.5} %EffCharge=95 %EffDischarge=95 dispmode=DEFAULT"
        )

        # --- 6) Monitors ---
        dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")


# Call the function to add components
add_components_to_dss()
print("✅ Added 10 sets of PV + Load + Storage to OpenDSS")

# ========== 3. Subsequent simulation and battery control logic ==========

battery_soc = {config['bess_name']: 50.0 for config in pv_bess_config}  # Initialize SOC=50% for all 10 batteries
time_step_hours = 0.5  # Each 30 minutes is one time step

dss.text("set mode=daily")
dss.text("set number=1")
dss.text("set stepsize=0.5h")

total_number = 48

time_steps = [i * 0.5 for i in range(total_number)]  # Used for x-axis (0, 0.5, 1.0, ..., 24)

# === Data structure: record voltage (p.u.) at each node and each time ===
node_voltages_over_time = {}

for i in range(total_number):
    print(f"\nTime step {i + 1}:")
    for config in pv_bess_config:
        bess_name = config['bess_name']
        pv_name = config['pv_name']
        load_name = config['load_name']
        pv_power_kW = config["pv_data"].iloc[i]["pv"]
        load_kw = config["load_data"].iloc[i]["load"]

        net_power = pv_power_kW - load_kw
        max_charge_power = config['bess_kWRated']
        max_discharge_power = config['bess_kWRated']
        battery_capacity_kWh = config['bess_kWhRated']

        print(f"  - {bess_name} | PV: {pv_power_kW:.2f}kW, Load: {load_kw:.2f}kW, Net: {net_power:.2f}kW")
        print(f"    Current SOC: {battery_soc[bess_name]:.2f}%")

        if net_power > 0:
            # ===== Try charging, stop precisely at 80% =====
            if battery_soc[bess_name] < 80.0:
                energy_needed_to_80 = ((80.0 - battery_soc[bess_name]) / 100.0) * battery_capacity_kWh
                actual_charge_power = min(net_power, max_charge_power)
                max_possible_energy = actual_charge_power * time_step_hours

                if max_possible_energy >= energy_needed_to_80:
                    charge_power_to_80 = energy_needed_to_80 / time_step_hours
                    charge_power_to_80 = min(charge_power_to_80, actual_charge_power)

                    battery_soc[bess_name] = 80.0
                    dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={charge_power_to_80} %stored={battery_soc[bess_name]}")
                    print(f"    Charging partially to 80%: {charge_power_to_80:.2f} kW, New SOC: {battery_soc[bess_name]:.2f}%")
                else:
                    soc_increment = (max_possible_energy / battery_capacity_kWh) * 100
                    battery_soc[bess_name] = min(battery_soc[bess_name] + soc_increment, 100.0)

                    dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={actual_charge_power} %stored={battery_soc[bess_name]}")
                    print(f"    Charging full half-hour: {actual_charge_power:.2f} kW, New SOC: {battery_soc[bess_name]:.2f}%")
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                print("    Battery SOC too high (>=80%), idling.")

        elif net_power < 0:
            # ===== Discharge logic =====
            if battery_soc[bess_name] > 20.0:
                discharge_power = min(abs(net_power), max_discharge_power)
                max_possible_discharge = (battery_soc[bess_name] - 20.0) / 100 * battery_capacity_kWh / time_step_hours
                discharge_power = min(discharge_power, max_possible_discharge)

                soc_decrement = (discharge_power * time_step_hours) / battery_capacity_kWh * 100
                battery_soc[bess_name] = max(battery_soc[bess_name] - soc_decrement, 20.0)

                dss.text(f"Edit Storage.{bess_name} State=DISCHARGING kW={-discharge_power} %stored={battery_soc[bess_name]}")
                print(f"    Discharging: {discharge_power:.2f} kW, New SOC: {battery_soc[bess_name]:.2f}%")
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                print("    Battery SOC too low (<=20%), idling.")
        else:
            # Net power = 0
            dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
            print("    Net power = 0, battery idling.")

    # Solve the simulation for the current time step
    dss.text("Solve")

    # Use circuit.nodes_names and circuit.buses_vmag_pu to get voltages of all nodes
    node_names_list = dss.circuit.nodes_names
    vmag_pu_list = dss.circuit.buses_vmag_pu

    # Store voltages into node_voltages_over_time
    for node_name, vmag in zip(node_names_list, vmag_pu_list):
        node_voltages_over_time.setdefault(node_name, []).append(vmag)

# Export all Monitors
dss.text("Export monitors all")
print("✅ Simulation complete and monitors exported.")
dss.text("set markpvsystems=yes")
dss.text("set markStorage=yes")
dss.text("plot circuit Power max=2000 y y labels=yes buswidth=2 C1=$FF0000 C2=$00FF00 markerpvs=3 markerstorage=2 markersizepvs=5 markersizestorage=8 offset=1")
# ========== 3. Read and plot Monitor data ==========

monitor_dir = os.path.dirname(dss_file)

for config in pv_bess_config:
    pv_name = config["pv_name"]
    load_name = config["load_name"]
    bess_name = config["bess_name"]

    # 3.1 Read PV Monitor
    pv_monitor_filename = f"IEEE13Nodeckt_Mon_{pv_name}_mon_1.csv"
    pv_monitor_path = os.path.join(monitor_dir, pv_monitor_filename)
    if os.path.exists(pv_monitor_path):
        df_pv = pd.read_csv(pv_monitor_path)
        print(f"\nReading PV monitor file: {pv_monitor_path}")
        print("Columns:", df_pv.columns.tolist())

        if ' P1 (kW)' in df_pv.columns:
            plt.figure(figsize=(8, 4))
            plt.plot(time_steps, -df_pv[' P1 (kW)'], label=f"{pv_name} - P1(kW)")
            plt.xlabel("Time (hours)")
            plt.ylabel("Power (kW)")
            plt.title(f"PV Monitor: {pv_name}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # 3.2 Read Load Monitor
    load_monitor_filename = f"IEEE13Nodeckt_Mon_{load_name}_mon_1.csv"
    load_monitor_path = os.path.join(monitor_dir, load_monitor_filename)
    if os.path.exists(load_monitor_path):
        df_load = pd.read_csv(load_monitor_path)
        print(f"\nReading Load monitor file: {load_monitor_path}")
        print("Columns:", df_load.columns.tolist())

        if ' P1 (kW)' in df_load.columns:
            plt.figure(figsize=(8, 4))
            plt.plot(time_steps, df_load[' P1 (kW)'], color='orange', label=f"{load_name} - P1(kW)")
            plt.xlabel("Time (hours)")
            plt.ylabel("Power (kW)")
            plt.title(f"Load Monitor: {load_name}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # 3.3 Read Battery Monitor
    storage_monitor_filename = f"IEEE13Nodeckt_Mon_{bess_name}_mon_1.csv"
    storage_monitor_path = os.path.join(monitor_dir, storage_monitor_filename)
    if os.path.exists(storage_monitor_path):
        df_batt = pd.read_csv(storage_monitor_path)
        print(f"\nReading Storage monitor file: {storage_monitor_path}")
        print("Columns:", df_batt.columns.tolist())

        if ' %kW Stored' in df_batt.columns:
            plt.figure(figsize=(8, 4))
            plt.plot(time_steps, df_batt[' %kW Stored'], color='green', label=f"{bess_name} - SOC(%)")
            plt.xlabel("Time (hours)")
            plt.ylabel("SOC (%)")
            plt.title(f"Battery Monitor: {bess_name}")
            plt.legend()
            plt.tight_layout()
            plt.show()

        if ' kW output' in df_batt.columns:
            plt.figure(figsize=(8, 4))
            plt.plot(time_steps, df_batt[' kW output'], color='red', label=f"{bess_name} - kW output")
            plt.xlabel("Time (hours)")
            plt.ylabel("Power (kW)")
            plt.title(f"Battery Monitor: {bess_name}")
            plt.legend()
            plt.tight_layout()
            plt.show()

# ========== 4. Plot node voltages by phase ==========

# Prepare three dictionaries, classifying nodes into phase 1/2/3
phase1_dict = {}
phase2_dict = {}
phase3_dict = {}

for node_name, v_list in node_voltages_over_time.items():
    # Here we assume node_name format like "632.1" or "650.3"
    if node_name.endswith(".1"):
        phase1_dict[node_name] = v_list
    elif node_name.endswith(".2"):
        phase2_dict[node_name] = v_list
    elif node_name.endswith(".3"):
        phase3_dict[node_name] = v_list
    else:
        # For more complex cases, like "650.1.2.3", additional handling or ignoring may be needed
        pass

# --- Phase 1 ---
plt.figure(figsize=(10, 5))
for node_name, voltages in phase1_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("Phase 1 Node Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Phase 2 ---
plt.figure(figsize=(10, 5))
for node_name, voltages in phase2_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("Phase 2 Node Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Phase 3 ---
plt.figure(figsize=(10, 5))
for node_name, voltages in phase3_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("Phase 3 Node Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ========== 5. Loadshape plots ==========

for config in pv_bess_config:
    dss.loadshapes.name = config['load_name']
    dss.text(f"plot loadshape object={config['load_name']}")

    dss.loadshapes.name = "pv1"
    dss.text("plot loadshape object=pv1")
