import os
import pandas as pd
import py_dss_interface
import matplotlib.pyplot as plt

# ========== 1. 初始化及前面已有的代码 ==========

# Initialize the DSS interface
dss = py_dss_interface.DSS()

# Compile the IEEE 13 Node Test Feeder
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")
print(f"✅ Current working directory: {script_path}")
dss.text(f"compile [{dss_file}]")
print(f"✅ IEEE 13 Node Test Feeder compiled: {dss_file}")

# Define file paths
pv_file = os.path.join(script_path, "pv_data", "PV_OutputPower_OpenDSS.txt")
load1_file = os.path.join(script_path, "load_data", "Load1_OpenDSS_kW_NoNormalization.txt")
load2_file = os.path.join(script_path, "load_data", "Load2_OpenDSS_kW_NoNormalization.txt")
load3_file = os.path.join(script_path, "load_data", "Load3_OpenDSS_kW_NoNormalization.txt")

# Read files
pv_data = pd.read_csv(pv_file, header=None, names=["pv"])
load1_data = pd.read_csv(load1_file, header=None, names=["LOAD1"])
load2_data = pd.read_csv(load2_file, header=None, names=["LOAD2"])
load3_data = pd.read_csv(load3_file, header=None, names=["LOAD3"])

# Combine load data
load_data = pd.concat([load1_data, load2_data, load3_data], axis=1)

# Define configuration
pv_bess_config = [
    {"pv_name": "pv1", "bess_name": "Battery1", "load_name": "LOAD1", "bus_name": "632.1", "kV": 2.4, "pv_kVA": 10,
     "load_kW": 1, "pf": 0.95, "bess_kWRated": 10, "bess_kWhRated": 13.5},
    {"pv_name": "pv2", "bess_name": "Battery2", "load_name": "LOAD2", "bus_name": "671.1", "kV": 2.4, "pv_kVA": 10,
     "load_kW": 1, "pf": 0.95, "bess_kWRated": 10, "bess_kWhRated": 13.5},
    {"pv_name": "pv3", "bess_name": "Battery3", "load_name": "LOAD3", "bus_name": "633.1", "kV": 2.4, "pv_kVA": 10,
     "load_kW": 1, "pf": 0.95, "bess_kWRated": 10, "bess_kWhRated": 13.5}
]

def add_components_to_dss():
    for config in pv_bess_config:
        # Create Loadshape for PV
        dss.text(f"New Loadshape.{config['pv_name']} npts=48 minterval=30 mult=(file={pv_file}) useactual=yes")

        # Define PVSystem
        dss.text(
            f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} kVA={config['pv_kVA']} "
            f"Pmpp={config['pv_kVA']} irradiance=1 %cutin=0.1 %cutout=0.1 daily={config['pv_name']}"
        )

        # Add Loadshape for Load
        loadshape_file = os.path.join(script_path, "load_data",
                                      f"Load{config['load_name'][-1]}_OpenDSS_kW_NoNormalization.txt")
        dss.text(
            f"New Loadshape.shape{config['load_name']} npts=48 minterval=30 mult=(file={loadshape_file}) useactual=yes"
        )

        # Add Load
        dss.text(
            f"New Load.{config['load_name']} Phases=1 Bus1={config['bus_name']} kV={config['kV']} "
            f"kW={config['load_kW']} PF={config['pf']} daily=shape{config['load_name']}"
        )

        # Add Battery Storage
        dss.text(
            f"New Storage.{config['bess_name']} Phases=1 Bus1={config['bus_name']} kV={config['kV']} "
            f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
            f"kWhStored={config['bess_kWhRated'] * 0.5} %EffCharge=95 %EffDischarge=95 dispmode=DEFAULT"
        )

        # Add Monitors for PV, Storage, and Load
        dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")


add_components_to_dss()
print("✅ Components added to OpenDSS")

# ========== 2. 仿真与电池控制逻辑 ==========

battery_soc = {config['bess_name']: 50.0 for config in pv_bess_config}  # Initial SOC at 50%
time_step_hours = 0.5  # 30 minutes in hours

# 设置仿真模式
dss.text("set mode=daily")
dss.text("set stepsize=0.5h")
dss.text("set number=1")

total_number = 48
time_steps = [i * 0.5 for i in range(total_number)]  # 用来做 x 轴 (0, 0.5, 1.0, ..., 24)

# === 存储结构：记录每个 node 在每个时刻的电压( p.u. ) ===
node_voltages_over_time = {}

for i in range(total_number):
    print(f"\nTime step {i + 1}:")
    for config in pv_bess_config:
        bess_name = config['bess_name']
        pv_name = config['pv_name']
        load_name = config['load_name']
        pv_power_kW = pv_data["pv"][i]
        load_kw = load_data[config['load_name']][i]

        net_power = pv_power_kW - load_kw
        max_charge_power = config['bess_kWRated']
        max_discharge_power = config['bess_kWRated']
        battery_capacity_kWh = config['bess_kWhRated']

        print(f"  - {bess_name} | PV: {pv_power_kW:.2f}kW, Load: {load_kw:.2f}kW, Net: {net_power:.2f}kW")
        print(f"    Current SOC: {battery_soc[bess_name]:.2f}%")

        if net_power > 0:
            # ===== 尝试充电，精确到达 80% 就停 =====
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
            # ===== 放电逻辑 =====
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

    # 仿真求解当前时间步
    dss.text("Solve")

    # 用 circuit.nodes_names 和 circuit.buses_vmag_pu 获取所有节点电压
    node_names_list = dss.circuit.nodes_names
    vmag_pu_list = dss.circuit.buses_vmag_pu

    # 存储电压到 node_voltages_over_time
    for node_name, vmag in zip(node_names_list, vmag_pu_list):
        node_voltages_over_time.setdefault(node_name, []).append(vmag)

# 导出所有 Monitor
dss.text("Export monitors all")
print("✅ Simulation complete and monitors exported.")

# ========== 3. 读取并绘制 Monitor 数据 ==========

monitor_dir = os.path.dirname(dss_file)

for config in pv_bess_config:
    pv_name = config["pv_name"]
    load_name = config["load_name"]
    bess_name = config["bess_name"]

    # 3.1 读取 PV Monitor
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

    # 3.2 读取 Load Monitor
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

    # 3.3 读取 Battery Monitor
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

# ========== 4. 分相绘制节点电压 ==========

# 准备三个字典，把节点区分到相1/2/3
phase1_dict = {}
phase2_dict = {}
phase3_dict = {}

for node_name, v_list in node_voltages_over_time.items():
    # 这里假设 node_name 格式如 "632.1" 或 "650.3"
    if node_name.endswith(".1"):
        phase1_dict[node_name] = v_list
    elif node_name.endswith(".2"):
        phase2_dict[node_name] = v_list
    elif node_name.endswith(".3"):
        phase3_dict[node_name] = v_list
    else:
        # 如果有更复杂的情况，比如 "650.1.2.3" 等，需要另行处理或忽略
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

# ========== 5. Loadshape图 ==========

dss.loadshapes.name = "shapeLOAD1"
dss.text("plot loadshape object=shapeLOAD1")
dss.loadshapes.name = "shapeLOAD2"
dss.text("plot loadshape object=shapeLOAD2")
dss.loadshapes.name = "shapeLOAD3"
dss.text("plot loadshape object=shapeLOAD3")

dss.loadshapes.name = "pv1"
dss.text("plot loadshape object=pv1")

dss.loadshapes.name = "pv2"
dss.text("plot loadshape object=pv2")

dss.loadshapes.name = "pv3"
dss.text("plot loadshape object=pv3")
print("✅ All plots generated.")
