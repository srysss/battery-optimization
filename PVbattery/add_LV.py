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

# ========== 1.1. 在 634 母线派生出 10 条低压线路 (单相, 0.01 miles) ==========
# 注意：这里用到 linecode=mtx605 (IEEE13里定义的单相线参数)，你也可换成 mtx607 或自定义
for i in range(1, 11):
    line_cmd = (
        f"New Line.LVLine_{i} phases=1 "
        f"bus1=634.1 bus2=User{i}.1 "
        f"linecode=mtx605 length=0.01 units=mi"
    )
    dss.text(line_cmd)
dss.text(f"Set Voltagebases=[115, 4.16, .48]")
dss.text(f"calcv")

print("✅ 已从 634.1 派生出 10 条低压线路: LVLine_1 ~ LVLine_10.")

# ========== 2. 定义 10 组“低压” PV+Load+Storage 配置, 挂在 User1.1 ~ User10.1 上 ==========

pv_bess_config_lv = []
for i in range(1, 11):
    conf = {
        "pv_name": f"pv{i}",
        "bess_name": f"Battery{i}",
        "load_name": f"LOAD{i}",
        "bus_name": f"User{i}.1",  # 关键：对应刚才新建的母线
        "kV": 0.277,  # 低压母线电压
        "pv_kVA": 10,
        "load_kW": 1,
        "pf": 0.95,
        "bess_kWRated": 10,
        "bess_kWhRated": 13.5
    }
    pv_bess_config_lv.append(conf)

# ========== 2.1 读取 PV/Load shape 文件并存入 config (示例) ==========

# 假设每个 PV/Load 使用相同的时序文件 (如 pv_data/PV_OpenDSS_kW.txt)
# 也可根据 i, 生成不同文件名
for config in pv_bess_config_lv:
    pv_filename = os.path.join(script_path, "pv_data", "PV_OpenDSS_kW.txt")
    load_filename = os.path.join(script_path, "load_data", f"{config['load_name']}_OpenDSS_kW.txt")

    # 若没有对应的 LOADi_OpenDSS_kW.txt，可先用同一个示例文件替代
    # load_filename = os.path.join(script_path, "load_data", "GenericLoad_OpenDSS_kW.txt")

    # 读文件
    if os.path.exists(pv_filename):
        df_pv = pd.read_csv(pv_filename, header=None, names=["pv"])
    else:
        raise FileNotFoundError(f"Cannot find PV file: {pv_filename}")

    if os.path.exists(load_filename):
        df_load = pd.read_csv(load_filename, header=None, names=["load"])
    else:
        # 如无专门文件，先假设一个简单的全1.0倍(或任意值)曲线
        df_load = pd.DataFrame({"load": [1.0] * 48})

    config["pv_data"] = df_pv
    config["load_data"] = df_load


# ========== 2.2 定义一个函数: 在 OpenDSS 中创建 PV/Load/Storage + 监控 ==========

def add_components_to_dss(pv_bess_list):
    """
    遍历每个PV/Load/Storage配置:
     - 创建Loadshape
     - 定义PVSystem
     - 定义Load (附加Loadshape)
     - 定义Storage
     - 定义Monitor
    """
    for config in pv_bess_list:
        pv_name = config["pv_name"]
        load_name = config["load_name"]
        bess_name = config["bess_name"]

        # 1) PV Loadshape
        pv_file_txt = os.path.join(script_path, "pv_data", "PV_OpenDSS_kW.txt")
        dss.text(
            f"New Loadshape.{pv_name} npts=48 minterval=30 "
            f"mult=(file={pv_file_txt}) useactual=yes"
        )

        # 2) PVSystem
        dss.text(
            f"New PVSystem.{pv_name} phases=1 bus1={config['bus_name']} "
            f"kV={config['kV']} kVA={config['pv_kVA']} "
            f"Pmpp={config['pv_kVA']} "
            f"irradiance=1 %cutin=0.1 %cutout=0.1 "
            f"daily={pv_name}"
        )

        # 3) Loadshape for Load
        load_file_txt = os.path.join(script_path, "load_data", f"{load_name}_OpenDSS_kW.txt")
        # 如无真实文件，可以自己做:
        dss.text(
            f"New Loadshape.{load_name} npts=48 minterval=30 "
            f"mult=(file={load_file_txt}) useactual=yes"
        )

        # 4) 创建Load
        dss.text(
            f"New Load.{load_name} phases=1 bus1={config['bus_name']} "
            f"kV={config['kV']} kW={config['load_kW']} PF={config['pf']} "
            f"daily={load_name}"
        )

        # 5) 创建Storage
        dss.text(
            f"New Storage.{bess_name} phases=1 bus1={config['bus_name']} kV={config['kV']} "
            f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
            f"kWhStored={config['bess_kWhRated'] * 0.5} %EffCharge=95 %EffDischarge=95 "
            f"dispmode=DEFAULT"
        )

        # 6) Monitors
        dss.text(f"New Monitor.{pv_name}_mon Element=PVSystem.{pv_name} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{load_name}_mon Element=Load.{load_name} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{bess_name}_mon Element=Storage.{bess_name} Terminal=1 Mode=7")


# ========== 2.3 在OpenDSS中创建这10组分布式资源 ==========

add_components_to_dss(pv_bess_config_lv)
print("✅ 已在 634 低压支路(UserX.1)上创建 10 组 PV+Load+Storage.")

# ========== 3. Subsequent simulation and battery control logic ==========

# 如果你要套用原本的电池调度逻辑，可以直接对 pv_bess_config_lv 做循环
# 下方示例基本照搬你的原逻辑：每日48步 (30分钟间隔), net_power>0则充电, net_power<0则放电

battery_soc = {conf['bess_name']: 50.0 for conf in pv_bess_config_lv}  # 初始化SOC=50%
time_step_hours = 0.5  # 30分钟间隔

dss.text("set mode=daily")
dss.text("set number=1")
dss.text("set stepsize=0.5h")

total_number = 48
time_steps = [i * 0.5 for i in range(total_number)]  # x轴(单位小时)

node_voltages_over_time = {}

for step_idx in range(total_number):
    print(f"\nTime step {step_idx + 1}/{total_number}:")

    # === 简单电池充放电逻辑 ===
    for config in pv_bess_config_lv:
        bess_name = config['bess_name']
        pv_name = config['pv_name']
        load_name = config['load_name']

        # 从我们先前读到的 DataFrame 里获取对应时刻的PV/Load
        pv_power_kW = config["pv_data"].iloc[step_idx]["pv"]
        load_kW = config["load_data"].iloc[step_idx]["load"]
        net_power = pv_power_kW - load_kW  # 正表示剩余可充电，负表示不足需放电

        max_charge_power = config['bess_kWRated']
        max_discharge_power = config['bess_kWRated']
        battery_capacity_kWh = config['bess_kWhRated']

        current_soc = battery_soc[bess_name]

        print(
            f"  - {bess_name} at {config['bus_name']}: PV={pv_power_kW:.1f}kW, Load={load_kW:.1f}kW, Net={net_power:.1f}kW, SOC={current_soc:.1f}%")

        if net_power > 0:
            # -- 充电，目标80%
            if current_soc < 80.0:
                energy_needed_to_80 = ((80.0 - current_soc) / 100.0) * battery_capacity_kWh
                actual_charge_power = min(net_power, max_charge_power)
                max_possible_energy = actual_charge_power * time_step_hours
                if max_possible_energy >= energy_needed_to_80:
                    charge_power_to_80 = energy_needed_to_80 / time_step_hours
                    charge_power_to_80 = min(charge_power_to_80, actual_charge_power)
                    battery_soc[bess_name] = 80.0
                    dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={charge_power_to_80} %stored={80.0}")
                    print(f"    -> Charge partial to 80%: {charge_power_to_80:.1f} kW, new SOC=80.0%")
                else:
                    # 整个时段都能充
                    soc_incr = (max_possible_energy / battery_capacity_kWh) * 100
                    new_soc = min(current_soc + soc_incr, 100.0)
                    battery_soc[bess_name] = new_soc
                    dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={actual_charge_power} %stored={new_soc}")
                    print(f"    -> Charge full 30min: {actual_charge_power:.1f} kW, new SOC={new_soc:.1f}%")
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                print("    -> Battery SOC >=80%, IDLING.")

        elif net_power < 0:
            # -- 放电，目标SOC>=20%
            if current_soc > 20.0:
                discharge_power = min(abs(net_power), max_discharge_power)
                max_possible_discharge = (current_soc - 20.0) / 100.0 * battery_capacity_kWh / time_step_hours
                discharge_power = min(discharge_power, max_possible_discharge)

                soc_decr = (discharge_power * time_step_hours) / battery_capacity_kWh * 100
                new_soc = max(current_soc - soc_decr, 20.0)
                battery_soc[bess_name] = new_soc
                # 在OpenDSS里放电用负kW（注意有时也用正kW+设置DispMode=SOURCE）
                dss.text(f"Edit Storage.{bess_name} State=DISCHARGING kW={-discharge_power} %stored={new_soc}")
                print(f"    -> Discharge {discharge_power:.1f} kW, new SOC={new_soc:.1f}%")
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                print("    -> Battery SOC <=20%, IDLING.")
        else:
            # net_power=0
            dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
            print("    -> Net=0, battery IDLING.")

    # Solve current time step
    dss.text("Solve")

    # 记录每个母线电压
    node_names_list = dss.circuit.nodes_names
    vmag_pu_list = dss.circuit.buses_vmag_pu
    for node_name, vmag in zip(node_names_list, vmag_pu_list):
        node_voltages_over_time.setdefault(node_name, []).append(vmag)

# ========== 3.1 Export all Monitors ==========

dss.text("Export monitors all")
print("✅ Simulation complete and monitors exported.")
buscoords_path = os.path.join(script_path, "../feeders/13bus/IEEE13Node_BusXY.csv")
dss.text(f"Buscoords [{buscoords_path}]")
dss.text("set markpvsystems=yes")
dss.text("set markStorage=yes")
dss.text("plot circuit Power max=2000 y y labels=yes buswidth=2 C1=$FF0000 C2=$00FF00 markerpvs=3 markerstorage=2 markersizepvs=5 markersizestorage=8 offset=1")
# ========== 3.2 读取并绘制 Monitor 结果 ==========

monitor_dir = os.path.dirname(dss_file)
for config in pv_bess_config_lv:
    pv_name = config["pv_name"]
    load_name = config["load_name"]
    bess_name = config["bess_name"]

    # PV Monitor
    pv_monitor_filename = f"IEEE13Nodeckt_Mon_{pv_name}_mon_1.csv"
    pv_monitor_path = os.path.join(monitor_dir, pv_monitor_filename)
    if os.path.exists(pv_monitor_path):
        df_pv = pd.read_csv(pv_monitor_path)
        print(f"\nReading PV monitor file: {pv_monitor_path}")
        if ' P1 (kW)' in df_pv.columns:
            plt.figure(figsize=(8, 4))
            plt.plot(time_steps, -df_pv[' P1 (kW)'], label=f"{pv_name} - P1(kW)")
            plt.xlabel("Time (hours)")
            plt.ylabel("Power (kW)")
            plt.title(f"PV Monitor: {pv_name}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Load Monitor
    load_monitor_filename = f"IEEE13Nodeckt_Mon_{load_name}_mon_1.csv"
    load_monitor_path = os.path.join(monitor_dir, load_monitor_filename)
    if os.path.exists(load_monitor_path):
        df_load = pd.read_csv(load_monitor_path)
        print(f"\nReading Load monitor file: {load_monitor_path}")
        if ' P1 (kW)' in df_load.columns:
            plt.figure(figsize=(8, 4))
            plt.plot(time_steps, df_load[' P1 (kW)'], color='orange', label=f"{load_name} - P1(kW)")
            plt.xlabel("Time (hours)")
            plt.ylabel("Power (kW)")
            plt.title(f"Load Monitor: {load_name}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Battery Monitor
    storage_monitor_filename = f"IEEE13Nodeckt_Mon_{bess_name}_mon_1.csv"
    storage_monitor_path = os.path.join(monitor_dir, storage_monitor_filename)
    if os.path.exists(storage_monitor_path):
        df_batt = pd.read_csv(storage_monitor_path)
        print(f"\nReading Storage monitor file: {storage_monitor_path}")
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

# ========== 4. 绘制节点电压 (p.u.) ==========

phase1_dict = {}
phase2_dict = {}
phase3_dict = {}

for node_name, v_list in node_voltages_over_time.items():
    if node_name.endswith(".1"):
        phase1_dict[node_name] = v_list
    elif node_name.endswith(".2"):
        phase2_dict[node_name] = v_list
    elif node_name.endswith(".3"):
        phase3_dict[node_name] = v_list
    else:
        pass  # 可能是多相标识，可按需解析

plt.figure(figsize=(10, 5))
for node_name, voltages in phase1_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.title("Phase 1 Node Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
for node_name, voltages in phase2_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.title("Phase 2 Node Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
for node_name, voltages in phase3_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.title("Phase 3 Node Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("✅ Done!")
