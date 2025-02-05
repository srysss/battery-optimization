import os
import pandas as pd
import py_dss_interface
import matplotlib.pyplot as plt

# ========== 1. Initialization and previous setup ==========

# 初始化 DSS 接口
dss = py_dss_interface.DSS()

# 编译 IEEE 13 Node Test Feeder
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")
dss.text(f"compile [{dss_file}]")

# ========== 1.1 从不同的中压母线通过变压器派生出 10 个低压用户母线 ==========
# 选择具有节点 "1" 的中压母线（参考 IEEE 13 节点系统中的母线信息）
medium_buses = ["650", "rg60", "633", "671", "692", "675", "652", "670", "632", "684"]

# 设置电压基准：高压 115 kV，中压 4.16 kV，低压 0.277 kV
dss.text("Set Voltagebases=[115, 4.16, 0.277]")

# 为每个用户创建一个变压器，将中压侧接在相应母线的 1 号节点上，低压侧接到新的用户母线 UserX.1
for i in range(1, 11):
    med_bus = medium_buses[i - 1]
    # 注意：这里将所有参数写在一行，不使用 "~" 续行符，避免转换错误
    transformer_cmd = (
        f"New Transformer.User{i} phases=1 windings=2 XHL=0.01 "
        f"wdg=1 bus={med_bus}.1 kV=4.16 conn=wye "
        f"wdg=2 bus=User{i}.1 kV=0.277 conn=wye"
    )
    dss.text(transformer_cmd)

dss.text("calcv")
print("✅ 已通过变压器从不同的中压母线派生出 10 个低压用户母线: User1.1 ~ User10.1")

# ========== 2. 定义 10 组“低压” PV+Load+Storage 配置, 挂在 User1.1 ~ User10.1 上 ==========
pv_bess_config_lv = []
for i in range(1, 11):
    conf = {
        "pv_name": f"pv{i}",
        "bess_name": f"Battery{i}",
        "load_name": f"LOAD{i}",
        "bus_name": f"User{i}.1",  # 对应上面新建的低压母线
        "kV": 0.277,             # 低压母线电压
        "pv_kVA": 10,
        "load_kW": 1,
        "pf": 0.95,
        "bess_kWRated": 10,
        "bess_kWhRated": 13.5
    }
    pv_bess_config_lv.append(conf)

# ========== 2.1 读取 PV/Load shape 文件并存入 config (示例) ==========
for config in pv_bess_config_lv:
    pv_filename = os.path.join(script_path, "pv_data", "PV_OpenDSS_kW.txt")
    load_filename = os.path.join(script_path, "load_data", f"{config['load_name']}_OpenDSS_kW.txt")

    # 读取 PV 数据文件
    if os.path.exists(pv_filename):
        df_pv = pd.read_csv(pv_filename, header=None, names=["pv"])
    else:
        raise FileNotFoundError(f"Cannot find PV file: {pv_filename}")

    # 读取 Load 数据文件；如果不存在，则使用默认的全 1.0 曲线
    if os.path.exists(load_filename):
        df_load = pd.read_csv(load_filename, header=None, names=["load"])
    else:
        df_load = pd.DataFrame({"load": [1.0] * 48})

    config["pv_data"] = df_pv
    config["load_data"] = df_load

# ========== 2.2 定义一个函数: 在 OpenDSS 中创建 PV/Load/Storage + 监控 ==========
def add_components_to_dss(pv_bess_list):
    """
    遍历每个 PV/Load/Storage 配置：
      - 创建 Loadshape
      - 定义 PVSystem
      - 定义 Load (附加 Loadshape)
      - 定义 Storage
      - 定义 Monitor
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
        dss.text(
            f"New Loadshape.{load_name} npts=48 minterval=30 "
            f"mult=(file={load_file_txt}) useactual=yes"
        )

        # 4) 创建 Load
        dss.text(
            f"New Load.{load_name} phases=1 bus1={config['bus_name']} "
            f"kV={config['kV']} kW={config['load_kW']} PF={config['pf']} "
            f"daily={load_name}"
        )

        # 5) 创建 Storage
        dss.text(
            f"New Storage.{bess_name} phases=1 bus1={config['bus_name']} kV={config['kV']} "
            f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
            f"kWhStored={config['bess_kWhRated'] * 0.2} %EffCharge=95 %EffDischarge=95 "
            f"dispmode=DEFAULT"
        )

        # 6) 创建 Monitors
        dss.text(f"New Monitor.{pv_name}_mon Element=PVSystem.{pv_name} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{load_name}_mon Element=Load.{load_name} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{bess_name}_mon Element=Storage.{bess_name} Terminal=1 Mode=7")

# ========== 2.3 在 OpenDSS 中创建这 10 组分布式资源 ==========
add_components_to_dss(pv_bess_config_lv)
print("✅ 已在低压用户母线 (User1.1 ~ User10.1) 上创建 10 组 PV+Load+Storage.")

# ========== 3. Subsequent simulation and battery control logic ==========
# 定义交易数据文件路径
agile_purchase_file = os.path.join(script_path, "Agile_pricing_data", "Agile_pricing_data_1.csv")
agile_sale_file = os.path.join(script_path, "Agile_Outgoing_pricing_data", "Agile_Outgoing_pricing_data_1.csv")

# 读取 CSV 文件（无表头）
df_purchase = pd.read_csv(agile_purchase_file, header=None, names=["time", "price"])
df_sale = pd.read_csv(agile_sale_file, header=None, names=["time", "price"])
purchase_prices = df_purchase["price"].tolist()
sale_prices = df_sale["price"].tolist()

total_revenue = 0.0
total_purchase_cost = 0.0
electricity_cost_over_time = []
battery_soc = {conf['bess_name']: 20.0 for conf in pv_bess_config_lv}  # 初始 SOC 设为 20%
time_step_hours = 0.5  # 每步 30 分钟

dss.text("set mode=daily")
dss.text("set number=1")
dss.text("set stepsize=0.5h")

total_number = 48
time_steps = [i * 0.5 for i in range(total_number)]
node_voltages_over_time = {}

# ========== 仿真循环 ==========
for step_idx in range(total_number):
    print(f"\nTime step {step_idx + 1}/{total_number}:")
    for config in pv_bess_config_lv:
        bess_name = config['bess_name']
        pv_name = config['pv_name']
        load_name = config['load_name']

        # 获取当前时刻的 PV 和 Load 数据
        pv_power_kW = config["pv_data"].iloc[step_idx]["pv"]
        load_kW = config["load_data"].iloc[step_idx]["load"]
        net_power = pv_power_kW - load_kW  # 正值：多余功率；负值：短缺

        max_charge_power = config['bess_kWRated']
        max_discharge_power = config['bess_kWRated']
        battery_capacity_kWh = config['bess_kWhRated']
        current_soc = battery_soc[bess_name]
        energy_charged = 0.0
        energy_discharged = 0.0

        print(f"  - {bess_name} at {config['bus_name']}: PV={pv_power_kW:.1f}kW, Load={load_kW:.1f}kW, Net={net_power:.1f}kW, SOC={current_soc:.1f}%")

        if net_power > 0:
            # 多余功率情况：尝试充电，目标充至 80%
            if current_soc < 80.0:
                energy_needed_to_80 = ((80.0 - current_soc) / 100.0) * battery_capacity_kWh
                actual_charge_power = min(net_power, max_charge_power)
                max_possible_energy = actual_charge_power * time_step_hours * 0.95
                if max_possible_energy >= energy_needed_to_80:
                    charge_power_to_80 = energy_needed_to_80 / time_step_hours
                    charge_power_to_80 = min(charge_power_to_80, actual_charge_power)
                    battery_soc[bess_name] = 80.0
                    energy_charged = energy_needed_to_80
                    dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={charge_power_to_80} %stored=80.0")
                    print(f"    -> Charge partial to 80%: {charge_power_to_80:.1f} kW, new SOC=80.0%")
                else:
                    soc_incr = (max_possible_energy / battery_capacity_kWh) * 100
                    new_soc = min(current_soc + soc_incr, 100.0)
                    battery_soc[bess_name] = new_soc
                    energy_charged = max_possible_energy
                    dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={actual_charge_power} %stored={new_soc}")
                    print(f"    -> Charge full 30min: {actual_charge_power:.1f} kW, new SOC={new_soc:.3f}%")
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                print("    -> Battery SOC >=80%, IDLING.")
            surplus_energy = max(net_power * time_step_hours - energy_charged, 0.0)
            revenue = (surplus_energy * sale_prices[step_idx]) / 100
            total_revenue += revenue
            if surplus_energy > 0:
                print(f"    -> Surplus energy: {surplus_energy:.2f} kWh sold at price {sale_prices[step_idx]:.4f}, revenue += {revenue:.4f}")
        elif net_power < 0:
            # 功率不足情况：尝试放电，目标保持 SOC 至少 20%
            if current_soc > 20.0:
                discharge_power = min(abs(net_power), max_discharge_power)
                max_possible_discharge = (((current_soc - 20.0) / 100.0) * battery_capacity_kWh) / time_step_hours
                discharge_power = min(discharge_power, max_possible_discharge)
                energy_discharged = discharge_power * time_step_hours * 0.95
                soc_decr = (energy_discharged / battery_capacity_kWh) * 100
                new_soc = max(current_soc - soc_decr, 20.0)
                battery_soc[bess_name] = new_soc
                dss.text(f"Edit Storage.{bess_name} State=DISCHARGING kW={-discharge_power} %stored={new_soc}")
                print(f"    -> Discharge {discharge_power:.1f} kW, new SOC={new_soc:.3f}%")
                shortage_energy = max(abs(net_power) * time_step_hours - energy_discharged, 0.0)
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                print("    -> Battery SOC <=20%, IDLING.")
                shortage_energy = abs(net_power) * time_step_hours
            cost = (shortage_energy * purchase_prices[step_idx]) / 100
            total_purchase_cost += cost
            if shortage_energy > 0:
                print(f"    -> Shortage energy: {shortage_energy:.2f} kWh bought at price {purchase_prices[step_idx]:.4f}, cost += {cost:.4f}")
        else:
            dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
            print("    -> Net=0, battery IDLING.")

    # 求解当前时段
    dss.text("Solve")
    current_electricity_cost = total_purchase_cost - total_revenue
    electricity_cost_over_time.append(current_electricity_cost)
    print(f"Time step {step_idx + 1}: Electricity cost = £{current_electricity_cost:.2f}")

    # 记录每个母线的电压（按节点记录）
    node_names_list = dss.circuit.nodes_names
    vmag_pu_list = dss.circuit.buses_vmag_pu
    for node_name, vmag in zip(node_names_list, vmag_pu_list):
        node_voltages_over_time.setdefault(node_name, []).append(vmag)

# ========== 输出交易汇总（单位：£）=========
print("\n================ Price transaction summary ================")
print(f"Total revenue from surplus energy: £{total_revenue:.2f}")
print(f"Total purchase cost for shortage energy: £{total_purchase_cost:.2f}")
print(f"Net profit: £{total_revenue - total_purchase_cost:.2f}")
print("=============================================\n")

# ========== 3.1 Export all Monitors ==========
dss.text("Export monitors all")
print("✅ Simulation complete and monitors exported.")
buscoords_path = os.path.join(script_path, "../feeders/13bus/IEEE13Node_BusXY.csv")
dss.text(f"Buscoords [{buscoords_path}]")
dss.text("set markpvsystems=yes")
dss.text("set markStorage=yes")
dss.text(
    "plot circuit Power max=2000 y y labels=yes buswidth=2 C1=$FF0000 C2=$00FF00 markerpvs=3 markerstorage=2 markersizepvs=5 markersizestorage=8 offset=1"
)

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
user_bus_dict = {}  # 用于存储 UserX.1 母线的电压数据

# 根据节点名称判断所属相，注意这里对名称进行 strip() 清理空格，并转为小写比较
for node_name, v_list in node_voltages_over_time.items():
    clean_name = node_name.strip()
    if clean_name.endswith(".1"):
        if clean_name.lower().startswith("user"):
            user_bus_dict[clean_name] = v_list
        else:
            phase1_dict[clean_name] = v_list
    elif clean_name.endswith(".2"):
        phase2_dict[clean_name] = v_list
    elif clean_name.endswith(".3"):
        phase3_dict[clean_name] = v_list

# 绘制非 User 母线的 Phase 1 节点电压
plt.figure(figsize=(10, 5))
for node_name, voltages in phase1_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.title("Phase 1 Node Voltages (Excluding User Buses)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 绘制 User 母线电压
plt.figure(figsize=(10, 5))
for node_name, voltages in user_bus_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.title("User Bus Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 绘制 Phase 2 节点电压
plt.figure(figsize=(10, 5))
for node_name, voltages in phase2_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.title("Phase 2 Node Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 绘制 Phase 3 节点电压
plt.figure(figsize=(10, 5))
for node_name, voltages in phase3_dict.items():
    plt.plot(time_steps, voltages, label=node_name)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.title("Phase 3 Node Voltages Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 绘制电力交易成本曲线
plt.figure(figsize=(10, 5))
plt.plot(time_steps, electricity_cost_over_time,
         marker='o', linestyle='-', linewidth=1, markersize=4,
         color='purple', label='Total Electricity Cost')
plt.xlabel("Time (hours)")
plt.ylabel("Cost (£)")
plt.title("Total Electricity Cost (Purchase - Revenue) Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("✅ Done!")
