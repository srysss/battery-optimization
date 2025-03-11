#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import py_dss_interface
import math
import datetime
import getpass
from tqdm import tqdm  # 若未安装，请 pip install tqdm

#########################################
# ========== 1. 定义10个低压PV+负荷配置（无储能） ==========
#########################################
script_path = os.path.dirname(os.path.abspath(__file__))

# 定义PV安装的bus编号列表
bus_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18]

pv_configs = []
for i, bus in enumerate(bus_numbers):
    conf = {
        "pv_name": f"pv{i+1}",
        "load_name": f"LOAD{i+1}",
        "bus_name": f"{bus}.1",   # 对应的bus编号
        "kV": 0.23,
        "pv_kVA": 6,   # PV额定容量
        "load_kW": 1,  # 初始负荷容量（实际时值由Loadshape控制）
        "pf": 0.95,
    }
    pv_configs.append(conf)

#########################################
# ========== 1.1 读取选定月份的 PV & Load 数据 ==========
#########################################
# 选择仿真的月份（1到12）——此处选择2月
chosen_month = 1
month_name = datetime.date(1900, chosen_month, 1).strftime("%B")  # 得到 "February"

# 构造月度数据所在的目录（PV和Load数据已按照月份存放）
pv_month_dir = os.path.join(script_path, "pv_month", month_name)
load_month_dir = os.path.join(script_path, "load_month", month_name)

# 时间步长（单位：小时）
time_step_hours = 0.5

# 读取每个配置对应的月度数据，并存入配置字典中
for config in pv_configs:
    pv_txt = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}.txt")
    if not os.path.exists(pv_txt):
        raise FileNotFoundError(f"Cannot find monthly PV file: {pv_txt}")
    df_pv = pd.read_csv(pv_txt, header=None, names=["pv"])
    config["pv_data"] = df_pv
    df_pv["pv"] = df_pv["pv"] * (config['pv_kVA'] / 3.0)

    load_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    if not os.path.exists(load_txt):
        raise FileNotFoundError(f"Cannot find monthly LOAD file: {load_txt}")
    df_load = pd.read_csv(load_txt, header=None, names=["load"])
    config["load_data"] = df_load

# 假设所有数据文件行数相同，取第一个PV数据的行数作为总时步数
T = len(pv_configs[0]["pv_data"])
print(f"Chosen month = {month_name}, total time steps (T) = {T}")

#########################################
# ========== 2. 读取电价数据 ==========
#########################################
agile_purchase_file = os.path.join(script_path, "Agile_pricing_data", "Agile_pricing_2021.csv")
agile_sale_file = os.path.join(script_path, "Agile_Outgoing_pricing_data", "Agile_Outgoing_pricing_2021.csv")

# 读取电价数据，并解析时间字段
df_purchase = pd.read_csv(agile_purchase_file, header=None, names=["time", "price"], parse_dates=["time"])
df_sale = pd.read_csv(agile_sale_file, header=None, names=["time", "price"], parse_dates=["time"])

# 过滤出所选月份的数据（只比较月份，不考虑年份）
df_purchase_month = df_purchase[df_purchase["time"].dt.month == chosen_month]
df_sale_month = df_sale[df_sale["time"].dt.month == chosen_month]

# 检查是否有足够的数据覆盖所有时步
if len(df_purchase_month) < T or len(df_sale_month) < T:
    raise ValueError("电价数据不足以覆盖所选月份的所有时步")

purchase_prices_month = df_purchase_month["price"].to_numpy()[:T]
sale_prices_month = df_sale_month["price"].to_numpy()[:T]

#########################################
# ========== 3. Setup OpenDSS ==========
#########################################
dss = py_dss_interface.DSS()
dss_file = os.path.join(script_path, "../../feeders/Feeder_3/Master33.dss")
dss.text(f"compile [{dss_file}]")
dss.text("calcv")

# 为每个 PV 和 Load 创建 Loadshape 及对象
for config in pv_configs:
    pv_ls_name = config["pv_name"] + "_ls"
    pv_shape_txt = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}_pu.txt")
    dss.text(
        f"New Loadshape.{pv_ls_name} npts={T} interval={time_step_hours} "
        f"mult=(file={pv_shape_txt}) useactual=yes"
    )
    dss.text(
        f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} "
        f"kV={config['kV']} kVA={config['pv_kVA']} Pmpp={config['pv_kVA']} "
        f"irradiance=1 %cutin=0.1 %cutout=0.1 daily={pv_ls_name}"
    )

    load_ls_name = config["load_name"] + "_ls"
    load_shape_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    dss.text(
        f"New Loadshape.{load_ls_name} npts={T} interval={time_step_hours} "
        f"mult=(file={load_shape_txt}) useactual=yes"
    )
    dss.text(
        f"New Load.{config['load_name']} phases=1 bus1={config['bus_name']} "
        f"kV={config['kV']} kW={config['load_kW']} PF={config['pf']} "
        f"daily={load_ls_name}"
    )

    # 创建监视器（用于后续查看PV和Load运行数据）
    dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")

print("✅ Created monthly-based PV & Load with their Loadshapes in DSS.")

#########################################
# ========== 4. 用文本数据直接计算购售电逻辑 ==========
#########################################
household_cost_pv_only = np.zeros(len(pv_configs))
household_load_energy = np.zeros(len(pv_configs))
household_pv_supply = np.zeros(len(pv_configs))

total_revenue = 0.0
total_purchase = 0.0
electricity_cost_over_time = []

print("\n[INFO] Starting electricity purchase/sale calculation...")
for step in tqdm(range(T), desc="Calculating net costs"):
    for i, config in enumerate(pv_configs):
        pv_val = config["pv_data"].iloc[step]["pv"]
        load_val = config["load_data"].iloc[step]["load"]
        net_val = pv_val - load_val  # 正值表示PV多余上网，负值表示需从电网购买
        if net_val > 0:
            surplus = net_val * time_step_hours
            rev = surplus * sale_prices_month[step] / 100.0
            total_revenue += rev
            household_cost_pv_only[i] -= rev
        else:
            shortage = abs(net_val) * time_step_hours
            cost = shortage * purchase_prices_month[step] / 100.0
            total_purchase += cost
            household_cost_pv_only[i] += cost

        load_energy = load_val * time_step_hours
        pv_energy = pv_val * time_step_hours
        household_load_energy[i] += load_energy
        household_pv_supply[i] += min(pv_energy, load_energy)

    net_cost = total_purchase - total_revenue
    electricity_cost_over_time.append(net_cost)
print("[INFO] Finished electricity purchase/sale calculation.\n")

print("====================== PV-Only Scenario (Monthly, Text Data) ======================")
print(f"Total revenue from surplus: £{total_revenue:.2f}")
print(f"Total purchase cost: £{total_purchase:.2f}")
print(f"Community net cost: £{household_cost_pv_only.sum():.2f}")

#########################################
# ========== 5. 创建输出文件夹（按月份存放） ==========
#########################################
base_results_dir = os.path.join(script_path, "Results", "PV_Only")
figures_dir = os.path.join(base_results_dir, "Figures", month_name)
csv_dir = os.path.join(base_results_dir, "CSV", month_name)
monitor_dir = os.path.join(base_results_dir, "Monitor_Exports", month_name)
for folder in [base_results_dir, figures_dir, csv_dir, monitor_dir]:
    os.makedirs(folder, exist_ok=True)

#########################################
# ========== 6. 绘制购售电成本曲线 ==========
#########################################
plt.figure(figsize=(10, 5))
time_range_hours = [i * time_step_hours for i in range(T)]
plt.plot(time_range_hours, electricity_cost_over_time, marker='o', linestyle='-')
plt.xlabel("Time (hours)")
plt.ylabel("Cost (£)")
plt.title(f"Total Electricity Cost Over Time (PV-Only) - {month_name}")
plt.grid(True)
cost_curve_png = os.path.join(figures_dir, f"PV_only_cost_curve_{month_name}.png")
plt.savefig(cost_curve_png, dpi=300, bbox_inches='tight')
print(f"[DEBUG] Saved figure: {cost_curve_png}")
plt.close()

#########################################
# ========== 7. 记录节点电压及Transformer高压侧数据 ==========
#########################################
print("\n[INFO] Setting daily mode for time-series simulation in OpenDSS...")
dss.text("set mode=daily")
dss.text("set number=1")
dss.text(f"set stepsize={time_step_hours}h")

node_voltages_over_time = {}
monthly_p_high = []  # 用于记录Transformer高压侧的购电功率

print("[INFO] Starting time-series simulation to collect node voltages and transformer data...")
for step in tqdm(range(T), desc="Time-series steps"):
    dss.text("Solve")
    # 设置活动元件为Transformer，采集高压侧功率
    dss.circuit.set_active_element("Transformer.TR1")
    trans_power = dss.cktelement.powers
    # 累加变压器高压侧的各相功率：假设传输功率放在第0、2、4位
    p_high = trans_power[0] + trans_power[2] + trans_power[4]
    monthly_p_high.append(p_high)
    dss.circuit.set_active_element("")
    # 记录节点电压数据
    node_names = dss.circuit.nodes_names
    vmag_pu = dss.circuit.buses_vmag_pu
    for name, v in zip(node_names, vmag_pu):
        node_voltages_over_time.setdefault(name, []).append(v)
print("[INFO] Node voltages and transformer data recorded.\n")

#########################################
# ========== 8. 导出监视器数据 ==========
#########################################
print("[INFO] Exporting monitors individually...")
monitor_export_dir = monitor_dir
os.makedirs(monitor_export_dir, exist_ok=True)
dss.text(f"Set DataPath={monitor_export_dir}")

for config in pv_configs:
    pv_mon_cmd = f"Export monitor {config['pv_name']}_mon"
    load_mon_cmd = f"Export monitor {config['load_name']}_mon"
    print(f"[DEBUG] Exporting {pv_mon_cmd}")
    dss.text(pv_mon_cmd)
    print(f"[DEBUG] Exporting {load_mon_cmd}")
    dss.text(load_mon_cmd)
print(f"[INFO] Monitor data exported to: {monitor_export_dir}")

#########################################
# ========== 9. 生成节点电压CSV文件 ==========
#########################################
df_voltages = pd.DataFrame(node_voltages_over_time)
df_voltages.insert(0, "Time (hours)", [i * time_step_hours for i in range(T)])
node_voltage_csv = os.path.join(csv_dir, f"PV_only_node_voltages_{month_name}.csv")
df_voltages.to_csv(node_voltage_csv, index=False)
print(f"[DEBUG] Saved node voltage CSV: {node_voltage_csv}")

#########################################
# ========== 10. 计算月度Transformer购电及社区SSR ==========
#########################################
monthly_load_energy_day = 0.0
for config in pv_configs:
    load_array = config["load_data"]["load"].to_numpy()[:T]
    monthly_load_energy_day += load_array.sum() * time_step_hours
total_load_energy_month = monthly_load_energy_day
total_grid_purchase_energy = sum(p * time_step_hours for p in monthly_p_high if p > 0)
if total_load_energy_month > 0:
    community_SSR_central = (1 - total_grid_purchase_energy / total_load_energy_month) * 100
else:
    community_SSR_central = 0

#########################################
# ========== 11. 计算电压指标并生成详细CSV汇总 ==========
#########################################
# 针对目标bus（配置中的bus_name）计算电压指标
target_buses = [conf["bus_name"] for conf in pv_configs]
voltage_metrics = {}
for bus in target_buses:
    if bus in node_voltages_over_time:
        voltages = np.array(node_voltages_over_time[bus])
        avg_voltage = voltages.mean()
        min_voltage = voltages.min()
        max_voltage = voltages.max()
        p2p_voltage = max_voltage - min_voltage
        exceed_count = int(np.sum((voltages < 0.99) | (voltages > 1.01)))
        rmse = np.sqrt(np.mean((voltages - 1.0)**2))
        voltage_metrics[bus] = {
            "Avg Voltage (p.u.)": avg_voltage,
            "Min Voltage (p.u.)": min_voltage,
            "Max Voltage (p.u.)": max_voltage,
            "P2P Voltage (p.u.)": p2p_voltage,
            "Exceed Count": exceed_count,
            "RMSE (p.u.)": rmse
        }
    else:
        voltage_metrics[bus] = {
            "Avg Voltage (p.u.)": np.nan,
            "Min Voltage (p.u.)": np.nan,
            "Max Voltage (p.u.)": np.nan,
            "P2P Voltage (p.u.)": np.nan,
            "Exceed Count": np.nan,
            "RMSE (p.u.)": np.nan
        }

all_voltages = np.concatenate(list(node_voltages_over_time.values()))
comm_avg_voltage = all_voltages.mean()
comm_min_voltage = all_voltages.min()
comm_max_voltage = all_voltages.max()
comm_p2p_voltage = comm_max_voltage - comm_min_voltage
comm_exceed_count = int(np.sum((all_voltages < 0.98) | (all_voltages > 1.02)))
comm_rmse = np.sqrt(np.mean((all_voltages - 1.0)**2))

# 生成详细汇总CSV文件（格式与代码1类似，但电池相关字段置为“N/A”或0）
detailed_csv = os.path.join(csv_dir, f"PV_only_summary_detailed_{month_name}.csv")
rows = []
for i, config in enumerate(pv_configs):
    bus = config["bus_name"]
    user_label = bus.split(".")[0].lower()
    cost = household_cost_pv_only[i]
    # 计算每个用户的SSR（仅用PV供给的SSR仍通过文本数据计算）
    ssr = (household_pv_supply[i] / household_load_energy[i] * 100) if household_load_energy[i] > 0 else 0
    cycle_deg = "N/A"  # 无储能，不存在循环寿命衰减
    vm = voltage_metrics.get(bus, {
        "Avg Voltage (p.u.)": np.nan,
        "Min Voltage (p.u.)": np.nan,
        "Max Voltage (p.u.)": np.nan,
        "P2P Voltage (p.u.)": np.nan,
        "Exceed Count": np.nan,
        "RMSE (p.u.)": np.nan
    })
    rows.append([user_label, cost, "N/A", ssr, cycle_deg,
                 vm["Avg Voltage (p.u.)"], vm["Min Voltage (p.u.)"], vm["Max Voltage (p.u.)"],
                 vm["P2P Voltage (p.u.)"], vm["Exceed Count"], vm["RMSE (p.u.)"]])
# community row：采用月度仿真计算的SSR
community_cost = household_cost_pv_only.sum()
rows.append(["community", community_cost, 0.0, community_SSR_central,
             "N/A", comm_avg_voltage, comm_min_voltage, comm_max_voltage,
             comm_p2p_voltage, comm_exceed_count, comm_rmse])
df_summary = pd.DataFrame(rows, columns=["User",
                                           "Monthly Cost (PV-Only) (￡)",
                                           "Cost Savings (￡)",
                                           "SSR (%)",
                                           "Cycle-Life Degradation (%)",
                                           "Avg Voltage (p.u.)",
                                           "Min Voltage (p.u.)",
                                           "Max Voltage (p.u.)",
                                           "P2P Voltage (p.u.)",
                                           "Exceed Count",
                                           "RMSE (p.u.)"])
df_summary.to_csv(detailed_csv, index=False)
print(f"Detailed CSV summary generated: {detailed_csv}")

print("✅ Done! (PV-Only Scenario - Full code with detailed summary, transformer-based SSR, and voltage metrics)")