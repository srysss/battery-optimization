#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import py_dss_interface
import pyswarms as ps
import math
import datetime
import getpass
from tqdm import tqdm  # For progress bar
import rainflow  # For battery cycle life analysis
import calendar

#####################################################
# ========== Auxiliary Functions: SOC->DoD and Cycle Detection ==========
#####################################################
def soc_to_dod(soc_series):
    """Convert SOC series to DoD series, where DoD = 100 - SOC."""
    return [100.0 - s for s in soc_series]

def detect_micro_cycles_with_extract(soc_series, T=25.0):
    """
    Use rainflow.extract_cycles() to extract cycles from the battery DoD series
    and compute the degradation (DCL) for each cycle.
    Returns the total cycle life degradation percentage and detailed cycle info.
    """
    dod_series = soc_to_dod(soc_series)
    extracted_cycles = rainflow.extract_cycles(dod_series)
    total_dcl = 0.0
    cycle_list = []
    for (rng, mean, count, i_start, i_end) in extracted_cycles:
        if rng < 0.1:
            cycle_dcl = 0.0
            N_cycle = 1e12
        else:
            D = rng / 100.0
            denom = 7.1568e-6 * math.exp(0.02717 * (273.15 + T)) * (D ** 0.4904)
            N_cycle = (20.0 / denom) ** 2 if denom > 0 else 1e12
            cycle_dcl = count / N_cycle
        total_dcl += cycle_dcl
        cycle_list.append({
            'range_dod': rng,
            'mean': mean,
            'count': count,
            'i_start': i_start,
            'i_end': i_end,
            'Nmax': N_cycle,
            'dcl': cycle_dcl
        })
    return total_dcl * 100.0, cycle_list

#####################################################
# ========== 1. Define 10 Low-Voltage PV+Load+Storage Configurations ==========
#####################################################
script_path = os.path.dirname(os.path.abspath(__file__))

bus_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18]
pv_bess_config_lv = []
for i, bus in enumerate(bus_numbers):
    conf = {
        "pv_name": f"pv{i+1}",
        "bess_name": f"Battery{i+1}",
        "load_name": f"LOAD{i+1}",
        "bus_name": f"{bus}.1",
        "kV": 0.23,
        "pv_kVA": 6,   # 此处pv_kVA代表额定功率（单位：kW）
        "load_kW": 1,
        "pf": 0.95,
        "bess_kWRated": 5,
        "bess_kWhRated": 13.5,
    }
    pv_bess_config_lv.append(conf)

#####################################################
# ========== 1.1 Read Monthly PV & Load Data ==========
#####################################################
chosen_month = 1
month_name = datetime.date(1900, chosen_month, 1).strftime("%B")  # e.g., "January"
pv_month_dir = os.path.join(script_path, "pv_month", month_name)
load_month_dir = os.path.join(script_path, "load_month", month_name)
time_step_hours = 0.5

for config in pv_bess_config_lv:
    pv_txt = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}.txt")
    if not os.path.exists(pv_txt):
        raise FileNotFoundError(f"Cannot find monthly PV file: {pv_txt}")
    # 读取原始数据，列名设为 pv_raw，原始数据以3kW基准
    df_pv_raw = pd.read_csv(pv_txt, header=None, names=["pv_raw"])
    # 根据原始数据以3kW为基准，将数据转换为实际kW值用于Python计算
    df_pv_raw["pv_kw"] = df_pv_raw["pv_raw"] * (config['pv_kVA'] / 3.0)
    # 保存完整的PV数据（包含原始数据和kW数据）供后续Python计算使用
    config["pv_data"] = df_pv_raw.copy()

    load_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    if not os.path.exists(load_txt):
        raise FileNotFoundError(f"Cannot find monthly LOAD file: {load_txt}")
    df_load = pd.read_csv(load_txt, header=None, names=["load"])
    config["load_data"] = df_load

T = len(pv_bess_config_lv[0]["pv_data"])
print(f"Chosen month = {month_name}, total time steps (T) = {T}")

#####################################################
# ========== 1.2 Read Electricity Pricing Data ==========
#####################################################
agile_purchase_file = os.path.join(script_path, "Agile_pricing_data", "Agile_pricing_2021.csv")
agile_sale_file = os.path.join(script_path, "Agile_Outgoing_pricing_data", "Agile_Outgoing_pricing_2021.csv")
df_purchase = pd.read_csv(agile_purchase_file, header=None, names=["time", "price"], parse_dates=["time"])
df_sale = pd.read_csv(agile_sale_file, header=None, names=["time", "price"], parse_dates=["time"])
df_purchase_month = df_purchase[df_purchase["time"].dt.month == chosen_month]
df_sale_month = df_sale[df_sale["time"].dt.month == chosen_month]
if len(df_purchase_month) < T or len(df_sale_month) < T:
    raise ValueError("Not enough pricing data for the chosen month.")
purchase_prices = df_purchase_month["price"].to_numpy()[:T]
sale_prices = df_sale_month["price"].to_numpy()[:T]

#####################################################
# ========== Save Converted PV Data File for OpenDSS ==========
#####################################################
# 仅转换用于OpenDSS的加载曲线文件，将kW值转换为pu值，而不影响Python后续的kW计算。
# pu值计算公式： pu = 实际kW / 额定kW = (pv_raw * (pv_kVA/3.0)) / pv_kVA = pv_raw/3.0
for config in pv_bess_config_lv:
    # 生成包含pu值的转换文件，文件名中增加 "_pu" 标识
    conv_file = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}_pu.txt")
    df_pv_pu = pd.DataFrame()
    df_pv_pu["pv"] = config["pv_data"]["pv_raw"] / 3.0
    df_pv_pu.to_csv(conv_file, index=False, header=False)
    config["converted_pv_file"] = conv_file

#####################################################
# ========== 2. Setup OpenDSS & Create Components (Daily-like Simulation) ==========
#####################################################
dss = py_dss_interface.DSS()
dss_file = os.path.join(script_path, "../../feeders/Feeder_3/Master33.dss")
dss.text(f"compile [{dss_file}]")
dss.text("calcv")

for config in pv_bess_config_lv:
    pv_ls_name = config["pv_name"] + "_ls"
    pv_txt = config["converted_pv_file"]  # 使用pu值加载曲线文件
    dss.text(f"New Loadshape.{pv_ls_name} npts={T} interval={time_step_hours} mult=(file={pv_txt}) useactual=yes")
    # PVSystem使用额定功率（kW），实际输出由 Pmpp * pu值 得到
    dss.text(f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kVA={config['pv_kVA']} Pmpp={config['pv_kVA']} irradiance=1 %cutin=0.1 %cutout=0.1 daily={pv_ls_name}")
    load_ls_name = config["load_name"] + "_ls"
    load_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    dss.text(f"New Loadshape.{load_ls_name} npts={T} interval={time_step_hours} mult=(file={load_txt}) useactual=yes")
    dss.text(f"New Load.{config['load_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kW={config['load_kW']} PF={config['pf']} daily={load_ls_name}")
    dss.text(f"New Storage.{config['bess_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
             f"kWhStored={config['bess_kWhRated']*0.2} %EffCharge=95 %EffDischarge=95 dispmode=DEFAULT")
    dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")

print("✅ DSS components have been created.")

#####################################################
# ========== Create Output Folders with Month Subdirectories ==========
#####################################################
results_dir = os.path.join(script_path, "Results")
bat_dir = os.path.join(results_dir, "Battery_Scenario")
figures_dir = os.path.join(bat_dir, "Figures", month_name)
csv_dir = os.path.join(bat_dir, "CSV", month_name)
monitor_dir = os.path.join(bat_dir, "Monitor_Exports", month_name)
for folder in [results_dir, bat_dir, figures_dir, csv_dir, monitor_dir]:
    os.makedirs(folder, exist_ok=True)

#####################################################
# ========== 3. Time-Series Simulation with Battery Control (Daily-like Simulation) ==========
#####################################################
dss.text("set mode=daily")
dss.text("set number=1")
dss.text(f"set stepsize={time_step_hours}h")

total_revenue = 0.0
total_purchase = 0.0
electricity_cost_over_time = []

n = len(pv_bess_config_lv)
household_cost_only_pv = np.zeros(n)
household_cost_with_batt = np.zeros(n)
household_load_energy = np.zeros(n)
household_pv_only_supply = np.zeros(n)
household_bess_supply = np.zeros(n)
grid_power_without = np.zeros((n, T))
grid_power_with = np.zeros((n, T))

battery_soc = {conf['bess_name']: 20.0 for conf in pv_bess_config_lv}
soc_history = {conf['bess_name']: [] for conf in pv_bess_config_lv}
battery_schedule = {conf['bess_name']: [] for conf in pv_bess_config_lv}

node_voltages_over_time = {}

print("\n[INFO] Starting time-series simulation with battery control...")
for step in tqdm(range(T), desc="Time Steps"):
    for i, config in enumerate(pv_bess_config_lv):
        bess_name = config['bess_name']
        soc_now = battery_soc[bess_name]
        # Python计算时使用实际kW值（字段pv_kw）
        pv_val = config["pv_data"].iloc[step]["pv_kw"]
        ld_val = config["load_data"].iloc[step]["load"]
        net_t = pv_val - ld_val

        grid_power_without[i, step] = net_t * time_step_hours

        if net_t > 0:
            surplus_energy = net_t * time_step_hours
            revenue = surplus_energy * sale_prices[step] / 100.0
            household_cost_only_pv[i] -= revenue
        else:
            shortage_energy = abs(net_t) * time_step_hours
            cost = shortage_energy * purchase_prices[step] / 100.0
            household_cost_only_pv[i] += cost

        load_energy = ld_val * time_step_hours
        pv_energy = pv_val * time_step_hours
        household_load_energy[i] += load_energy
        household_pv_only_supply[i] += min(pv_energy, load_energy)

        max_chg_power = config['bess_kWRated']
        max_dis_power = config['bess_kWRated']
        capacity = config['bess_kWhRated']
        used_energy_chg = 0.0
        used_energy_dis = 0.0
        used_energy_grid = 0.0

        battery_command = 0.0

        if net_t > 0:
            available_surplus = net_t * time_step_hours
            if soc_now < 80.0:
                needed_to_80 = (80.0 - soc_now) / 100.0 * capacity
                required_energy_from_surplus = needed_to_80 / 0.95
                if available_surplus >= required_energy_from_surplus:
                    battery_soc[bess_name] = 80.0
                    used_energy_grid = required_energy_from_surplus
                    used_energy_chg = needed_to_80
                else:
                    used_energy_grid = available_surplus
                    available_stored_energy = used_energy_grid * 0.95
                    soc_incr = (available_stored_energy / capacity) * 100
                    new_soc = min(soc_now + soc_incr, 80.0)
                    battery_soc[bess_name] = new_soc
                    used_energy_chg = available_stored_energy
            battery_command = - (used_energy_grid / time_step_hours)
            surplus_after_chg = max(available_surplus - used_energy_grid, 0.0)
            revenue = surplus_after_chg * sale_prices[step] / 100.0
            total_revenue += revenue
            household_cost_with_batt[i] -= revenue

        elif net_t < 0:
            needed_energy = abs(net_t) * time_step_hours
            if soc_now > 20.0:
                max_effective_discharge = (soc_now - 20.0) / 100.0 * capacity * 0.95
                if max_effective_discharge >= needed_energy:
                    battery_energy_to_discharge = needed_energy / 0.95
                    battery_soc[bess_name] = soc_now - (battery_energy_to_discharge / capacity) * 100
                    used_energy_dis = needed_energy
                else:
                    battery_soc[bess_name] = 20.0
                    used_energy_dis = max_effective_discharge
                shortage_after_dis = needed_energy - used_energy_dis
            else:
                shortage_after_dis = needed_energy
            battery_command = (used_energy_dis / time_step_hours)
            cost_b = shortage_after_dis * purchase_prices[step] / 100.0
            total_purchase += cost_b
            household_cost_with_batt[i] += cost_b
        else:
            battery_command = 0.0

        battery_schedule[bess_name].append(battery_command)

        grid_power_with[i, step] = net_t * time_step_hours - (used_energy_grid if net_t > 0 else 0) + (used_energy_dis if net_t < 0 else 0)
        local_supply = min(load_energy, pv_energy + (used_energy_dis if net_t < 0 else 0))
        household_bess_supply[i] += local_supply
        soc_history[bess_name].append(battery_soc[bess_name])
    dss.text("Solve")
    curr_net_cost = total_purchase - total_revenue
    electricity_cost_over_time.append(curr_net_cost)
    for nd, v in zip(dss.circuit.nodes_names, dss.circuit.buses_vmag_pu):
        node_voltages_over_time.setdefault(nd, []).append(v)

print("[INFO] Time-series simulation finished.")

#####################################################
# ========== Compute Community Metrics ==========
#####################################################
community_cost_with = household_cost_with_batt.sum()
community_cost_pv = household_cost_only_pv.sum()
community_SSR = (household_bess_supply.sum() / household_load_energy.sum() * 100
                 if household_load_energy.sum() > 1e-9 else 0.0)
diff_pv = np.ptp(grid_power_without, axis=1)
diff_bess = np.ptp(grid_power_with, axis=1)
household_RPVD = np.where(diff_pv != 0, (diff_pv - diff_bess) / diff_pv * 100, 0.0)
community_grid_without = np.sum(grid_power_without, axis=0)
community_grid_with = np.sum(grid_power_with, axis=0)
community_diff_pv = np.ptp(community_grid_without)
community_diff_bess = np.ptp(community_grid_with)
community_RPVD = ((community_diff_pv - community_diff_bess) / community_diff_pv * 100
                  if community_diff_pv != 0 else 0.0)
household_savings = household_cost_only_pv - household_cost_with_batt
community_savings = community_cost_pv - community_cost_with

#####################################################
# ========== Battery Micro-cycle Detection ==========
#####################################################
battery_degradation = {}
for config in pv_bess_config_lv:
    bname = config["bess_name"]
    user_soc_list = np.array(soc_history[bname])
    total_dcl_pct, _ = detect_micro_cycles_with_extract(user_soc_list, T=25.0)
    battery_degradation[bname] = total_dcl_pct

#####################################################
# ========== Accumulate Monthly Data ==========
#####################################################
final_monthly_battery_schedules = {bname: np.array(battery_schedule[bname]) for bname in battery_schedule}
monthly_household_cost_with_battery = {conf["bess_name"]: household_cost_with_batt[i] for i, conf in enumerate(pv_bess_config_lv)}
monthly_household_cost_only_pv = {conf["bess_name"]: household_cost_only_pv[i] for i, conf in enumerate(pv_bess_config_lv)}
monthly_cost_curve_over_time = electricity_cost_over_time.copy()
monthly_soc_history = {bname: np.array(soc_history[bname]) for bname in soc_history}

#####################################################
# ========== Generate Final Monthly Data ==========
#####################################################
monthly_battery_degradation = {}
for config in pv_bess_config_lv:
    bname = config["bess_name"]
    monthly_soc_series = monthly_soc_history[bname]
    monthly_dcl, _ = detect_micro_cycles_with_extract(monthly_soc_series, T=25.0)
    monthly_battery_degradation[bname] = monthly_dcl

community_degradation = np.mean(list(monthly_battery_degradation.values()))

# ***** 生成 battery 的 loadshape 文件，转换为 pu 值（以小时为单位），固定小数点格式 *****
for conf in pv_bess_config_lv:
    bname = conf["bess_name"]
    base_power = conf["bess_kWRated"]
    pu_schedule = np.array(final_monthly_battery_schedules[bname]) / base_power
    loadshape_filename = os.path.join(csv_dir, f"{bname}_monthly_schedule_{month_name}.txt")
    np.savetxt(loadshape_filename, pu_schedule, delimiter=",", fmt="%.6f")

#####################################################
# ========== Monthly Simulation using Loadshape Files ==========
#####################################################
T_month = T
dss = py_dss_interface.DSS()
dss_file = os.path.join(script_path, "../../feeders/Feeder_3/Master33.dss")
dss.text(f"compile [{dss_file}]")
dss.text("calcv")

# 重新创建PV、Load、Storage和Monitor设备，确保PV部分使用转换后的数据文件
for config in pv_bess_config_lv:
    # PV：使用转换后的文件（mult参数中的值为pu）
    pv_ls_name = config["pv_name"] + "_ls"
    pv_txt = config["converted_pv_file"]
    dss.text(f"New Loadshape.{pv_ls_name} npts={T_month} interval={time_step_hours} mult=(file={pv_txt}) useactual=yes")
    dss.text(f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kVA={config['pv_kVA']} Pmpp={config['pv_kVA']} irradiance=1 %cutin=0.1 %cutout=0.1 daily={pv_ls_name}")
    # Load
    load_ls_name = config["load_name"] + "_ls"
    load_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    dss.text(f"New Loadshape.{load_ls_name} npts={T_month} interval={time_step_hours} mult=(file={load_txt}) useactual=yes")
    dss.text(f"New Load.{config['load_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kW={config['load_kW']} PF={config['pf']} daily={load_ls_name}")
    # Storage：使用导出的 loadshape 文件进行调度
    battery_ls_name = config["bess_name"] + "_monthly_ls"
    monthly_schedule_file = os.path.join(csv_dir, f"{config['bess_name']}_monthly_schedule_{month_name}.txt")
    dss.text(f"New Loadshape.{battery_ls_name} npts={T_month} interval={time_step_hours} mult=(file={monthly_schedule_file}) useactual=yes")
    dss.text(f"New Storage.{config['bess_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
             f"kWhStored={config['bess_kWhRated']*0.2} %EffCharge=95 %EffDischarge=95 dispmode=FOLLOW daily={battery_ls_name}")
    # Monitor
    dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")

dss.text("set mode=daily")
dss.text("set number=1")
dss.text(f"set stepsize={time_step_hours}h")

node_names = dss.circuit.nodes_names
node_voltages_monthly = {nd: [] for nd in node_names}
monthly_p_high = []
for t in range(T_month):
    dss.text("solve")
    dss.circuit.set_active_element("Transformer.TR1")
    trans_power = dss.cktelement.powers
    p_high = trans_power[0] + trans_power[2] + trans_power[4]
    monthly_p_high.append(p_high)
    dss.circuit.set_active_element("")
    vmag_pu = dss.circuit.buses_vmag_pu
    for nd, v in zip(node_names, vmag_pu):
        node_voltages_monthly[nd].append(v)
monthly_load_energy_day = 0.0
for config in pv_bess_config_lv:
    load_array = config["load_data"]["load"].to_numpy()[:T]
    monthly_load_energy_day += load_array.sum() * time_step_hours
total_load_energy_month = monthly_load_energy_day
total_grid_purchase_energy = sum(p * time_step_hours for p in monthly_p_high if p > 0)
if total_load_energy_month > 0:
    community_SSR_central = (1 - total_grid_purchase_energy / total_load_energy_month) * 100
else:
    community_SSR_central = 0

dss.text("Set DataPath=" + monitor_dir)
for config in pv_bess_config_lv:
    dss.text(f"Export monitor {config['pv_name']}_mon")
    dss.text(f"Export monitor {config['load_name']}_mon")
    dss.text(f"Export monitor {config['bess_name']}_mon")

monthly_voltage_csv = os.path.join(csv_dir, f"Node_Voltagesy_battery_{month_name}.csv")
df_monthly_voltages = pd.DataFrame(node_voltages_monthly)
df_monthly_voltages.to_csv(monthly_voltage_csv, index=False)

all_voltages = np.concatenate(list(node_voltages_monthly.values()))
comm_avg_voltage = all_voltages.mean()
comm_min_voltage = all_voltages.min()
comm_max_voltage = all_voltages.max()
comm_p2p_voltage = comm_max_voltage - comm_min_voltage
comm_exceed_count = int(np.sum((all_voltages < 0.98) | (all_voltages > 1.02)))
comm_rmse = np.sqrt(np.mean((all_voltages - 1.0)**2))

voltage_metrics = {}
target_buses = [conf["bus_name"] for conf in pv_bess_config_lv]
for bus in target_buses:
    if bus in node_voltages_monthly:
        voltages = np.array(node_voltages_monthly[bus])
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

monthly_battery_degradation = {}
for config in pv_bess_config_lv:
    bname = config["bess_name"]
    monthly_soc_series = np.array(soc_history[bname])
    monthly_dcl, _ = detect_micro_cycles_with_extract(monthly_soc_series, T=25.0)
    monthly_battery_degradation[bname] = monthly_dcl

community_degradation = np.mean(list(monthly_battery_degradation.values()))

monthly_csv_filename = os.path.join(csv_dir, f"Battery_summary_{month_name}.csv")
rows = []
for config in pv_bess_config_lv:
    bname = config["bess_name"]
    user_label = config["bus_name"].split(".")[0].lower()
    cost_with = monthly_household_cost_with_battery[bname]
    cost_only = monthly_household_cost_only_pv[bname]
    saving = cost_only - cost_with
    ssr_val = "N/A"
    cycle_deg = monthly_battery_degradation[bname]
    vm = voltage_metrics.get(config["bus_name"], {
        "Avg Voltage (p.u.)": np.nan,
        "Min Voltage (p.u.)": np.nan,
        "Max Voltage (p.u.)": np.nan,
        "P2P Voltage (p.u.)": np.nan,
        "Exceed Count": np.nan,
        "RMSE (p.u.)": np.nan
    })
    rows.append([user_label, cost_with, saving, ssr_val, cycle_deg,
                 vm["Avg Voltage (p.u.)"], vm["Min Voltage (p.u.)"], vm["Max Voltage (p.u.)"],
                 vm["P2P Voltage (p.u.)"], vm["Exceed Count"], vm["RMSE (p.u.)"]])
rows.append(["community",
             sum(monthly_household_cost_with_battery.values()),
             sum(monthly_household_cost_only_pv.values()) - sum(monthly_household_cost_with_battery.values()),
             community_SSR_central,
             community_degradation,
             comm_avg_voltage,
             comm_min_voltage,
             comm_max_voltage,
             comm_p2p_voltage,
             comm_exceed_count,
             comm_rmse])
df_monthly_summary = pd.DataFrame(rows, columns=["User",
                                                 "Monthly Cost (Battery) (￡)",
                                                 "Cost Savings (￡)",
                                                 "SSR (%)",
                                                 "Cycle-Life Degradation (%)",
                                                 "Avg Voltage (p.u.)",
                                                 "Min Voltage (p.u.)",
                                                 "Max Voltage (p.u.)",
                                                 "P2P Voltage (p.u.)",
                                                 "Exceed Count",
                                                 "RMSE (p.u.)"])
df_monthly_summary.to_csv(monthly_csv_filename, index=False)

time_hours_month = np.arange(0, T_month * time_step_hours, time_step_hours)
plt.figure(figsize=(10, 5))
plt.plot(time_hours_month, monthly_cost_curve_over_time, marker='o', linestyle='-')
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative Net Cost (£)")
plt.title(f"Monthly Electricity Cost Over Time - {month_name}")
plt.grid(True)
monthly_cost_curve_png = os.path.join(figures_dir, f"Battery_cost_curve_{month_name}.png")
plt.savefig(monthly_cost_curve_png, dpi=300, bbox_inches='tight')
plt.close()



print("✅ Completed! (Monthly CSV summary with voltage metrics generated)")