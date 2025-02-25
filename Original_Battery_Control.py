#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import py_dss_interface
import math
import datetime
import getpass
from tqdm import tqdm  # For progress bar
import rainflow  # For battery cycle life analysis

# Set font to Arial to help eliminate missing glyph warnings.
plt.rcParams["font.family"] = "Arial"

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

# Use the same bus numbers as in the original feeder (Feeder_3)
bus_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18]

pv_bess_config_lv = []
for i, bus in enumerate(bus_numbers):
    conf = {
        "pv_name": f"pv{i+1}",
        "bess_name": f"Battery{i+1}",
        "load_name": f"LOAD{i+1}",
        "bus_name": f"{bus}.1",   # Corresponding bus number
        "kV": 0.23,
        "pv_kVA": 3,
        "load_kW": 1,
        "pf": 0.95,
        "bess_kWRated": 3,
        "bess_kWhRated": 3
    }
    pv_bess_config_lv.append(conf)

#####################################################
# ========== 1.1 Read Monthly PV & Load Data ==========
#####################################################
# Choose the simulation month (1-12), here February is chosen
chosen_month = 7
month_name = datetime.date(1900, chosen_month, 1).strftime("%B")  # e.g., "February"

# Set the directories for monthly data
pv_month_dir = os.path.join(script_path, "pv_month", month_name)
load_month_dir = os.path.join(script_path, "load_month", month_name)

# Time step in hours
time_step_hours = 0.5

# Read each configuration's monthly data and store in the config dictionary
for config in pv_bess_config_lv:
    pv_txt = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}.txt")
    if not os.path.exists(pv_txt):
        raise FileNotFoundError(f"Cannot find monthly PV file: {pv_txt}")
    df_pv = pd.read_csv(pv_txt, header=None, names=["pv"])
    config["pv_data"] = df_pv

    load_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    if not os.path.exists(load_txt):
        raise FileNotFoundError(f"Cannot find monthly LOAD file: {load_txt}")
    df_load = pd.read_csv(load_txt, header=None, names=["load"])
    config["load_data"] = df_load

# Assume all files have the same number of rows
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
# ========== 2. Setup OpenDSS & Create Components ==========
#####################################################
dss = py_dss_interface.DSS()
dss_file = os.path.join(script_path, "../../feeders/Feeder_3/Master33.dss")
dss.text(f"compile [{dss_file}]")
dss.text("calcv")

for config in pv_bess_config_lv:
    # Create PV loadshape and PV system
    pv_ls_name = config["pv_name"] + "_ls"
    pv_txt = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}.txt")
    dss.text(f"New Loadshape.{pv_ls_name} npts={T} interval={time_step_hours} mult=(file={pv_txt}) useactual=yes")
    dss.text(f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kVA={config['pv_kVA']} Pmpp={config['pv_kVA']} irradiance=1 %cutin=0.1 %cutout=0.1 daily={pv_ls_name}")

    # Create load loadshape and then create the load object
    load_ls_name = config["load_name"] + "_ls"
    load_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    dss.text(f"New Loadshape.{load_ls_name} npts={T} interval={time_step_hours} mult=(file={load_txt}) useactual=yes")
    dss.text(f"New Load.{config['load_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kW={config['load_kW']} PF={config['pf']} daily={load_ls_name}")

    # Create Storage (Battery) with initial stored energy at 20% of capacity
    dss.text(f"New Storage.{config['bess_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
             f"kWhStored={config['bess_kWhRated']*0.2} %EffCharge=95 %EffDischarge=95 dispmode=DEFAULT")

    # Create monitors for PV, Load, and Battery
    dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")

print("✅ DSS components have been created.")

#####################################################
# ========== Create Output Folders ==========
#####################################################
# 所有输出存放在 Results 文件夹下，本代码输出存放在 Results/Battery_Scenario
results_dir = os.path.join(script_path, "Results")
bat_dir = os.path.join(results_dir, "Battery_Scenario")
figures_dir = os.path.join(bat_dir, "Figures")
csv_dir = os.path.join(bat_dir, "CSV")
monitor_dir = os.path.join(bat_dir, "Monitor_Exports")
for folder in [results_dir, bat_dir, figures_dir, csv_dir, monitor_dir]:
    os.makedirs(folder, exist_ok=True)

#####################################################
# ========== 3. Time-Series Simulation with Battery Control ==========
#####################################################
dss.text("set mode=daily")
dss.text("set number=1")
dss.text(f"set stepsize={time_step_hours}h")

total_revenue = 0.0
total_purchase = 0.0
electricity_cost_over_time = []

# Record the cost for "PV-only" and "With Battery" scenarios for each user
household_cost_only_pv = np.zeros(len(pv_bess_config_lv))
household_cost_with_batt = np.zeros(len(pv_bess_config_lv))

# For SSR calculation: record load energy and local supply (PV-only and with battery)
household_load_energy = np.zeros(len(pv_bess_config_lv))
household_pv_only_supply = np.zeros(len(pv_bess_config_lv))
household_bess_supply = np.zeros(len(pv_bess_config_lv))

# Initialize battery SOC for each unit (%), initial value = 20%
battery_soc = {conf['bess_name']: 20.0 for conf in pv_bess_config_lv}
soc_history = {conf['bess_name']: [] for conf in pv_bess_config_lv}

# For RPVD: record grid power (in kWh) for both scenarios per user at each timestep
n = len(pv_bess_config_lv)
grid_power_without = np.zeros((n, T))
grid_power_with = np.zeros((n, T))

node_voltages_over_time = {}

print("\n[INFO] Starting time-series simulation with battery control...")
for step in tqdm(range(T), desc="Time Steps"):
    for i, config in enumerate(pv_bess_config_lv):
        bess_name = config['bess_name']
        soc_now = battery_soc[bess_name]
        pv_val = config["pv_data"].iloc[step]["pv"]
        ld_val = config["load_data"].iloc[step]["load"]
        net_t = pv_val - ld_val  # Positive: surplus, Negative: shortage

        # Record grid power for PV-only scenario (in kWh)
        grid_power_without[i, step] = net_t * time_step_hours

        # PV-only cost calculation
        if net_t > 0:
            surplus = net_t * time_step_hours
            revenue = surplus * sale_prices[step] / 100.0
            household_cost_only_pv[i] -= revenue
        else:
            shortage = abs(net_t) * time_step_hours
            cost = shortage * purchase_prices[step] / 100.0
            household_cost_only_pv[i] += cost

        # Record load and PV energy (kWh)
        load_energy = ld_val * time_step_hours
        pv_energy = pv_val * time_step_hours
        household_load_energy[i] += load_energy
        household_pv_only_supply[i] += min(pv_energy, load_energy)

        # Battery control logic:
        # Charging (net_t > 0) uses negative kW values (as per sign convention)
        # Discharging (net_t < 0) uses positive kW values.
        max_chg_power = config['bess_kWRated']
        max_dis_power = config['bess_kWRated']
        capacity = config['bess_kWhRated']

        used_energy_chg = 0.0
        used_energy_dis = 0.0

        if net_t > 0:
            if soc_now < 80.0:
                needed_to_80 = (80.0 - soc_now) / 100.0 * capacity
                possible_chg = net_t * time_step_hours * 0.95  # charging efficiency
                if possible_chg >= needed_to_80:
                    battery_soc[bess_name] = 80.0
                    used_energy_chg = needed_to_80
                    chg_power = used_energy_chg / time_step_hours
                    chg_power = min(chg_power, max_chg_power)
                    # Use negative kW for charging
                    dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={-chg_power} %stored=80")
                else:
                    soc_incr = (possible_chg / capacity) * 100
                    new_soc = min(soc_now + soc_incr, 100.0)
                    battery_soc[bess_name] = new_soc
                    used_energy_chg = possible_chg
                    chg_power = min(net_t, max_chg_power)
                    dss.text(f"Edit Storage.{bess_name} State=CHARGING kW={-chg_power} %stored={new_soc}")
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
            surplus_after_chg = max(net_t * time_step_hours - used_energy_chg, 0.0)
            revenue = surplus_after_chg * sale_prices[step] / 100.0
            total_revenue += revenue
            household_cost_with_batt[i] -= revenue

        elif net_t < 0:
            # Discharge branch (use positive kW values)
            if soc_now > 20.0:
                available_energy = (soc_now - 20.0) / 100.0 * capacity * 0.95
                needed_energy = abs(net_t) * time_step_hours
                if available_energy >= needed_energy:
                    battery_soc[bess_name] = soc_now - (needed_energy / capacity) * 100
                    used_energy_dis = needed_energy
                    discharge_power = min(needed_energy / time_step_hours, max_dis_power)
                    dss.text(f"Edit Storage.{bess_name} State=DISCHARGING kW={discharge_power} %stored={battery_soc[bess_name]}")
                else:
                    battery_soc[bess_name] = 20.0
                    used_energy_dis = available_energy
                    discharge_power = min(available_energy / time_step_hours, max_dis_power)
                    dss.text(f"Edit Storage.{bess_name} State=DISCHARGING kW={discharge_power} %stored=20")
                shortage_after_dis = needed_energy - used_energy_dis
            else:
                dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")
                shortage_after_dis = abs(net_t) * time_step_hours
            cost_b = shortage_after_dis * purchase_prices[step] / 100.0
            total_purchase += cost_b
            household_cost_with_batt[i] += cost_b
        else:
            dss.text(f"Edit Storage.{bess_name} State=IDLING kW=0")

        # Record grid power for With-Battery scenario (in kWh)
        grid_power_with[i, step] = net_t * time_step_hours - used_energy_chg + used_energy_dis

        # Record local supply (PV plus any discharged energy)
        local_supply = min(load_energy, pv_energy + used_energy_dis)
        household_bess_supply[i] += local_supply

        # Record current battery SOC
        soc_history[bess_name].append(battery_soc[bess_name])
    dss.text("Solve")
    curr_net_cost = total_purchase - total_revenue
    electricity_cost_over_time.append(curr_net_cost)

    # Record node voltages
    node_names = dss.circuit.nodes_names
    vmag_pu = dss.circuit.buses_vmag_pu
    for nd, v in zip(node_names, vmag_pu):
        node_voltages_over_time.setdefault(nd, []).append(v)

print("[INFO] Time-series simulation finished.")

#####################################################
# ========== 4. Compute Community Metrics and RPVD ==========
#####################################################
community_cost_with = household_cost_with_batt.sum()
community_cost_pv = household_cost_only_pv.sum()
community_SSR = (household_bess_supply.sum() / household_load_energy.sum() * 100
                 if household_load_energy.sum() > 1e-9 else 0.0)

# Compute RPVD for each user:
# RPVD = ((peak-to-peak of grid_power_without - peak-to-peak of grid_power_with) / (peak-to-peak of grid_power_without))*100
diff_pv = np.ptp(grid_power_without, axis=1)
diff_bess = np.ptp(grid_power_with, axis=1)
household_RPVD = np.where(diff_pv != 0, (diff_pv - diff_bess) / diff_pv * 100, 0.0)

# Compute community RPVD: sum grid power of all users and then compute peak-to-peak differences
community_grid_without = np.sum(grid_power_without, axis=0)
community_grid_with = np.sum(grid_power_with, axis=0)
community_diff_pv = np.ptp(community_grid_without)
community_diff_bess = np.ptp(community_grid_with)
community_RPVD = (community_diff_pv - community_diff_bess) / community_diff_pv * 100 if community_diff_pv != 0 else 0.0

# Compute cost savings (PV-only cost minus with-battery cost)
household_savings = household_cost_only_pv - household_cost_with_batt
community_savings = community_cost_pv - community_cost_with

print("\n=================== Battery Scenario Metrics ===================")
print(f"Total Revenue from Sales: £{total_revenue:.2f}")
print(f"Total Purchase Cost: £{total_purchase:.2f}")
print(f"Community Net Cost (With Battery): £{community_cost_with:.2f}")
print(f"Community PV-Only Cost: £{community_cost_pv:.2f}")
print(f"Community Cost Savings: £{community_savings:.2f}")
print(f"Community SSR: {community_SSR:.2f}%")
print(f"Community RPVD: {community_RPVD:.2f}%\n")

for i, config in enumerate(pv_bess_config_lv):
    user_label = config["bus_name"].split(".")[0].lower()
    cost_pv_only = household_cost_only_pv[i]
    cost_with_batt = household_cost_with_batt[i]
    rpvd_val = household_RPVD[i]
    saving = household_savings[i]
    print(f"  - {user_label}: PV-Only Cost = {cost_pv_only:.4f}, With Battery Cost = {cost_with_batt:.4f}, "
          f"Cost Savings = £{saving:.4f}, RPVD = {rpvd_val:.2f}%")

#####################################################
# ========== 5. Battery Micro-cycle Detection ==========
#####################################################
# 同时记录每个用户的循环寿命降解
battery_degradation = {}
print("\n------ Battery Micro-cycle Detection ------")
for config in pv_bess_config_lv:
    bname = config["bess_name"]
    user_label = config["bus_name"].split(".")[0].lower()
    user_soc_list = soc_history[bname]
    total_dcl_pct, cycle_details = detect_micro_cycles_with_extract(user_soc_list, T=25.0)
    battery_degradation[user_label] = total_dcl_pct
    print(f"User '{user_label}': Total Cycle-Life Degradation = {total_dcl_pct:.4f}%")
    # Uncomment below to print detailed cycle information
    # for cyc in cycle_details:
    #     print(cyc)

#####################################################
# ========== 6. Generate CSV Summary File ==========
#####################################################
csv_filename = os.path.join(csv_dir, f"Battery_summary_{month_name}.csv")
rows = []
for i, config in enumerate(pv_bess_config_lv):
    user_label = config["bus_name"].split(".")[0].lower()
    cost_with = household_cost_with_batt[i]
    rpvd = household_RPVD[i]
    ssr = (household_bess_supply[i] / household_load_energy[i] * 100) if household_load_energy[i] > 0 else 0.0
    saving = household_savings[i]
    cycle_deg = battery_degradation.get(user_label, None)
    rows.append([user_label, cost_with, saving, rpvd, ssr, cycle_deg])
# Append community summary row (对于社区整体，循环寿命降解以"N/A"表示)
rows.append(["community", community_cost_with, community_savings, community_RPVD, community_SSR, "N/A"])
df_summary = pd.DataFrame(rows, columns=["User", "Monthly Cost (Battery) (￡)",
                                           "Cost Savings (￡)", "RPVD (%)", "SSR (%)", "Cycle-Life Degradation (%)"])
df_summary.to_csv(csv_filename, index=False)
print(f"\n================ CSV file '{csv_filename}' generated =================")

#####################################################
# ========== 7. Plot Cost and Node Voltage Curves ==========
#####################################################
time_hours = [i * time_step_hours for i in range(T)]

# 1) Plot the cost curve
plt.figure(figsize=(10, 5))
plt.plot(time_hours, electricity_cost_over_time, marker='o', linestyle='-')
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative Net Cost (£)")
plt.title(f" Electricity Cost Over Time (Battery Scenario) - {month_name}")
plt.grid(True)
cost_curve_png = os.path.join(figures_dir, f"Battery_cost_curve_{month_name}.png")
plt.savefig(cost_curve_png, dpi=300, bbox_inches='tight')
print(f"[DEBUG] Cost curve saved: {cost_curve_png}")
plt.show()
plt.close()

# 2) Plot node voltage profiles
plt.figure(figsize=(12, 8))
for node, voltages in node_voltages_over_time.items():
    plt.plot(time_hours, voltages, label=node)
plt.xlabel("Time (hours)")
plt.ylabel("Voltage (p.u.)")
plt.title(f"Node Voltage Profile Over Time - {month_name}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.68)
voltage_profile_png = os.path.join(figures_dir, f"Battery_voltage_profile_{month_name}.png")
plt.savefig(voltage_profile_png, dpi=300, bbox_inches='tight')
print(f"[DEBUG] Voltage profile saved: {voltage_profile_png}")
plt.show()
plt.close()

# Export Node Voltage data to CSV
voltage_csv_filename = os.path.join(csv_dir, f"Node_voltage_{month_name}.csv")
df_voltage = pd.DataFrame(node_voltages_over_time, index=time_hours)
df_voltage.index.name = "Time (hours)"
df_voltage.to_csv(voltage_csv_filename)
print(f"[DEBUG] Node Voltage CSV saved: {voltage_csv_filename}")

#####################################################
# ========== 8. Export Monitor Data ==========
#####################################################
dss.text(f"Set DataPath={monitor_dir}")
for config in pv_bess_config_lv:
    dss.text(f"Export monitor {config['pv_name']}_mon")
    dss.text(f"Export monitor {config['load_name']}_mon")
    dss.text(f"Export monitor {config['bess_name']}_mon")
print(f"[INFO] Monitor data exported to: {monitor_dir}")

print("✅ Completed! (Battery Scenario with RPVD, Cost Savings, revised discharge logic, Node Voltage CSV, and battery cycle degradation in summary)")
