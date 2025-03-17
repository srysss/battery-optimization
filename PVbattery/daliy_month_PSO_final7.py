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
import rainflow
import calendar

###################################################
# ========== 0. Create Output Folders =========
###################################################
script_path = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_path, "Results")
bat_dir = os.path.join(results_dir, "Battery_Scenario_PSO_Monthly")  # 按月存放结果
figures_dir = os.path.join(bat_dir, "Figures")
csv_dir = os.path.join(bat_dir, "CSV")
monitor_dir = os.path.join(bat_dir, "Monitor_Exports")
for folder in [results_dir, bat_dir, figures_dir, csv_dir, monitor_dir]:
    os.makedirs(folder, exist_ok=True)

# 初始调度文件存放位置（修改为月级别）
base_schedule_dir = os.path.join(bat_dir, "Initial_Schedules")
os.makedirs(base_schedule_dir, exist_ok=True)

###################################################
# ========== 1. Define System Config =========
###################################################
bus_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18]
pv_bess_config = []
for i, bus in enumerate(bus_numbers):
    conf = {
        "pv_name": f"pv{i + 1}",
        "bess_name": f"Battery{i + 1}",
        "load_name": f"LOAD{i + 1}",
        "bus_name": f"{bus}.1",  # 与原馈线母线号一致
        "kV": 0.23,
        "pv_kVA": 6,
        "load_kW": 1,
        "pf": 0.95,
        # 以下参数决定 SoC 限制，充放电过程中将使用这两个值
        "bess_lowlimit": 10,
        "bess_highlimit": 90,
        "bess_kWRated": 5,
        "bess_kWhRated": 13.5,
    }
    pv_bess_config.append(conf)

###################################################
# ========== 2. Common Settings and Pricing Data =========
###################################################
time_step_hours = 0.5
T_day = int(24 / time_step_hours)  # 每天48个时间步
T = T_day

selected_year = 2021
selected_month = 7
num_days = calendar.monthrange(selected_year, selected_month)[1]
# 使用英文月份名称，确保所有月度文件名一致，例如 "January"
month_name = datetime.datetime(selected_year, selected_month, 1).strftime("%B")

agile_purchase_file = os.path.join(script_path, "Agile_pricing_data", "Agile_pricing_2021.csv")
agile_sale_file = os.path.join(script_path, "Agile_Outgoing_pricing_data", "Agile_Outgoing_pricing_2021.csv")
df_purchase = pd.read_csv(agile_purchase_file, header=None, names=["time", "price"], parse_dates=["time"])
df_sale = pd.read_csv(agile_sale_file, header=None, names=["time", "price"], parse_dates=["time"])

###################################################
# ========== 3. Utilities =========
###################################################
def soc_to_dod(soc_series):
    return [100.0 - s for s in soc_series]

def detect_micro_cycles_with_extract(soc_series, T_temp=25.0):
    dod_series = soc_to_dod(soc_series)
    extracted = rainflow.extract_cycles(dod_series)
    total_dcl = 0.0
    cycle_list = []
    for (rng, mean, count, i_start, i_end) in extracted:
        if rng < 0.1:
            cycle_dcl = 0.0
            N_cycle = 1e12
        else:
            D = rng / 100.0
            denom = 7.1568e-6 * math.exp(0.02717 * (273.15 + T_temp)) * (D ** 0.4904)
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

def calculate_metrics_vectorized(battery_powers, initial_soc=None):
    global pv_bess_config, purchase_prices, sale_prices, time_step_hours, T
    n = len(pv_bess_config)
    pv_array = np.empty((n, T))
    load_array = np.empty((n, T))
    battery_capacity = np.empty((n,))
    bess_kWRated = np.empty((n,))
    for i, config in enumerate(pv_bess_config):
        pv_array[i, :] = config["pv_data"]["pv"].to_numpy()
        load_array[i, :] = config["load_data"]["load"].to_numpy()
        battery_capacity[i] = config["bess_kWhRated"]
        bess_kWRated[i] = config["bess_kWRated"]
    sale_prices_arr = np.array(sale_prices)
    purchase_prices_arr = np.array(purchase_prices)
    if initial_soc is None:
        battery_soc = np.full(n, 20.0)
    else:
        battery_soc = initial_soc.copy()
    soc_history = np.empty((n, T))
    household_cost_with_battery = np.zeros(n)
    household_cost_only_pv = np.zeros(n)
    grid_power_without = np.zeros((n, T))
    grid_power_with = np.zeros((n, T))
    only_pv_supply = np.zeros(n)
    bess_supply = np.zeros(n)
    total_load_energy = np.zeros(n)
    total_revenue = 0.0
    total_purchase_cost = 0.0
    cumulative_cost_over_time = np.zeros(T)
    running_cost = 0.0
    # 这里保存经过充放电约束调整后的调度
    adjusted_battery_schedule = np.zeros((n, T))
    efficiency_charging = 0.95
    efficiency_discharging = 0.95

    # 提前提取所有电池的上下限（单位：%）
    lower_limits_all = np.array([config["bess_lowlimit"] for config in pv_bess_config])
    upper_limits_all = np.array([config["bess_highlimit"] for config in pv_bess_config])

    for t in range(T):
        pv_t = pv_array[:, t]
        load_t = load_array[:, t]
        net_power = pv_t - load_t  # kW
        battery_power = battery_powers[:, t]
        battery_power = np.clip(battery_power, -bess_kWRated, bess_kWRated)

        # 充电过程：使用配置中的 bess_highlimit 替换默认的80.0
        charge_idx = battery_power < 0
        if np.any(charge_idx):
            allowed_stored = battery_capacity[charge_idx] * ((upper_limits_all[charge_idx] - battery_soc[charge_idx]) / 100.0)
            allowed_grid = allowed_stored / efficiency_charging
            desired_grid = -battery_power[charge_idx] * time_step_hours
            actual_grid = np.minimum(desired_grid, allowed_grid)
            battery_power[charge_idx] = - actual_grid / time_step_hours

        # 放电过程：使用配置中的 bess_lowlimit 替换默认的20.0
        discharge_idx = battery_power > 0
        if np.any(discharge_idx):
            available_energy = battery_capacity[discharge_idx] * ((battery_soc[discharge_idx] - lower_limits_all[discharge_idx]) / 100.0)
            desired_delivered = battery_power[discharge_idx] * time_step_hours
            allowed_delivered = available_energy * efficiency_discharging
            actual_delivered = np.minimum(desired_delivered, allowed_delivered)
            battery_power[discharge_idx] = actual_delivered / time_step_hours

        adjusted_battery_schedule[:, t] = battery_power.copy()

        battery_power = np.round(battery_power, 6)

        # 计算充电和放电能量（单位：kWh），这里不对能量结果进行四舍五入
        energy_drawn_for_charging = np.where(battery_power < 0, -battery_power * time_step_hours, 0.0)
        energy_delivered_from_discharge = np.where(battery_power > 0, battery_power * time_step_hours, 0.0)

        # 计算 SoC 增量（单位：%），再更新 SoC
        delta_soc = ((energy_drawn_for_charging * efficiency_charging) - (
                    energy_delivered_from_discharge / efficiency_discharging)) / battery_capacity * 100
        battery_soc += delta_soc

        grid_energy = net_power * time_step_hours - energy_drawn_for_charging + energy_delivered_from_discharge
        cost_inc = np.zeros(n)
        pos_grid = grid_energy > 0
        if np.any(pos_grid):
            grid_revenue = (grid_energy[pos_grid] * sale_prices_arr[t]) / 100.0
            cost_inc[pos_grid] = -grid_revenue
            total_revenue += np.sum(grid_revenue)
        neg_grid = grid_energy < 0
        if np.any(neg_grid):
            grid_cost = (np.abs(grid_energy[neg_grid]) * purchase_prices_arr[t]) / 100.0
            cost_inc[neg_grid] = grid_cost
            total_purchase_cost += np.sum(grid_cost)
        household_cost_with_battery += cost_inc
        cost_only_inc = np.where(net_power >= 0,
                                 -net_power * time_step_hours * sale_prices_arr[t] / 100.0,
                                 np.abs(net_power) * time_step_hours * purchase_prices_arr[t] / 100.0)
        household_cost_only_pv += cost_only_inc
        grid_power_without[:, t] = net_power * time_step_hours
        grid_power_with[:, t] = grid_energy
        load_energy = load_t * time_step_hours
        pv_energy = pv_t * time_step_hours
        only_pv_supply += np.minimum(pv_energy, load_energy)
        local_supply = np.minimum(load_energy, pv_energy + energy_delivered_from_discharge)
        bess_supply += local_supply

        total_load_energy += load_energy
        soc_history[:, t] = battery_soc
        running_cost += np.sum(cost_inc)
        cumulative_cost_over_time[t] = running_cost

    electricity_cost = total_purchase_cost - total_revenue
    diff_pv = np.ptp(grid_power_without, axis=1)
    diff_bess = np.ptp(grid_power_with, axis=1)
    household_RPVD = np.where(diff_pv != 0, (diff_pv - diff_bess) / diff_pv * 100, 0.0)
    community_grid_without = np.sum(grid_power_without, axis=0)
    community_grid_with = np.sum(grid_power_with, axis=0)
    community_diff_pv = np.ptp(community_grid_without)
    community_diff_bess = np.ptp(community_grid_with)
    community_RPVD = ((community_diff_pv - community_diff_bess) / community_diff_pv * 100
                      if community_diff_pv != 0 else 0.0)
    household_cost_with_battery_dict = {}
    household_cost_only_pv_dict = {}
    household_RPVD_dict = {}
    grid_power_without_dict = {}
    grid_power_with_dict = {}
    soc_history_dict = {}
    for i in range(n):
        bname = pv_bess_config[i]["bess_name"]
        household_cost_with_battery_dict[bname] = household_cost_with_battery[i]
        household_cost_only_pv_dict[bname] = household_cost_only_pv[i]
        household_RPVD_dict[pv_bess_config[i]["bus_name"]] = household_RPVD[i]
        grid_power_without_dict[pv_bess_config[i]["bus_name"]] = grid_power_without[i, :].tolist()
        grid_power_with_dict[pv_bess_config[i]["bus_name"]] = grid_power_with[i, :].tolist()
        soc_history_dict[bname] = soc_history[i, :].tolist()
    SSR_individual = np.where(total_load_energy > 0, bess_supply / total_load_energy * 100, 0.0)
    SSR_community = (np.sum(bess_supply) / np.sum(total_load_energy) * 100) if np.sum(total_load_energy) > 0 else 0.0

    return (electricity_cost,
            household_cost_with_battery_dict,
            household_cost_only_pv_dict,
            household_RPVD_dict,
            community_RPVD,
            total_purchase_cost - total_revenue,
            grid_power_without_dict,
            grid_power_with_dict,
            soc_history_dict,
            cumulative_cost_over_time,
            adjusted_battery_schedule,
            SSR_individual,
            SSR_community,
            total_revenue,
            total_purchase_cost,
            battery_soc)

def load_baseline_schedule(pv_bess_config, base_schedule_dir):
    baseline_schedules = {}
    month_schedule_dir = os.path.join(base_schedule_dir, "Monthly")
    os.makedirs(month_schedule_dir, exist_ok=True)
    for config in pv_bess_config:
        bess_name = config["bess_name"]
        schedule_file = os.path.join(month_schedule_dir, f"{bess_name}_initial_schedule.txt")
        if os.path.exists(schedule_file):
            schedule = np.loadtxt(schedule_file, delimiter=",")
            baseline_schedules[bess_name] = schedule
        else:
            baseline_schedules[bess_name] = np.zeros(T_day)
            np.savetxt(schedule_file, baseline_schedules[bess_name], delimiter=",", fmt="%.6f")
    return baseline_schedules

def get_cost_function_single(household_index, daily_initial_soc):
    def cost_function(particles):
        n_particles = particles.shape[0]
        costs = np.zeros(n_particles)
        for i in range(n_particles):
            battery_schedule = particles[i]
            battery_powers = np.zeros((len(pv_bess_config), T))
            battery_powers[household_index, :] = battery_schedule
            electricity_cost, household_cost_with_battery_dict, *_ = calculate_metrics_vectorized(battery_powers,
                                                                                                  initial_soc=daily_initial_soc)
            bess_name = pv_bess_config[household_index]['bess_name']
            costs[i] = household_cost_with_battery_dict[bess_name]
        return costs
    return cost_function

###################################################
# ========== 4. Monthly Simulation Loop =========
###################################################
monthly_optimized_battery_schedules = {config["bess_name"]: [] for config in pv_bess_config}
monthly_household_cost_with_battery = {config["bess_name"]: 0.0 for config in pv_bess_config}
monthly_household_cost_only_pv = {config["bess_name"]: 0.0 for config in pv_bess_config}
monthly_cost_curve_over_time = []
monthly_soc_history = {config["bess_name"]: [] for config in pv_bess_config}
community_revenue = 0.0
community_purchase_cost = 0.0
n = len(pv_bess_config)
prev_battery_soc = np.full(n, 20.0)

# -------------------- 新增：提前创建 CSV、Figures、Monitor 的 Monthly_月份 文件夹，作为每日文件存放路径 --------------------
monthly_csv_folder = os.path.join(csv_dir, f"Monthly_{month_name}")
os.makedirs(monthly_csv_folder, exist_ok=True)
monthly_figures_folder = os.path.join(figures_dir, f"Monthly_{month_name}")
os.makedirs(monthly_figures_folder, exist_ok=True)
monthly_monitor_folder = os.path.join(monitor_dir, f"Monthly_{month_name}")
os.makedirs(monthly_monitor_folder, exist_ok=True)

for day in range(1, num_days + 1):
    day_folder = "Day" + str(day)
    selected_date = pd.to_datetime(datetime.datetime(selected_year, selected_month, day))
    # 修改：将每日 CSV、Figures、Monitor 文件放在对应的 Monthly 文件夹下
    daily_csv_dir = os.path.join(monthly_csv_folder, day_folder)
    daily_figures_dir = os.path.join(monthly_figures_folder, day_folder)
    daily_monitor_dir = os.path.join(monthly_monitor_folder, day_folder)
    for folder in [daily_csv_dir, daily_figures_dir, daily_monitor_dir]:
        os.makedirs(folder, exist_ok=True)
    pv_day_dir = os.path.join(script_path, "pv_day", day_folder)
    load_day_dir = os.path.join(script_path, "load_day", day_folder)
    for config in pv_bess_config:
        pv_file = os.path.join(pv_day_dir, f"PV_kW_{day_folder}.txt")
        if os.path.exists(pv_file):
            dfp = pd.read_csv(pv_file, header=None, names=["pv"])
            dfp["pv"] = dfp["pv"] * (config['pv_kVA'] / 3.0)
            dfp = dfp.iloc[:T_day]
        else:
            raise FileNotFoundError(f"Cannot find: {pv_file}")
        load_file = os.path.join(load_day_dir, f"{config['load_name']}_kW_{day_folder}.txt")
        if os.path.exists(load_file):
            dfl = pd.read_csv(load_file, header=None, names=["load"])
            dfl = dfl.iloc[:T_day]
        else:
            raise FileNotFoundError(f"Cannot find: {load_file}")
        config["pv_data"] = dfp
        config["load_data"] = dfl
    df_purchase_day = df_purchase[df_purchase["time"].dt.date == selected_date.date()]
    df_sale_day = df_sale[df_sale["time"].dt.date == selected_date.date()]
    if len(df_purchase_day) < T_day or len(df_sale_day) < T_day:
        raise ValueError("Not enough pricing data for the selected day.")
    purchase_prices = df_purchase_day["price"].to_numpy()[:T_day]
    sale_prices = df_sale_day["price"].to_numpy()[:T_day]
    dss = py_dss_interface.DSS()
    dss_file = os.path.join(script_path, "../../feeders/Feeder_3/Master33.dss")
    dss.text(f"compile [{dss_file}]")
    dss.text("calcv")
    for config in pv_bess_config:
        pv_ls_name = config["pv_name"] + "_ls"
        original_pv_file = os.path.join(pv_day_dir, f"PV_kW_{day_folder}.txt")
        dfp_conv = pd.read_csv(original_pv_file, header=None, names=["pv"])
        # 修改部分：将原始以3kW基准的PV数据转换成pu值（pu = 原始数据/3）
        dfp_conv["pv"] = dfp_conv["pv"] / 3.0
        conv_file = os.path.join(pv_day_dir, f"{config['pv_name']}_PV_{day_folder}_conv.txt")
        dfp_conv.to_csv(conv_file, header=False, index=False, float_format="%.6f")
        dss.text(f"New Loadshape.{pv_ls_name} npts={T_day} interval={time_step_hours} mult=(file={conv_file}) useactual=yes")
        dss.text(f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
                 f"kVA={config['pv_kVA']} Pmpp={config['pv_kVA']} irradiance=1 %cutin=0.1 %cutout=0.1 daily={pv_ls_name}")
        load_ls_name = config["load_name"] + "_ls"
        load_txt = os.path.join(load_day_dir, f"{config['load_name']}_kW_{day_folder}.txt")
        dss.text(f"New Loadshape.{load_ls_name} npts={T_day} interval={time_step_hours} mult=(file={load_txt}) useactual=yes")
        dss.text(f"New Load.{config['load_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
                 f"kW={config['load_kW']} PF={config['pf']} daily={load_ls_name}")

        dss.text(f"New Storage.{config['bess_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
                 f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} %IdlingkW=0 "
                 f"kWhStored={config['bess_kWhRated'] * 0.2} %EffCharge=95 %EffDischarge=95 %reserve={config['bess_lowlimit']} dispmode=FOLLOW")

        dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")
        dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")
    daily_initial_soc = prev_battery_soc.copy()
    zero_battery_powers = np.zeros((len(pv_bess_config), T_day))
    (_, _, household_cost_only_pv, _, _, _, _, _, _, _, _, _, _, _, _, final_soc_dummy) = \
        calculate_metrics_vectorized(zero_battery_powers, initial_soc=daily_initial_soc)
    for config in pv_bess_config:
        _ = household_cost_only_pv[config["bess_name"]]
    options = {'c1': 1.5, 'c2': 1.7, 'w': 0.7}
    n_particles = 1500
    iterations = 100
    baseline_schedules = load_baseline_schedule(pv_bess_config, base_schedule_dir)
    optimized_battery_schedules = {}
    for idx, config in enumerate(pv_bess_config):
        bess_name = config["bess_name"]
        print(f"Optimizing day {day} for {bess_name}")  # 修改后的打印语句
        baseline_schedule = baseline_schedules[bess_name]
        T_local = len(baseline_schedule)
        initial_particles = np.array([baseline_schedule + np.random.uniform(-0.01, 0.01, size=T_local)
                                        for _ in range(n_particles)])
        bounds = (-config['bess_kWRated'] * np.ones(T_local), config['bess_kWRated'] * np.ones(T_local))
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=T_local,
                                              options=options, bounds=bounds, init_pos=initial_particles)
        cost, pos = optimizer.optimize(get_cost_function_single(idx, daily_initial_soc), iters=iterations, verbose=True)
        optimized_battery_schedules[bess_name] = pos
    best_battery_powers = np.zeros((len(pv_bess_config), T_day))
    for idx, config in enumerate(pv_bess_config):
        best_battery_powers[idx, :] = optimized_battery_schedules[config['bess_name']]
    (elec_cost_opt,
     household_cost_with_battery_opt,
     household_cost_only_pv_opt,
     household_RPVD_opt,
     community_RPVD_opt,
     net_cost_opt,
     grid_power_without_opt,
     grid_power_with_opt,
     soc_history_opt,
     cost_curve_over_time,
     adjusted_battery_schedule,
     SSR_individual,
     SSR_community,
     total_revenue,
     total_purchase_cost,
     final_battery_soc) = calculate_metrics_vectorized(best_battery_powers, initial_soc=daily_initial_soc)
    community_pv_only_cost = sum(household_cost_only_pv_opt.values())
    community_battery_cost = sum(household_cost_with_battery_opt.values())
    community_cost_savings = community_pv_only_cost - community_battery_cost

    # 保存经过约束调整后的日度调度文件
    for idx, config in enumerate(pv_bess_config):
        bess_name = config["bess_name"]
        battery_adj_schedule = adjusted_battery_schedule[idx, :]
        battery_adj_schedule_pu = battery_adj_schedule / config["bess_kWRated"]
        daily_schedule_file = os.path.join(daily_csv_dir, f"{bess_name}_optimized_schedule.txt")
        np.savetxt(daily_schedule_file, battery_adj_schedule_pu, delimiter=",", fmt="%.6f")
        battery_ls_name = bess_name + "_opt_ls"
        dss.text(f"New Loadshape.{battery_ls_name} npts={T_day} interval={time_step_hours} mult=(file={daily_schedule_file}) useactual=yes")
        dss.text(f"Edit Storage.{bess_name} dispmode=FOLLOW %reserve={config['bess_lowlimit']} %IdlingkW=0 daily={battery_ls_name}")
    dss.text("set mode=daily")
    dss.text("set number=1")
    dss.text(f"set stepsize={time_step_hours}h")
    node_names = dss.circuit.nodes_names
    node_voltages_over_time = {nd: [] for nd in node_names}
    for t in range(T_day):
        dss.text("solve")
        vmag_pu = dss.circuit.buses_vmag_pu
        for nd, v in zip(node_names, vmag_pu):
            node_voltages_over_time[nd].append(v)
    dss.text(f"Set DataPath={daily_monitor_dir}")
    for config in pv_bess_config:
        dss.text(f"Export monitor {config['pv_name']}_mon")
        dss.text(f"Export monitor {config['load_name']}_mon")
        dss.text(f"Export monitor {config['bess_name']}_mon")
    voltages_csv = os.path.join(daily_csv_dir, f"Node_Voltages_{day_folder}.csv")
    df_voltages = pd.DataFrame(node_voltages_over_time)
    df_voltages.to_csv(voltages_csv, index=False)
    csv_filename = os.path.join(daily_csv_dir, f"Battery_summary_PSO_{day_folder}.csv")
    rows = []
    for config in pv_bess_config:
        bname = config["bess_name"]
        user_label = config["bus_name"].split(".")[0].lower()
        cost_with = household_cost_with_battery_opt[bname]
        cost_only = household_cost_only_pv_opt[bname]
        saving = cost_only - cost_with
        rows.append([user_label, cost_with, saving])
    rows.append(["community", community_battery_cost, community_cost_savings])
    df_summary = pd.DataFrame(rows, columns=["User", "Daily Cost (Battery PSO)", "Cost Savings"])
    df_summary.to_csv(csv_filename, index=False)
    t_hours = np.arange(0, T_day * time_step_hours, time_step_hours)
    plt.figure(figsize=(10, 5))
    plt.plot(t_hours, cost_curve_over_time, marker='o', linestyle='-')
    plt.xlabel("Time (hours)")
    plt.ylabel("Cumulative Net Cost (£)")
    plt.title(f"Daily Electricity Cost Over Time - {day_folder}")
    plt.grid(True)
    cost_curve_png = os.path.join(daily_figures_dir, f"Battery_cost_curve_{day_folder}.png")
    plt.savefig(cost_curve_png, dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12, 8))
    for nd, voltages in node_voltages_over_time.items():
        plt.plot(t_hours, voltages, label=nd)
    plt.xlabel("Time (hours)")
    plt.ylabel("Voltage (p.u.)")
    plt.title(f"Node Voltage Profile Over Time - {day_folder}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.68)
    voltage_profile_png = os.path.join(daily_figures_dir, f"Battery_voltage_profile_{day_folder}.png")
    plt.savefig(voltage_profile_png, dpi=300, bbox_inches='tight')
    plt.close()
    # 累计保存月度数据，这里也改为保存经过约束后的调度
    for idx, config in enumerate(pv_bess_config):
        bname = config["bess_name"]
        monthly_optimized_battery_schedules[bname].append(adjusted_battery_schedule[idx, :])
        monthly_household_cost_with_battery[bname] += household_cost_with_battery_opt[bname]
        monthly_household_cost_only_pv[bname] += household_cost_only_pv_opt[bname]
        monthly_soc_history[bname].append(np.array(soc_history_opt[bname]))
    if len(monthly_cost_curve_over_time) == 0:
        monthly_cost_curve_over_time = cost_curve_over_time.copy()
    else:
        offset = monthly_cost_curve_over_time[-1]
        monthly_cost_curve_over_time = np.concatenate([monthly_cost_curve_over_time, cost_curve_over_time + offset])
    community_revenue += total_revenue
    community_purchase_cost += total_purchase_cost
    prev_battery_soc = final_battery_soc.copy()

final_monthly_battery_schedules = {}
monthly_battery_degradation = {}
for config in pv_bess_config:
    bname = config["bess_name"]
    final_monthly_battery_schedules[bname] = np.concatenate(monthly_optimized_battery_schedules[bname])
    monthly_soc_series = np.concatenate(monthly_soc_history[bname])
    monthly_dcl, _ = detect_micro_cycles_with_extract(monthly_soc_series, T_temp=25.0)
    monthly_battery_degradation[bname] = monthly_dcl

community_degradation = np.mean(list(monthly_battery_degradation.values()))

# --------------------
# 新增部分：创建 Monthly_月份 子文件夹，并输出月度 CSV 文件
monthly_csv_folder = os.path.join(csv_dir, f"Monthly_{month_name}")
os.makedirs(monthly_csv_folder, exist_ok=True)

monthly_soc_csv = os.path.join(monthly_csv_folder, f"Battery_SOC_{month_name}.csv")
monthly_soc_df = pd.DataFrame()
for config in pv_bess_config:
    bname = config["bess_name"]
    soc_series = np.concatenate(monthly_soc_history[bname])
    monthly_soc_df[bname] = soc_series
monthly_soc_df.to_csv(monthly_soc_csv, index=False)

for config in pv_bess_config:
    bname = config["bess_name"]
    schedule = np.concatenate(monthly_optimized_battery_schedules[bname]) / config["bess_kWRated"]
    np.savetxt(os.path.join(monthly_csv_folder, f"{bname}_monthly_schedule_{month_name}.txt"),
               schedule, delimiter=",", fmt="%.6f")

T_month = T_day * num_days

# --------------------
# 以下部分利用整月调度输入 OpenDSS仿真，并采用中央储能方式计算社区SSR，同时记录月度电压数据
pv_month_dir = os.path.join(script_path, "pv_month", month_name)
load_month_dir = os.path.join(script_path, "load_month", month_name)

dss = py_dss_interface.DSS()
dss.text("Clear")
dss_file = os.path.join(script_path, "../../feeders/Feeder_3/Master33.dss")
dss.text(f"compile [{dss_file}]")
dss.text("calcv")
for config in pv_bess_config:
    pv_ls_name = config["pv_name"] + "_ls_monthly"
    conv_pv_file = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}_pu.txt")
    dss.text(f"New Loadshape.{pv_ls_name} npts={T_month} interval={time_step_hours} mult=(file={conv_pv_file}) useactual=yes")
    dss.text(f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kVA={config['pv_kVA']} Pmpp={config['pv_kVA']} irradiance=1 %cutin=0.1 %cutout=0.1 daily={pv_ls_name}")

    load_ls_name = config["load_name"] + "_ls_monthly"
    load_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    dss.text(f"New Loadshape.{load_ls_name} npts={T_month} interval={time_step_hours} mult=(file={load_txt}) useactual=yes")
    dss.text(f"New Load.{config['load_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kW={config['load_kW']} PF={config['pf']} daily={load_ls_name}")
    bname = config["bess_name"]
    # 月调度文件路径（之前保存的是经过约束后的月调度）
    monthly_schedule_file = os.path.join(monthly_csv_folder, f"{bname}_monthly_schedule_{month_name}.txt")
    battery_ls_name = f"{bname}_ls_monthly"
    dss.text(f"New Loadshape.{battery_ls_name} npts={T_month} interval={time_step_hours} mult=(file={monthly_schedule_file}) useactual=yes")

    dss.text(f"New Storage.{config['bess_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} %IdlingkW=0 "
             f"kWhStored={config['bess_kWhRated'] * 0.2} %EffCharge=95 %EffDischarge=95 %reserve={config['bess_lowlimit']} dispmode=FOLLOW daily={battery_ls_name}")

    dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")
dss.text("New Monitor.Transformer_TR1_mon Element=Transformer.TR1 Terminal=1 ppolar=no Mode=1")

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
monthly_monitor_dir = os.path.join(monitor_dir, f"Monthly_{month_name}")
os.makedirs(monthly_monitor_dir, exist_ok=True)
dss.text(f"Set DataPath={monthly_monitor_dir}")
for config in pv_bess_config:
    dss.text(f"Export monitor {config['pv_name']}_mon")
    dss.text(f"Export monitor {config['load_name']}_mon")
    dss.text(f"Export monitor {config['bess_name']}_mon")
dss.text("Export monitor Transformer_TR1_mon")
monthly_load_energy_day = 0.0
for config in pv_bess_config:
    load_array = config["load_data"]["load"].to_numpy()[:T_day]
    monthly_load_energy_day += load_array.sum() * time_step_hours
total_load_energy_month = monthly_load_energy_day * num_days
total_grid_purchase_energy = sum(p * time_step_hours for p in monthly_p_high if p > 0)
if total_load_energy_month > 0:
    community_SSR = (1 - total_grid_purchase_energy / total_load_energy_month) * 100
else:
    community_SSR = 0
monthly_voltage_csv = os.path.join(monthly_csv_folder, f"Node_Voltages_PSO_{month_name}.csv")
df_monthly_voltages = pd.DataFrame(node_voltages_monthly)
df_monthly_voltages.to_csv(monthly_voltage_csv, index=False)
all_voltages = np.concatenate(list(node_voltages_monthly.values()))
comm_avg_voltage = all_voltages.mean()
comm_min_voltage = all_voltages.min()
comm_max_voltage = all_voltages.max()
comm_p2p_voltage = comm_max_voltage - comm_min_voltage
comm_exceed_count = int(np.sum((all_voltages < 0.99) | (all_voltages > 1.01)))
comm_rmse = np.sqrt(np.mean((all_voltages - 1.0) ** 2))
rows = []
for config in pv_bess_config:
    bname = config["bess_name"]
    user_label = config["bus_name"].split(".")[0].lower()
    cost_with = monthly_household_cost_with_battery[bname]
    cost_only = monthly_household_cost_only_pv[bname]
    saving = cost_only - cost_with
    vm = {}
    if config["bus_name"] in node_voltages_monthly:
        voltages = np.array(node_voltages_monthly[config["bus_name"]])
        vm["Avg Voltage (p.u.)"] = voltages.mean()
        vm["Min Voltage (p.u.)"] = voltages.min()
        vm["Max Voltage (p.u.)"] = voltages.max()
        vm["P2P Voltage (p.u.)"] = voltages.max() - voltages.min()
        vm["Exceed Count"] = int(np.sum((voltages < 0.98) | (voltages > 1.02)))
        vm["RMSE (p.u.)"] = np.sqrt(np.mean((voltages - 1.0) ** 2))
    else:
        vm["Avg Voltage (p.u.)"] = np.nan
        vm["Min Voltage (p.u.)"] = np.nan
        vm["Max Voltage (p.u.)"] = np.nan
        vm["P2P Voltage (p.u.)"] = np.nan
        vm["Exceed Count"] = np.nan
        vm["RMSE (p.u.)"] = np.nan
    rows.append([user_label, cost_with, saving, "N/A", monthly_battery_degradation.get(bname, np.nan),
                 vm["Avg Voltage (p.u.)"], vm["Min Voltage (p.u.)"], vm["Max Voltage (p.u.)"],
                 vm["P2P Voltage (p.u.)"], vm["Exceed Count"], vm["RMSE (p.u.)"]])
rows.append(["community",
             sum(monthly_household_cost_with_battery.values()),
             sum(monthly_household_cost_only_pv.values()) - sum(monthly_household_cost_with_battery.values()),
             community_SSR,
             community_degradation,
             comm_avg_voltage,
             comm_min_voltage,
             comm_max_voltage,
             comm_p2p_voltage,
             comm_exceed_count,
             comm_rmse])
monthly_csv_filename = os.path.join(monthly_csv_folder, f"Battery_summary_PSO_{month_name}.csv")
df_monthly_summary = pd.DataFrame(rows, columns=["User",
                                                 "Monthly Cost (Battery PSO)",
                                                 "Cost Savings",
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
