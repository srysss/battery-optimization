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

###################################################
# ========== 0. Create Output Folders =========
###################################################
script_path = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_path, "Results")
bat_dir = os.path.join(results_dir, "Battery_Scenario_PSO")  # PSO结果存放文件夹
figures_dir = os.path.join(bat_dir, "Figures")
csv_dir = os.path.join(bat_dir, "CSV")
monitor_dir = os.path.join(bat_dir, "Monitor_Exports")
for folder in [results_dir, bat_dir, figures_dir, csv_dir, monitor_dir]:
    os.makedirs(folder, exist_ok=True)

###################################################
# ========== 1. Define System Config =========
###################################################
# 配置10个低压PV+Load+Storage系统，母线编号如下
bus_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18]
pv_bess_config = []
for i, bus in enumerate(bus_numbers):
    conf = {
        "pv_name": f"pv{i + 1}",
        "bess_name": f"Battery{i + 1}",
        "load_name": f"LOAD{i + 1}",
        "bus_name": f"{bus}.1",  # 与原馈线母线号一致
        "kV": 0.23,
        "pv_kVA": 3,
        "load_kW": 1,
        "pf": 0.95,
        "bess_kWRated": 3,
        "bess_kWhRated": 3
    }
    pv_bess_config.append(conf)

###################################################
# ========== 2. Read Monthly Data =========
###################################################
# 选择仿真月份（1-12），此处选择7月
chosen_month = 7
month_name = datetime.date(1900, chosen_month, 1).strftime("%B")  # 例如 "July"

# 设置月数据所在目录
pv_month_dir = os.path.join(script_path, "pv_month", month_name)
load_month_dir = os.path.join(script_path, "load_month", month_name)

# 时间步长（单位：小时）
time_step_hours = 0.5

# 逐个配置读取对应的月数据文件
for config in pv_bess_config:
    pv_file = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}.txt")
    load_file = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    if os.path.exists(pv_file):
        dfp = pd.read_csv(pv_file, header=None, names=["pv"])
    else:
        raise FileNotFoundError(f"Cannot find: {pv_file}")
    if os.path.exists(load_file):
        dfl = pd.read_csv(load_file, header=None, names=["load"])
    else:
        raise FileNotFoundError(f"Cannot find: {load_file}")
    config["pv_data"] = dfp
    config["load_data"] = dfl

# 数据总步数，即该月的时间步数
T = len(pv_bess_config[0]["pv_data"])
print(f"Chosen month = {month_name}, total time steps (T) = {T}")

# 读取电价数据
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


###################################################
# ========== 3. Utilities for Micro-cycle =========
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


###################################################
# ========== 4. Initialize OpenDSS & Create Components =========
###################################################
dss = py_dss_interface.DSS()
dss_file = os.path.join(script_path, "../../feeders/Feeder_3/Master33.dss")
dss.text(f"compile [{dss_file}]")
dss.text("calcv")

for config in pv_bess_config:
    # 创建PV负荷曲线与PV系统
    pv_ls_name = config["pv_name"] + "_ls"
    pv_txt = os.path.join(pv_month_dir, f"pv_output_kW_{month_name}.txt")
    dss.text(f"New Loadshape.{pv_ls_name} npts={T} interval={time_step_hours} mult=(file={pv_txt}) useactual=yes")
    dss.text(f"New PVSystem.{config['pv_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kVA={config['pv_kVA']} Pmpp={config['pv_kVA']} irradiance=1 %cutin=0.1 %cutout=0.1 daily={pv_ls_name}")

    # 创建负荷曲线与负荷对象
    load_ls_name = config["load_name"] + "_ls"
    load_txt = os.path.join(load_month_dir, f"{config['load_name']}_kW_{month_name}.txt")
    dss.text(f"New Loadshape.{load_ls_name} npts={T} interval={time_step_hours} mult=(file={load_txt}) useactual=yes")
    dss.text(f"New Load.{config['load_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kW={config['load_kW']} PF={config['pf']} daily={load_ls_name}")

    # 创建储能（电池），初始存储能量为额定容量的20%
    dss.text(f"New Storage.{config['bess_name']} phases=1 bus1={config['bus_name']} kV={config['kV']} "
             f"kWRated={config['bess_kWRated']} kWhRated={config['bess_kWhRated']} "
             f"kWhStored={config['bess_kWhRated'] * 0.2} %EffCharge=95 %EffDischarge=95 dispmode=DEFAULT")

    # 创建监视器
    dss.text(f"New Monitor.{config['pv_name']}_mon Element=PVSystem.{config['pv_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['load_name']}_mon Element=Load.{config['load_name']} Terminal=1 ppolar=no Mode=1")
    dss.text(f"New Monitor.{config['bess_name']}_mon Element=Storage.{config['bess_name']} Terminal=1 Mode=7")

print("✅ DSS setup done.")


###################################################
# ========== 5. Define Simulation Function (Vectorized Metrics) =========
###################################################
def calculate_metrics_vectorized(battery_powers):
    """
    输入：
      - battery_powers: (n_users, T) 的调度矩阵（单位 kW）
    输出：
      返回各项指标及各用户数据（字典格式）
    """
    global pv_bess_config, purchase_prices, sale_prices, time_step_hours, T
    n = len(pv_bess_config)

    # 构造各用户数据矩阵
    pv_array = np.empty((n, T))
    load_array = np.empty((n, T))
    battery_capacity = np.empty(n)
    bess_kWRated = np.empty(n)
    bus_names = []
    bess_names = []
    for i, config in enumerate(pv_bess_config):
        pv_array[i, :] = config["pv_data"]["pv"].to_numpy()
        load_array[i, :] = config["load_data"]["load"].to_numpy()
        battery_capacity[i] = config["bess_kWhRated"]
        bess_kWRated[i] = config["bess_kWRated"]
        bus_names.append(config["bus_name"])
        bess_names.append(config["bess_name"])

    sale_prices_arr = np.array(sale_prices)
    purchase_prices_arr = np.array(purchase_prices)

    # 初始化变量
    battery_soc = np.full(n, 20.0)
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

    efficiency_charging = 0.95
    efficiency_discharging = 0.95

    for t in range(T):
        pv_t = pv_array[:, t]
        load_t = load_array[:, t]
        net_power = pv_t - load_t  # 单位：kW

        # PSO优化调度方案：battery_powers为传入的调度矩阵（单位 kW）
        battery_power = battery_powers[:, t]
        battery_power = np.clip(battery_power, -bess_kWRated, bess_kWRated)
        energy_action = battery_power * time_step_hours  # 理论能量变化（kWh）
        energy_charged = np.where(battery_power < 0, -energy_action * efficiency_charging, 0.0)
        energy_discharged = np.where(battery_power > 0, energy_action * efficiency_discharging, 0.0)
        proposed_new_soc = battery_soc + (energy_charged - energy_discharged) / battery_capacity * 100
        battery_soc = np.clip(proposed_new_soc, 20.0, 80.0)
        grid_energy = net_power * time_step_hours - energy_charged + energy_discharged

        # 计算成本：盈余时产生收入，不足时产生购电成本
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

        # PV-only 情况（不考虑电池调度）
        cost_only_inc = np.where(net_power >= 0,
                                 -net_power * time_step_hours * sale_prices_arr[t] / 100.0,
                                 np.abs(net_power) * time_step_hours * purchase_prices_arr[t] / 100.0)
        household_cost_only_pv += cost_only_inc

        grid_power_without[:, t] = net_power * time_step_hours
        grid_power_with[:, t] = grid_energy

        load_energy = load_t * time_step_hours
        pv_energy = pv_t * time_step_hours
        only_pv_supply += np.minimum(pv_energy, load_energy)
        local_supply = np.minimum(load_energy, pv_energy + energy_discharged)
        bess_supply += local_supply
        total_load_energy += load_energy

        soc_history[:, t] = battery_soc

    electricity_cost = total_purchase_cost - total_revenue

    # 计算 RPVD 指标
    diff_pv = np.ptp(grid_power_without, axis=1)
    diff_bess = np.ptp(grid_power_with, axis=1)
    household_RPVD = np.where(diff_pv != 0, (diff_pv - diff_bess) / diff_pv * 100, 0.0)
    community_grid_without = np.sum(grid_power_without, axis=0)
    community_grid_with = np.sum(grid_power_with, axis=0)
    community_diff_pv = np.ptp(community_grid_without)
    community_diff_bess = np.ptp(community_grid_with)
    community_RPVD = ((community_diff_pv - community_diff_bess) / community_diff_pv * 100
                      if community_diff_pv != 0 else 0.0)

    # 构造结果字典
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

    return (electricity_cost,
            household_cost_with_battery_dict,
            household_cost_only_pv_dict,
            household_RPVD_dict,
            community_RPVD,
            total_purchase_cost - total_revenue,
            grid_power_without_dict,
            grid_power_with_dict,
            soc_history_dict)


###################################################
# ========== 6. Single-User PSO Optimization Function =========
###################################################
def get_cost_function_single(household_index):
    """
    返回单用户的目标函数，优化目标为该用户全月净电费（仅考虑带电池情形）。
    其它用户调度均固定为 0。
    """

    def cost_function(particles):
        n_particles = particles.shape[0]
        costs = np.zeros(n_particles)
        for i in range(n_particles):
            battery_schedule = particles[i]  # 一维数组，长度为 T
            # 构造整个社区的电池调度矩阵：仅当前用户使用优化变量，其它用户均设为 0
            battery_powers = np.zeros((len(pv_bess_config), T))
            battery_powers[household_index, :] = battery_schedule
            electricity_cost, household_cost_with_battery_dict, *_ = calculate_metrics_vectorized(battery_powers)
            bess_name = pv_bess_config[household_index]['bess_name']
            costs[i] = household_cost_with_battery_dict[bess_name]
        return costs

    return cost_function


###################################################
# ========== 7. Main Simulation =========
###################################################
# -------------------------------
# 先计算 PV-only（不考虑电池调度）的成本
print("\n=============== Computing PV-Only Costs (No Battery) ===============")
zero_battery_powers = np.zeros((len(pv_bess_config), T))
(_, _, household_cost_only_pv, _, _, _, _, _, _) = calculate_metrics_vectorized(zero_battery_powers)
print("PV-Only Costs per user (no battery):")
for config in pv_bess_config:
    bname = config["bess_name"]
    user = config["bus_name"].split(".")[0].lower()
    print(f"  {user}: {household_cost_only_pv[bname]:.4f}")

# -------------------------------
# 针对每个用户单独优化调度
print("\n=============== PSO Optimized Scheduling (Single-User Optimization) ===============")
options = {'c1': 1.2, 'c2': 1.2, 'w': 0.5}
n_particles = 150
iterations = 150

optimized_battery_schedules = {}
for idx, config in enumerate(pv_bess_config):
    print(f"\n开始优化用户 {config['bus_name']} ...")
    bounds = (-3 * np.ones(T), 3 * np.ones(T))
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=T, options=options, bounds=bounds)
    cost, pos = optimizer.optimize(get_cost_function_single(idx), iters=iterations, verbose=True)
    optimized_battery_schedules[config['bess_name']] = pos
    print(f"用户 {config['bus_name']} 优化结束，最优成本：{cost:.4f}")

# 将各用户的最优调度组合成全社区调度矩阵
best_battery_powers = np.zeros((len(pv_bess_config), T))
for idx, config in enumerate(pv_bess_config):
    best_battery_powers[idx, :] = optimized_battery_schedules[config['bess_name']]

# 使用组合后的最优调度方案进行全社区仿真
(elec_cost_opt,
 household_cost_with_battery_opt,
 household_cost_only_pv_opt,
 household_RPVD_opt,
 community_RPVD_opt,
 net_cost_opt,
 grid_power_without_opt,
 grid_power_with_opt,
 soc_history_opt) = calculate_metrics_vectorized(best_battery_powers)

print("\n[PSO Optimized Scheduling Results]")
for config in pv_bess_config:
    bname = config["bess_name"]
    user = config["bus_name"].split(".")[0].lower()
    cost_with = household_cost_with_battery_opt[bname]
    cost_only = household_cost_only_pv_opt[bname]
    saving = cost_only - cost_with
    bess_cost = 291 * config["bess_kWhRated"] + 1200
    spbt = bess_cost / saving if saving > 0 else float("inf")
    rpvd = household_RPVD_opt[config["bus_name"]]
    print(
        f"  {user}: Monthly Cost = {cost_with:.4f}, Saving = {saving:.4f}, SPBT = {spbt:.2f} months, RPVD = {rpvd:.2f}%")

###################################################
# ========== 8. Generate CSV Summary and Plot =========
###################################################
csv_filename = os.path.join(csv_dir, f"Battery_summary_PSO_{month_name}.csv")
rows = []
for config in pv_bess_config:
    user_label = config["bus_name"].split(".")[0].lower()
    bname = config["bess_name"]
    cost_with = household_cost_with_battery_opt[bname]
    cost_only = household_cost_only_pv_opt[bname]
    saving = cost_only - cost_with
    rpvd = household_RPVD_opt[config["bus_name"]]
    rows.append([user_label, cost_with, saving, rpvd])
rows.append(["community", sum(household_cost_with_battery_opt.values()),
             sum(household_cost_only_pv_opt.values()) - sum(household_cost_with_battery_opt.values()),
             community_RPVD_opt])
df_summary = pd.DataFrame(rows, columns=["User", "Monthly Cost (Battery)", "Cost Savings", "RPVD (%)"])
df_summary.to_csv(csv_filename, index=False)
print(f"\n================ CSV file '{csv_filename}' generated =================")

# 可选：绘制 SOC 历程或网侧功率曲线图
plt.figure(figsize=(12, 8))
for idx, config in enumerate(pv_bess_config):
    plt.plot(soc_history_opt[config["bess_name"]], label=f"User {config['bus_name'].split('.')[0]}")
plt.xlabel('Time (half-hour steps)')
plt.ylabel('Battery SOC (%)')
plt.title(f'SOC Profiles with PSO Optimization - {month_name}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(figures_dir, f"SOC_profiles_PSO_{month_name}.png"))
print(f"SOC profiles plot saved to {figures_dir}")