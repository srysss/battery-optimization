import os
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
import numpy as np
import datetime

# Current working directory
script_path = os.path.dirname(os.path.abspath(__file__))

# ========== 2. Define 10 "Low Voltage" PV+Load+Storage Configurations ==========
pv_bess_config_lv = []
for i in range(1, 11):
    conf = {
        "pv_name": f"pv{i}",
        "bess_name": f"Battery{i}",
        "load_name": f"LOAD{i}",
        "bus_name": f"User{i}.1",  # Key: Corresponding to the new bus
        "kV": 0.277,  # Low voltage bus voltage
        "pv_kVA": 3,  # Unit: kW
        "load_kW": 1,
        "pf": 0.95,
        "bess_kWRated": 3,
        "bess_kWhRated": 3
    }
    pv_bess_config_lv.append(conf)

# ========== 2.1 Read PV/Load shape files and store in config ==========
for config in pv_bess_config_lv:
    pv_filename = os.path.join(script_path, "pv_data", "PV_OutputPower_3kW.txt")
    load_filename = os.path.join(script_path, "load_data", f"{config['load_name']}_OpenDSS_kW.txt")
    if os.path.exists(pv_filename):
        df_pv = pd.read_csv(pv_filename, header=None, names=["pv"])
    else:
        raise FileNotFoundError(f"Cannot find PV file: {pv_filename}")
    if os.path.exists(load_filename):
        df_load = pd.read_csv(load_filename, header=None, names=["load"])
    else:
        df_load = pd.DataFrame({"load": [1.0] * 48})
    config["pv_data"] = df_pv
    config["load_data"] = df_load

# Read price data
agile_purchase_file = os.path.join(script_path, "Agile_pricing_data", "Agile_pricing_data_1.csv")
agile_sale_file = os.path.join(script_path, "Agile_Outgoing_pricing_data", "Agile_Outgoing_pricing_data_1.csv")
df_purchase = pd.read_csv(agile_purchase_file, header=None, names=["time", "price"])
df_sale = pd.read_csv(agile_sale_file, header=None, names=["time", "price"])
purchase_prices = df_purchase["price"].tolist()
sale_prices = df_sale["price"].tolist()

total_number = 48
time_step_hours = 0.5  # 30-minute intervals
time_steps = [i * 0.5 for i in range(total_number)]  # Unit: hours


def calculate_metrics(battery_powers=None, original=False):
    """
    Calculates the electricity cost, SPBT, and RPVD.FalseTrue

    Args:
        battery_powers (numpy.ndarray): A 2D array of battery powers, where each row
            represents a battery and each column represents a time step. If None,
            the original logic is used.
        original (bool): If True, uses the original logic without PSO.

    Returns:
        tuple: A tuple containing:
            - electricity_cost (float): The total electricity cost.
            - household_cost_with_battery (dict): Household costs with battery.
            - household_cost_only_pv (dict): Household costs with only PV.
            - household_RPVD (dict): Household RPVD.
            - community_RPVD (float): Community RPVD.
            - community_spbt (float): Community SPBT.
            - grid_power_without (dict): Grid power without battery.
            - grid_power_with (dict): Grid power with battery.
    """

    global pv_bess_config_lv, purchase_prices, sale_prices, time_step_hours, total_number

    total_revenue = 0.0
    total_purchase_cost = 0.0

    # Initialize battery SOC for each user
    battery_soc = {conf['bess_name']: 20.0 for conf in pv_bess_config_lv}

    # Accumulate daily electricity costs for SPBT calculation
    household_cost_with_battery = {config['bess_name']: 0.0 for config in pv_bess_config_lv}
    household_cost_only_pv = {config['bess_name']: 0.0 for config in pv_bess_config_lv}

    # Store grid power data for peak-valley difference calculation
    grid_power_without = {config["bus_name"]: [] for config in pv_bess_config_lv}
    grid_power_with = {config["bus_name"]: [] for config in pv_bess_config_lv}

    # ========== Simulation Loop ==========
    for step_idx in range(total_number):
        total_self_supply = 0.0
        total_load_demand = 0.0

        for i, config in enumerate(pv_bess_config_lv):
            bess_name = config['bess_name']
            pv_name = config['pv_name']
            load_name = config['load_name']
            bus_name = config['bus_name']

            # Get current PV and load data
            pv_power_kW = config["pv_data"].iloc[step_idx]["pv"]
            load_kW = config["load_data"].iloc[step_idx]["load"]
            net_power = pv_power_kW - load_kW  # Net power in PV-only scenario

            max_charge_power = config['bess_kWRated']
            max_discharge_power = config['bess_kWRated']
            battery_capacity_kWh = config['bess_kWhRated']

            current_soc = battery_soc[bess_name]

            energy_charged = 0.0  # Current charging energy (kWh)
            energy_discharged = 0.0  # Current discharging energy (kWh)
            battery_cost_inc = 0.0  # Cost increment for this period

            if original:
                # Original Logic
                if net_power > 0:
                    # Excess generation, charge battery (target SOC <= 80%)
                    if current_soc < 80.0:
                        energy_needed_to_80 = ((80.0 - current_soc) / 100.0) * battery_capacity_kWh
                        actual_charge_power = min(net_power, max_charge_power)
                        max_possible_energy = actual_charge_power * time_step_hours * 0.95
                        if max_possible_energy >= energy_needed_to_80:
                            charge_power_to_80 = energy_needed_to_80 / time_step_hours
                            charge_power_to_80 = min(charge_power_to_80, actual_charge_power)
                            battery_soc[bess_name] = 80.0
                            energy_charged = energy_needed_to_80

                        else:
                            soc_incr = (max_possible_energy / battery_capacity_kWh) * 100
                            new_soc = min(current_soc + soc_incr, 100.0)
                            battery_soc[bess_name] = new_soc
                            energy_charged = max_possible_energy


                    surplus_energy = max(net_power * time_step_hours - energy_charged, 0.0)
                    revenue = (surplus_energy * sale_prices[step_idx]) / 100
                    total_revenue += revenue
                    battery_cost_inc = -revenue  # Sales revenue is negative cost


                elif net_power < 0:
                    # Load exceeds PV, discharge battery (target SOC >= 20%)
                    if current_soc > 20.0:
                        discharge_power = min(abs(net_power), max_discharge_power)
                        max_possible_discharge = (((current_soc - 20.0) / 100.0) * battery_capacity_kWh) / time_step_hours
                        discharge_power = min(discharge_power, max_possible_discharge)
                        energy_discharged = discharge_power * time_step_hours * 0.95
                        soc_decr = (energy_discharged / battery_capacity_kWh) * 100
                        new_soc = max(current_soc - soc_decr, 20.0)
                        battery_soc[bess_name] = new_soc


                        shortage_energy = max(abs(net_power) * time_step_hours - energy_discharged, 0.0)
                    else:

                        shortage_energy = abs(net_power) * time_step_hours

                    cost = (shortage_energy * purchase_prices[step_idx]) / 100
                    total_purchase_cost += cost
                    battery_cost_inc = cost  # Electricity purchase expenditure

                else:

                    battery_cost_inc = 0.0
            else:
                # Optimized Logic with PSO
                # Get battery power from PSO result
                battery_power = battery_powers[i, step_idx]  # kW. Positive for discharge, negative for charge

                if battery_power > 0:  # Discharge
                    energy_discharged = min(battery_power * time_step_hours,
                                            ((current_soc - 20.0) / 100.0) * battery_capacity_kWh)
                    soc_decr = (energy_discharged / battery_capacity_kWh) * 100
                    new_soc = max(current_soc - soc_decr, 20.0)
                    battery_soc[bess_name] = new_soc
                    shortage_energy = max(abs(net_power) * time_step_hours - energy_discharged, 0.0)

                elif battery_power < 0:  # Charge
                    energy_charged = min(abs(battery_power) * time_step_hours,
                                         ((80.0 - current_soc) / 100.0) * battery_capacity_kWh)
                    soc_incr = (energy_charged / battery_capacity_kWh) * 100
                    new_soc = min(current_soc + soc_incr, 80.0)
                    battery_soc[bess_name] = new_soc
                    surplus_energy = max(net_power * time_step_hours - energy_charged, 0.0)
                else:
                    surplus_energy = max(net_power * time_step_hours, 0.0)
                    shortage_energy = max(abs(net_power) * time_step_hours, 0.0)

                # Calculate cost
                if net_power > 0:
                    surplus_energy = max(net_power * time_step_hours - energy_charged, 0.0)
                    revenue = (surplus_energy * sale_prices[step_idx]) / 100
                    total_revenue += revenue
                    battery_cost_inc = -revenue  # Sales revenue is negative cost
                elif net_power < 0:
                    shortage_energy = max(abs(net_power) * time_step_hours - energy_discharged, 0.0)
                    cost = (shortage_energy * purchase_prices[step_idx]) / 100
                    total_purchase_cost += cost
                    battery_cost_inc = cost  # Electricity purchase expenditure
                else:
                    battery_cost_inc = 0.0

            # Accumulate electricity costs with BESS
            household_cost_with_battery[bess_name] += battery_cost_inc

            # Calculate electricity costs in the PV-only scenario
            net_power_only = pv_power_kW - load_kW
            if net_power_only > 0:
                surplus_only = net_power_only * time_step_hours
                revenue_only = (surplus_only * sale_prices[step_idx]) / 100
                cost_only_inc = -revenue_only
            elif net_power_only < 0:
                shortage_only = abs(net_power_only) * time_step_hours
                cost_only_inc = (shortage_only * purchase_prices[step_idx]) / 100
            else:
                cost_only_inc = 0.0
            household_cost_only_pv[bess_name] += cost_only_inc

            # Save grid-side power data for peak-valley difference calculation
            # PV-only scenario: directly use net_power_only
            grid_without_value = net_power_only

            # With BESS scenario:
            if original:
                if net_power > 0:
                    charge_power = energy_charged / time_step_hours  # kW
                    grid_with_value = net_power - charge_power
                elif net_power < 0:
                    discharge_power = energy_discharged / time_step_hours
                    grid_with_value = net_power + discharge_power
                else:
                    grid_with_value = 0.0
            else:
                grid_with_value = net_power - (energy_discharged / time_step_hours) + (energy_charged / time_step_hours)

            grid_power_without[bus_name].append(grid_without_value)
            grid_power_with[bus_name].append(grid_with_value)

    electricity_cost = total_purchase_cost - total_revenue

    # ========== Calculate Simple Payback Time (SPBT) for each household and the entire community ==========
    community_total_net_cost = 0.0
    community_total_daily_saving = 0.0

    for config in pv_bess_config_lv:
        bess_name = config['bess_name']
        # BESS cost only: unit cost £291/kWh + fixed installation fee £1200
        bess_cost = 291 * config['bess_kWhRated'] + 1200
        total_net_cost = bess_cost
        daily_saving = household_cost_only_pv[bess_name] - household_cost_with_battery[bess_name]
        community_total_net_cost += total_net_cost
        community_total_daily_saving += daily_saving

    if community_total_daily_saving > 0:
        community_spbt = community_total_net_cost / (community_total_daily_saving * 365)
    else:
        community_spbt = float('inf')

    # ========== Calculate Reduction of Peak-Valley Difference (RPVD) for each household and the entire community ==========
    household_RPVD = {}

    for config in pv_bess_config_lv:
        bus = config["bus_name"]
        # Calculate in PV-only scenario (take absolute value to ensure the difference is positive)
        peak_pv = max(grid_power_without[bus])
        valley_pv = min(grid_power_without[bus])
        diff_pv = abs(peak_pv - valley_pv)

        # Calculate in BESS scenario
        peak_bess = max(grid_power_with[bus])
        valley_bess = min(grid_power_with[bus])
        diff_bess = abs(peak_bess - valley_bess)

        if diff_pv != 0:
            RPVD = (diff_pv - diff_bess) / diff_pv * 100
        else:
            RPVD = 0.0

        household_RPVD[bus] = RPVD

    # Community-wide calculation
    community_grid_only = [sum(grid_power_without[bus][t] for bus in grid_power_without) for t in range(total_number)]
    community_grid_bess = [sum(grid_power_with[bus][t] for bus in grid_power_with) for t in range(total_number)]

    community_peak_pv = max(community_grid_only)
    community_valley_pv = min(community_grid_only)
    community_diff_pv = abs(community_peak_pv - community_valley_pv)

    community_peak_bess = max(community_grid_bess)
    community_valley_bess = min(community_grid_bess)
    community_diff_bess = abs(community_peak_bess - community_valley_bess)

    if community_diff_pv != 0:
        community_RPVD = (community_diff_pv - community_diff_bess) / community_diff_pv * 100
    else:
        community_RPVD = 0.0

    return electricity_cost, household_cost_with_battery, household_cost_only_pv, household_RPVD, community_RPVD, community_spbt, grid_power_without, grid_power_with


def cost_function(particles):
    """
    The cost function to be minimized by PSO.

    Args:
        particles (numpy.ndarray): A 3D array of particle positions, where each row
            represents a particle, each column represents a battery, and each slice
            represents a time step.

    Returns:
        numpy.ndarray: An array of cost values for each particle.
    """
    n_particles = particles.shape[0]
    costs = np.zeros(n_particles)

    for i in range(n_particles):
        battery_powers = particles[i].reshape(len(pv_bess_config_lv), total_number)
        electricity_cost, _, _, _, _, _, _, _ = calculate_metrics(battery_powers, original=False)
        costs[i] = electricity_cost

    return costs

# Set a random seed for reproducibility
np.random.seed(42)

# Get current date and time
now = datetime.datetime.now(datetime.timezone.utc)
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")

# Hardcoded User's Login
user_login = "srysss"

# ========== Original Logic Calculation ==========
print("=============== Original Logic Calculation ===============")
(original_electricity_cost,
 original_household_cost_with_battery,
 original_household_cost_only_pv,
 original_household_RPVD,
 original_community_RPVD,
 original_community_spbt,
 original_grid_power_without,
 original_grid_power_with) = calculate_metrics(original=True)

print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {formatted_date_time}")
print(f"Current User's Login: {user_login}\n")

print("【仅 PV 情景】")
for config in pv_bess_config_lv:
    bess_name = config["bess_name"]
    cost_only = original_household_cost_only_pv[bess_name]
    print(f"{bess_name}: Electricity Cost = {cost_only:.4f} /day")
community_cost_only = sum(original_household_cost_only_pv.values())
print(f"Community Electricity Cost(only Pv): ￡{community_cost_only:.4f} /day\n")

print("【带电池 原始调度 情景】")
for config in pv_bess_config_lv:
    bess_name = config["bess_name"]
    cost_with = original_household_cost_with_battery[bess_name]
    # 储能系统成本计算：单位成本 291/kWh + 固定安装费 1200
    bess_cost = 291 * config['bess_kWhRated'] + 1200
    # 每天节省的费用：仅 PV 情景成本减去带电池调度的成本
    daily_saving = original_household_cost_only_pv[bess_name] - cost_with
    # 计算简单投资回收期（SPBT），如果 daily_saving 不足则设为无穷大
    spbt = bess_cost / (daily_saving * 365) if daily_saving > 0 else float('inf')
    spbt_str = f"{spbt:.2f}" if spbt != float('inf') else "inf"
    # 获取降低峰谷差（RPVD）的百分比
    rpvd = original_household_RPVD[config["bus_name"]]
    print(f"{bess_name}: Electricity Cost = {cost_with:.4f} /day, SPBT = {spbt_str} years, RPVD = {rpvd:.2f}%")
community_cost_with = sum(original_household_cost_with_battery.values())
community_spbt = original_community_spbt
print(f"Community Electricity Cost(With Battery): ￡{community_cost_with:.4f} /day")
print(f"Community SPBT: {community_spbt:.2f} years")
print(f"Community RPVD: {original_community_RPVD:.2f}%\n")
# ========== PSO Optimization ==========
# Define PSO parameters
# Define PSO parameters (adjusted)
n_particles = 50  # Increased number of particles
iterations = 50 # Increased number of iterations
options = {'c1': 0.7, 'c2': 0.5, 'w': 0.7}  # Tuned PSO parameters

# Initialize the swarm
optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                     dimensions=len(pv_bess_config_lv) * total_number,
                                     options=options,
                                     bounds=(-3 * np.ones(len(pv_bess_config_lv) * total_number),
                                             3 * np.ones(len(pv_bess_config_lv) * total_number)))

# Run optimization
cost, pos = optimizer.optimize(cost_function, iters=iterations)

# Reshape the best position to get the battery powers
best_battery_powers = pos.reshape(len(pv_bess_config_lv), total_number)

# Calculate the metrics with optimized battery powers
(optimized_electricity_cost,
 optimized_household_cost_with_battery,
 optimized_household_cost_only_pv,
 optimized_household_RPVD,
 optimized_community_RPVD,
 optimized_community_spbt,
 optimized_grid_power_without,
 optimized_grid_power_with) = calculate_metrics(best_battery_powers, original=False)

print("【带电池优化调度情景】")
community_total_net_cost = 0.0
community_total_daily_saving = 0.0

for config in pv_bess_config_lv:
    bess_name = config["bess_name"]
    cost_with = optimized_household_cost_with_battery[bess_name]
    bess_cost = 291 * config['bess_kWhRated'] + 1200
    daily_saving = original_household_cost_only_pv[bess_name] - cost_with
    community_total_net_cost += bess_cost
    community_total_daily_saving += daily_saving
    spbt = bess_cost / (daily_saving * 365) if daily_saving > 0 else float('inf')
    spbt_str = f"{spbt:.2f}" if spbt != float('inf') else "inf"
    rpvd = optimized_household_RPVD[config["bus_name"]]
    print(f"{bess_name}: Electricity Cost = {cost_with:.4f} /day, SPBT = {spbt_str} years, RPVD = {rpvd:.2f}%")

community_cost_with = sum(optimized_household_cost_with_battery.values())
community_spbt = community_total_net_cost / (community_total_daily_saving * 365) if community_total_daily_saving > 0 else float('inf')
community_spbt_str = f"{community_spbt:.2f}" if community_spbt != float('inf') else "inf"

print(f"Community Electricity Cost(With Battery): ￡{community_cost_with:.4f} /day")
print(f"Community SPBT: {community_spbt_str} years")
print(f"Community RPVD: {optimized_community_RPVD:.2f}%")