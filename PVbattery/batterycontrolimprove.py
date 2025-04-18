import os
import pandas as pd
import py_dss_interface
import csv
import matplotlib.pyplot as plt

# 初始化 DSS 接口
dss = py_dss_interface.DSS()

# 编译 IEEE 13 Node Test Feeder
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")
print(f"✅ 当前工作目录: {script_path}")
dss.text(f"compile [{dss_file}]")
print(f"✅ IEEE 13 Node Test Feeder compiled: {dss_file}")

# 加载极端场景的 PV 和负载数据
try:
    pv_data = pd.read_csv(os.path.join(script_path, "pv_generation_extreme.csv"))
    load_data = pd.read_csv(os.path.join(script_path, "load_profile_extreme.csv"))
    print("✅ 极端 PV 数据加载成功")
    print("✅ 极端负载数据加载成功")
except FileNotFoundError as e:
    print(f"❌ 文件未找到: {e}")
    exit(1)
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit(1)

# 定义 PV 系统和储能系统的配置
pv_bess_config = [
    {"pv_name": "PV1", "bess_name": "Battery1", "kVA_rated": 500, "bus_name": "Bus1"},
    {"pv_name": "PV2", "bess_name": "Battery2", "kVA_rated": 500, "bus_name": "Bus2"},
    {"pv_name": "PV3", "bess_name": "Battery3", "kVA_rated": 500, "bus_name": "Bus3"},
]

# 设置电网电压上下限
v_min_limit = 0.94
v_max_limit = 1.1

# 结果文件路径
output_file = os.path.join(script_path, "simulation_results.csv")
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Hour", "PV_Name", "PV_Power_kW", "BESS_Name", "BESS_Power_kW", "SOC", "Load_Demand_kW", "Grid_Support_kW", "Voltage_pu"
    ])

    # 初始化电池的初始 SOC (State of Charge)
    battery_soc = {"Battery1": 50.0, "Battery2": 50.0, "Battery3": 50.0}  # 初始 SOC 为 50%

    # 设置电池最大充放电功率限制
    max_discharge_power = 50.0  # 最大放电功率 50 kW
    max_charge_power = 50.0     # 最大充电功率 50 kW

    # 初始化曲线数据
    pv_power_list = {"PV1": [0] * 24, "PV2": [0] * 24, "PV3": [0] * 24}
    load_demand_list = {"Load1": [0] * 24, "Load2": [0] * 24, "Load3": [0] * 24}
    soc_list = {"Battery1": [50.0] * 24, "Battery2": [50.0] * 24, "Battery3": [50.0] * 24}
    voltage_list = {"Bus1": [0] * 24, "Bus2": [0] * 24, "Bus3": [0] * 24}

    # 运行每小时仿真
    for hour in range(24):
        for config in pv_bess_config:
            pv_name = config["pv_name"]
            bess_name = config["bess_name"]
            kVA_rated = config["kVA_rated"]
            bus_name = config["bus_name"]

            # 读取 PV 辐照度数据
            irradiance = pv_data.loc[hour, pv_name]

            # 计算 PV 发电功率
            pv_power_kW = irradiance * kVA_rated
            pv_power_list[pv_name][hour] = pv_power_kW

            # 设置用户负载
            load_kw = load_data.loc[hour, f"Load{pv_bess_config.index(config) + 1} (kW)"]
            load_demand_list[f"Load{pv_bess_config.index(config) + 1}"][hour] = load_kw
            dss.text(f"Edit Load.Load{pv_bess_config.index(config) + 1} kW={load_kw}")

            # 计算净功率
            net_power = pv_power_kW - load_kw

            # 获取当前总线电压
            dss.solution.solve()
            bus_voltages = dss.circuit.bus_voltage_pu(bus_name)

            # 打印每个总线节点的电压
            print(f"Hour {hour}: {bus_name} Voltage = {bus_voltages[0]:.4f}")

            # 记录每个总线节点的电压
            voltage_list[bus_name][hour] = bus_voltages[0]

            # 默认电池状态为 IDLE
            battery_state = "IDLE"
            battery_power = 0.0
            grid_support = 0.0

            if net_power > 0:
                # PV 发电过剩，电池充电
                charge_power = min(net_power, max_charge_power)
                if battery_soc[bess_name] < 100.0:
                    battery_soc[bess_name] = min(battery_soc[bess_name] + charge_power * 0.1, 100.0)
                    battery_state = "CHARGING"
                    battery_power = charge_power
            elif net_power < 0:
                if battery_soc[bess_name] > 10.0:
                    # PV 发电不足，电池放电
                    discharge_power = min(abs(net_power), max_discharge_power)
                    battery_soc[bess_name] = max(battery_soc[bess_name] - discharge_power * 0.1, 0.0)
                    battery_state = "DISCHARGING"
                    battery_power = discharge_power

            # 更新 SOC 曲线数据
            soc_list[bess_name][hour] = battery_soc[bess_name]

            # 写入 CSV 文件
            writer.writerow([
                hour, pv_name, pv_power_kW, bess_name, battery_power, battery_soc[bess_name], load_kw, grid_support, bus_voltages[0]
            ])

            # 打印仿真信息
            print(
                f"Hour {hour}: {pv_name} Generation = {pv_power_kW:.2f} kW, Voltage = {bus_voltages[0]:.4f}, Load Demand = {load_kw:.2f} kW, Grid Support = {grid_support:.2f} kW")
            if battery_state == "DISCHARGING":
                print(f"Hour {hour}: {bess_name} is discharging {battery_power:.2f} kW, SOC = {battery_soc[bess_name]:.2f}%")
            elif battery_state == "CHARGING":
                print(f"Hour {hour}: {bess_name} is charging {battery_power:.2f} kW, SOC = {battery_soc[bess_name]:.2f}%")
            else:
                print(f"Hour {hour}: {bess_name} is idle, SOC = {battery_soc[bess_name]:.2f}%")

# 绘制 PV 发电量曲线
plt.figure(figsize=(12, 6))
for pv, power_values in pv_power_list.items():
    plt.plot(range(24), power_values, marker='o', label=f'{pv} Generation (kW)')
plt.title('PV Generation Over 24 Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Power (kW)')
plt.grid(True)
plt.legend()
plt.show()

# 绘制负载需求曲线
plt.figure(figsize=(12, 6))
for load, demand_values in load_demand_list.items():
    plt.plot(range(24), demand_values, marker='x', label=f'{load} Demand (kW)')
plt.title('Load Demand Over 24 Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Power (kW)')
plt.grid(True)
plt.legend()
plt.show()

# 绘制电池 SOC 变化曲线
plt.figure(figsize=(12, 6))
for battery, soc_values in soc_list.items():
    plt.plot(range(24), soc_values, marker='o', label=f'{battery} SOC (%)')
plt.title('Battery SOC Over 24 Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('SOC (%)')
plt.grid(True)
plt.legend()
plt.show()

# 绘制电网电压变化曲线
plt.figure(figsize=(12, 6))
for bus, voltage_values in voltage_list.items():
    plt.plot(range(24), voltage_values, marker='o', label=f'{bus} Voltage (p.u.)')
plt.axhline(y=v_min_limit, color='r', linestyle='--', label='Min Voltage Limit (0.94 p.u.)')
plt.axhline(y=v_max_limit, color='r', linestyle='--', label='Max Voltage Limit (1.1 p.u.)')
plt.title('Grid Voltage Over 24 Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Voltage (p.u.)')
plt.grid(True)
plt.legend()
plt.show()

print(f"Simulation results saved to {output_file}")
