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

# 加载新的 PV 和负载数据
try:
    pv_data = pd.read_excel(os.path.join(script_path, "real_solar_radiation.xlsx"))
    load_data = pd.read_excel(os.path.join(script_path, "real_load_profile.xlsx"))
    print("✅ Solar Radiation 数据加载成功")
    print("✅ Load Profile 数据加载成功")
except FileNotFoundError as e:
    print(f"❌ 文件未找到: {e}")
    exit(1)
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit(1)

# 定义 PV 系统和储能系统的配置
pv_bess_config = [
    {"pv_name": "PV1", "bess_name": "Battery1", "kVA_rated": 10, "bus_name": "671"},
    {"pv_name": "PV2", "bess_name": "Battery2", "kVA_rated": 10, "bus_name": "632"},
    {"pv_name": "PV3", "bess_name": "Battery3", "kVA_rated": 10, "bus_name": "633"},
]

# 设置电网电压上下限
v_min_limit = 0.94
v_max_limit = 1.1

# 定义电池容量和时间步长
battery_capacity_kWh = 10.0  # 电池总容量为 10 kWh
time_step_hours = 0.5  # 时间步长为半小时

# 结果文件路径
output_file = os.path.join(script_path, "simulation_results.csv")
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Time", "PV_Name", "PV_Power_kW", "BESS_Name", "BESS_Power_kW", "SOC", "Load_Demand_kW", "Grid_Support_kW", "Voltage_pu"
    ])

    # 初始化电池的初始 SOC (State of Charge)
    battery_soc = {"Battery1": 50.0, "Battery2": 50.0, "Battery3": 50.0}  # 初始 SOC 为 50%

    # 设置电池最大充放电功率限制
    max_discharge_power = 3.0  # 最大放电功率 3 kW
    max_charge_power = 3.0     # 最大充电功率 3 kW

    time_series = []
    soc_series = {"Battery1": [], "Battery2": [], "Battery3": []}
    pv_power_series = {"PV1": [], "PV2": [], "PV3": []}
    load_series = []
    voltage_series = {bus: [] for bus in dss.circuit.buses_names()}

    # 运行每半小时仿真
    for idx, row in load_data.iterrows():
        time = row['tstp']
        time_series.append(time)
        total_load = 0

        for config in pv_bess_config:
            pv_name = config["pv_name"]
            bess_name = config["bess_name"]
            bus_name = config["bus_name"]

            # 获取当前半小时的 Solar Radiation 数据
            solar_radiation = pv_data.loc[idx, "Solar Radiation"]

            # 假设 PV 系统的额定功率为 10 kW，计算 PV 发电功率
            pv_power_kW = 10 * (solar_radiation / 1000)  # 按比例缩放
            pv_power_series[pv_name].append(pv_power_kW)

            # 获取当前负载数据
            load_kw = row[f"Load{pv_bess_config.index(config) + 1}(kWh/hh)"] * 2  # 每半小时数据换算成 kW
            total_load += load_kw

            # 计算净功率
            net_power = pv_power_kW - load_kw

            # 解决方案仿真
            dss.solution_solve()

            # 获取所有总线的电压值
            all_bus_voltages = dss.circuit.buses_vmag_pu()

            # 默认电池状态为 IDLE
            battery_state = "IDLE"
            battery_power = 0.0
            grid_support = 0.0

            # 针对每个电池单独计算其充放电策略
            if net_power > 0:
                # PV 发电过剩，电池充电
                if battery_soc[bess_name] < 80.0:
                    charge_power = min(net_power, max_charge_power)
                    soc_increment = (charge_power * time_step_hours) / battery_capacity_kWh * 100  # 计算 SOC 增量
                    battery_soc[bess_name] = min(battery_soc[bess_name] + soc_increment, 100.0)
                    battery_state = "CHARGING"
                    battery_power = charge_power
            elif net_power < 0:
                # PV 发电不足，电池放电
                if battery_soc[bess_name] > 20.0:
                    discharge_power = min(abs(net_power), max_discharge_power)
                    max_possible_discharge = (battery_soc[bess_name] - 20.0) / 100 * battery_capacity_kWh / time_step_hours
                    discharge_power = min(discharge_power, max_possible_discharge)
                    soc_decrement = (discharge_power * time_step_hours) / battery_capacity_kWh * 100  # 计算 SOC 减少量
                    battery_soc[bess_name] = max(battery_soc[bess_name] - soc_decrement, 20.0)
                    battery_state = "DISCHARGING"
                    battery_power = discharge_power
                    grid_support = abs(net_power) - discharge_power
                else:
                    # 当电池达到最低 SOC 时，网格承担全部负载
                    grid_support = abs(net_power)

            soc_series[bess_name].append(battery_soc[bess_name])

            # 写入 CSV 文件
            writer.writerow([
                time, pv_name, pv_power_kW, bess_name, battery_power, battery_soc[bess_name], load_kw, grid_support, all_bus_voltages[dss.circuit.buses_names().index(bus_name)]
            ])

            # 打印仿真信息
            print(
                f"{time}: {pv_name} Generation = {pv_power_kW:.2f} kW, Voltage = {all_bus_voltages[dss.circuit.buses_names().index(bus_name)]:.4f}, Load Demand = {load_kw:.2f} kW, Grid Support = {grid_support:.2f} kW")
            if battery_state == "DISCHARGING":
                print(f"{time}: {bess_name} is discharging {battery_power:.2f} kW, SOC = {battery_soc[bess_name]:.2f}%")
            elif battery_state == "CHARGING":
                print(f"{time}: {bess_name} is charging {battery_power:.2f} kW, SOC = {battery_soc[bess_name]:.2f}%")
            else:
                print(f"{time}: {bess_name} is idle, SOC = {battery_soc[bess_name]:.2f}%")

        load_series.append(total_load)
        for bus in dss.circuit.buses_names():
            voltage_series[bus].append(all_bus_voltages[dss.circuit.buses_names().index(bus)])

# 绘制 SOC 图像
plt.figure()
for battery, soc in soc_series.items():
    plt.plot(time_series, soc, label=f"{battery} SOC")
plt.title("Battery SOC Over Time")
plt.xlabel("Time")
plt.ylabel("SOC (%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 绘制 PV 发电功率图像
plt.figure()
for pv, power in pv_power_series.items():
    plt.plot(time_series, power, label=f"{pv} Power")
plt.title("PV Power Over Time")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 绘制负载需求图像
plt.figure()
plt.plot(time_series, load_series, label="Total Load Demand")
plt.title("Load Demand Over Time")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 绘制总线电压图像
plt.figure()
for bus, voltage in voltage_series.items():
    plt.plot(time_series, voltage, label=f"Bus {bus} Voltage")
plt.title("Bus Voltages Over Time")
plt.xlabel("Time")
plt.ylabel("Voltage (pu)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
