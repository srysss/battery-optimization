import os
import pandas as pd
import py_dss_interface
import csv

# 初始化 DSS 接口
dss = py_dss_interface.DSS()

# 编译 IEEE 13 Node Test Feeder
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")
print(f"✅ 当前工作目录: {script_path}")
dss.text(f"compile [{dss_file}]")
print(f"✅ IEEE 13 Node Test Feeder compiled: {dss_file}")

# 加载 PV 和负载数据
try:
    pv_data = pd.read_csv(os.path.join(script_path, "pv_generation.csv"))
    load_data = pd.read_csv(os.path.join(script_path, "load_profile.csv"))
    print("✅ PV 数据加载成功")
    print("✅ 负载数据加载成功")
except FileNotFoundError as e:
    print(f"❌ 文件未找到: {e}")
    exit(1)
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit(1)

# 定义 PV 系统和储能系统的配置
pv_bess_config = [
    {"pv_name": "PV1", "bess_name": "Battery1", "kVA_rated": 500},
    {"pv_name": "PV2", "bess_name": "Battery2", "kVA_rated": 500},
    {"pv_name": "PV3", "bess_name": "Battery3", "kVA_rated": 500},
]

# 结果文件路径
output_file = os.path.join(script_path, "simulation_results.csv")
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Hour", "PV_Name", "PV_Power_kW", "PV_Power_kVAR",
        "BESS_Name", "BESS_Power_kW", "BESS_Power_kVAR", "SOC",
        "Load_Demand_kW", "Grid_Support_kW", "Total_Power_kW", "Total_Power_kVAR"
    ])

    # 初始化电池的初始 SOC (State of Charge)
    battery_soc = {"Battery1": 50.0, "Battery2": 50.0, "Battery3": 50.0}  # 初始 SOC 为 50%

    # 设置电池最大充放电功率限制
    max_discharge_power = 50.0  # 最大放电功率 50 kW
    max_charge_power = 50.0     # 最大充电功率 50 kW

    # 运行每小时仿真
    for hour in range(24):
        # 设置负载和 PV 数据
        for config in pv_bess_config:
            pv_name = config["pv_name"]
            bess_name = config["bess_name"]
            kVA_rated = config["kVA_rated"]

            # 读取 PV 辐照度数据
            irradiance = pv_data.loc[hour, pv_name]

            # 计算 PV 发电功率
            pv_power_kW = irradiance * kVA_rated

            # 设置 PV 发电
            dss.text(f"Edit PVSystem.{pv_name} irradiance={irradiance}")
            dss.text(f"Edit PVSystem.{pv_name} kVA={kVA_rated}")

            # 设置用户负载
            load_kw = load_data.loc[hour, "Load (kW)"]
            dss.text(f"Edit Load.Load1 kW={load_kw}")

            # 计算净功率
            net_power = pv_power_kW - load_kw

            # 默认电池状态为 IDLE
            battery_state = "IDLE"
            battery_power = 0.0
            grid_support = 0.0

            if net_power > 0:
                # PV 发电过剩，电池充电
                charge_power = min(net_power, max_charge_power)
                battery_soc[bess_name] = min(battery_soc[bess_name] + charge_power * 0.1, 100.0)
                battery_state = "CHARGING"
                battery_power = charge_power
            elif net_power < 0 and battery_soc[bess_name] > 10.0:
                # PV 发电不足，电池放电
                discharge_power = min(abs(net_power), max_discharge_power)
                battery_soc[bess_name] = max(battery_soc[bess_name] - discharge_power * 0.1, 0.0)
                battery_state = "DISCHARGING"
                battery_power = discharge_power
            else:
                # 电池 SOC 过低时由电网支持
                grid_support = abs(net_power)

            # 设置电池状态和功率
            dss.text(f"Edit Storage.{bess_name} state={battery_state} kW={battery_power}")
            dss.solution.solve()  # 更新电网状态

            # 获取总功率
            total_power = dss.circuit.total_power

            # 获取 PV 和电池的实时功率
            dss.circuit.set_active_element(f"PVSystem.{pv_name}")
            pv_power = dss.cktelement.powers

            dss.circuit.set_active_element(f"Storage.{bess_name}")
            bess_power = dss.cktelement.powers
            soc = battery_soc[bess_name]

            # 打印仿真信息，包括用户负载需求和电网支持
            print(
                f"Hour {hour}: {pv_name} Generation = {pv_power_kW:.2f} kW, Irradiance = {irradiance:.4f}, Load Demand = {load_kw:.2f} kW, Grid Support = {grid_support:.2f} kW")
            if battery_state == "DISCHARGING":
                print(f"Hour {hour}: {bess_name} is discharging {battery_power:.2f} kW, SOC = {soc:.2f}%")
            elif battery_state == "CHARGING":
                print(f"Hour {hour}: {bess_name} is charging {battery_power:.2f} kW, SOC = {soc:.2f}%")
            else:
                print(f"Hour {hour}: {bess_name} is idle, SOC = {soc:.2f}%")

            # 写入 CSV 文件
            writer.writerow([
                hour, pv_name, pv_power_kW, pv_power[1],
                bess_name, bess_power[0], bess_power[1], soc,
                load_kw, grid_support, total_power[0], total_power[1]
            ])

print(f"Simulation results saved to {output_file}")

