import py_dss_interface
import os
import pathlib
import numpy as np

script_path = os.path.dirname(os.path.abspath(__file__))

dss_file = os.path.join(script_path, "../feeders/13bus/IEEE13Nodeckt.dss")

dss = py_dss_interface.DSS()
dss.text(f"compile [{dss_file}]")
dss.text("solve")

bus_kv_dict = dict()

for bus in dss.circuit.buses_names:
    dss.circuit.set_active_bus(bus)
    bus_kv_dict[bus] = dss.bus.kv_base * np.sqrt(3)

print(bus_kv_dict)