import json
import numpy
import matplotlib.pyplot as plt
import os
plt.style.use(os.path.join(os.path.dirname(__file__), "../science.mplstyle"))

modelnames = []

with open("../1d_beam_replaced/1d.log", "r") as file:
    log_1d = json.load(file)

with open("disp.log", "r") as file:
    log_3d = json.load(file)

disp_1d = numpy.array(log_1d["disp"]) * 1e+3
disp_3d = - numpy.array(log_3d["disp"]) * 1e+3

plt.figure(figsize=(6, 3))
plt.plot(log_1d["times"], disp_1d, linestyle="dashed", marker="x", label="short-form B3 1D")
plt.plot(log_3d["times"], disp_3d, linestyle="solid", marker="o", label="MPS 3D")

plt.legend()
plt.xlabel("Time [day]")
plt.ylabel("Displacement [mm]")
plt.tight_layout()
plt.savefig("1d_vs_3d.pdf")
