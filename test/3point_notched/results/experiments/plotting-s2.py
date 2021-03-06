import json
import numpy
import matplotlib.pyplot as plt
import os
plt.style.use(os.path.join(os.path.dirname(__file__), "../../../science.mplstyle"))

plt.figure(figsize=(5, 3))

malvar = numpy.loadtxt("malvar-s2.csv", delimiter=",")
malvar_stress = malvar[:, 1]
plt.plot(malvar[:, 0], malvar_stress, marker="o", linestyle="dashed", label="experiment, Malvar Series 2")

with open("beam_malvar-s2.log", "r") as file:
    log = json.load(file)

force = -numpy.array(log["force"])

# Convert to [Newton]
force *= 1.0e+6
displ = numpy.array(log["displ"]) * 1e+3

plt.plot(displ, force, linestyle="solid", label="simulation")

plt.legend()
plt.xlabel("Displacement [mm]")
plt.xlim(right=0.8)
plt.ylabel("Force [N]")
plt.tight_layout()
plt.savefig("s2-strain-stress.pdf")
