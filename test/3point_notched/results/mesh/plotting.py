import json
import numpy
import matplotlib.pyplot as plt
import os
plt.style.use(os.path.join(os.path.dirname(__file__), "../../../science.mplstyle"))

plt.figure(figsize=(5, 3))
linestyles = ["dashed", "solid", "dotted", "dashdot"]

filenames = ["beam_malvar-s2-44229.log",
             "beam_malvar-s2-120552.log",
             "beam_malvar-s2-257979.log",
             "beam_malvar-s2-790884.log"]

legends = ["$\\SI{44e+3}{}$ DoF",
           "$\\SI{120e+3}{}$ DoF",
           "$\\SI{250e+3}{}$ DoF",
           "$\\SI{790e+3}{}$ DoF"]

for i, filename in enumerate(filenames):
    with open(filename, "r") as file:
        log = json.load(file)

    force = -numpy.array(log["force"])

    # Convert to [Newton]
    force *= 1.0e+6
    displ = numpy.array(log["displ"]) * 1e+3

    plt.plot(displ, force, linestyle=linestyles[i], label=legends[i])

plt.legend()
plt.xlabel("Displacement [mm]")
plt.ylabel("Force [N]")
plt.tight_layout()
plt.savefig("mesh-strain-stress.pdf")
