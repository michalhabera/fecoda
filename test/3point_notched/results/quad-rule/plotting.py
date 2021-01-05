import json
import numpy
import matplotlib.pyplot as plt
import os
plt.style.use(os.path.join(os.path.dirname(__file__), "../../../science.mplstyle"))

plt.figure(figsize=(5, 3))
linestyles = ["dashed", "solid", "dotted", "dashdot"]

filenames = ["beam_malvar-s2-44229-quad2.log",
             "beam_malvar-s2-44229-quad6.log",
             "beam_malvar-s2-44229-quad8.log",
             "beam_malvar-s2-44229-quad12.log"]

legends = ["4 pts, Zienkiewicz, Taylor",
           "24 pts, Keast",
           "125 pts, Gauss-Jacobi",
           "343 pts, Gauss-Jacobi"]

for i, filename in enumerate(filenames):
    with open(filename, "r") as file:
        log = json.load(file)

    force = -numpy.array(log["force"])

    # Convert to [Newton]
    force *= 1.0e+6
    displ = numpy.array(log["displ"]) * 1e+3

    plt.plot(displ, force, linestyle=linestyles[i], label=legends[i])

plt.legend(loc="upper right")
plt.xlabel("Displacement [mm]")
plt.ylabel("Force [N]")
plt.tight_layout()
plt.savefig("quadrule-strain-stress.pdf")
