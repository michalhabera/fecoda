import numpy
import re
import matplotlib.pyplot as plt
import os

plt.style.use(os.path.join(os.path.dirname(__file__), "../../../science.mplstyle"))

plt.figure(figsize=(5, 3))

p = re.compile("Simulation finished: ([0-9]*)")
times = []
processes = [1, 8, 16, 32, 64]

for proc in processes:
    with open(f"{proc}.log", "r") as file:
        log = file.readlines()

    times += [float(p.search(log[-1]).group(1))]

processes = numpy.array(processes)
plt.plot(processes, numpy.array(times), linestyle="solid", marker="s")

plt.ylim((0, 2000))
plt.xticks(processes, labels=processes)
plt.xlabel(r"\# processes")
plt.ylabel("Total simulation time [s]")
plt.tight_layout()
plt.savefig("malvar-total-weak.pdf")
