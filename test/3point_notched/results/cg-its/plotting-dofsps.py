import numpy
import re
import matplotlib.pyplot as plt
import os

plt.style.use(os.path.join(os.path.dirname(__file__), "../../../science.mplstyle"))

plt.figure(figsize=(5, 3))

p = re.compile("A0 solve dofs/s: ([0-9]*)")
tot = re.compile("total: ([0-9]*)")

filenames = ["64-44k.log", "64-257k.log", "64-790k.log"]
labels = ["$\\SI{44e+3}{}$ DoF", "$\\SI{257e+3}{}$ DoF", "$\\SI{790e+3}{}$ DoF"]
markers = ["x", "o", "s"]

for i in range(len(filenames)):
    with open(filenames[i], "r") as file:
        log = file.readlines()

    its0 = []
    its = 0
    for line in log:
        if "A0 solve dofs/s" in line:
            its += int(p.search(line).group(1))
        if ", total:" in line:
            total = int(tot.search(line).group(1))
            its /= total
            its0 += [its]
            its = 0

    plt.plot(numpy.array(its0) / 1e+6, linestyle="none", marker=markers[i], label=labels[i])

plt.legend()
plt.xlabel("Load step")
plt.ylabel("Avg. matrix solver performance [DoF/s]")
plt.tight_layout()
plt.savefig("malvar-dofsps.pdf")
