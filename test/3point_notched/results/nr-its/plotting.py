import re
import matplotlib.pyplot as plt
import os

plt.style.use(os.path.join(os.path.dirname(__file__), "../../../science.mplstyle"))

plt.figure(figsize=(5, 3))

p = re.compile("total: ([0-9]*)")

filenames = ["../cg-its/64-44k.log", "../cg-its/64-257k.log", "../cg-its/64-790k.log"]
labels = ["$\\SI{44e+3}{}$ DoF", "$\\SI{257e+3}{}$ DoF", "$\\SI{790e+3}{}$ DoF"]
markers = ["x", "o", "s"]

for i in range(len(filenames)):
    with open(filenames[i], "r") as file:
        log = file.readlines()

    its = []
    for line in log:
        if ", total:" in line:
            its += [int(p.search(line).group(1))]

    plt.plot(its, linestyle="dashed", marker=markers[i], label=labels[i])

plt.legend()
plt.xlabel("Load step")
plt.ylabel(r"\# total NR iterations")
plt.tight_layout()
plt.savefig("malvar-nr-its.pdf")
