import json
import numpy
import matplotlib.pyplot as plt
import argparse
import os
plt.style.use(os.path.join(os.path.dirname(__file__), "../science.mplstyle"))


parser = argparse.ArgumentParser()
parser.add_argument("--files", nargs="+", required=True)
parser.add_argument("--data")
parser.add_argument("--out", default="")
parser.add_argument("--legend", nargs="+", required=True)
args = parser.parse_args()

legend = []
modelnames = []

plt.figure(figsize=(5, 3))

colors = ["black", "blue", "red", "grey"]
markers = ["s", "o", "x", "^"]

for i, filename in enumerate(args.files):
    with open(filename, "r") as file:
        log = json.load(file)

    plt.plot(numpy.array(log["times"])[1:], log["compl"][1:], marker="|",
             linestyle="solid", color=colors[i], label=args.legend[i])
    modelnames.append(filename)

    count = filename.split(".")[0].split("_")[-1]
    lee = numpy.loadtxt(f"{args.data}_{count}.csv", delimiter="\t")
    plt.plot(lee[:, 0], lee[:, 1], marker=markers[i], linestyle="", color=colors[i])

plt.legend()
plt.xlabel("Time [days]")
plt.ylabel(r"$J [\SI{E-6}{\mega \pascal^{-1}}]$")
plt.xlim(left=1.0e-3)
plt.xscale("log")
plt.tight_layout()
plt.savefig(f"{args.data}_{args.out}.pdf")
