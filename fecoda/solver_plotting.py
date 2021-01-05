import json
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
args = parser.parse_args()

with open(args.file, "r") as file:
    log = json.load(file)

plt.plot(log["times"], log["a0_its"], marker="x", linestyle="dotted")
plt.xlabel("Time [days]")
plt.ylabel("A00 CG iterations")
plt.tight_layout()
plt.savefig("a00_its.pdf")

plt.clf()
plt.plot(log["times"], log["a0_dofsps"], marker="o", linestyle="dotted")
plt.xlabel("Time [days]")
plt.ylabel("A00 performance [dofs/s]")
plt.tight_layout()
plt.savefig("a00_dofsps.pdf")
