import json
import numpy
import matplotlib.pyplot as plt
import os
plt.style.use(os.path.join(os.path.dirname(__file__), "../science.mplstyle"))

modelnames = []

fc = 30
ac = 6
wc = 0.6
c = 300

q1 = 1.0E-12 * 126.77 * fc ** (-0.5)
q2 = 1.0E-12 * 185.4 * c ** 0.5 * fc ** (-0.9)
q3 = 0.29 * (wc) ** 4 * q2  # Pa^{-1}
q4 = 1.0E-12 * 20.3 * (ac) ** (-0.7)  # Pa^{-1}

m = 0.5
n = 0.1


def Q(t, t0):
    r = 1.7 * t0 ** 0.12 + 8
    Z = t0 ** (-m) * numpy.log(1 + (t - t0) ** n)
    Q_f = (0.086 * t0 ** (2 / 9) + 1.21 * t0**(4 / 9)) ** -1

    return Q_f * (1.0 + (Q_f / Z) ** r) ** (-1 / r)


def J(t, t0):
    return q1 + q2 * Q(t, t0) + q3 * numpy.log(1.0 + (t - t0)**n) + q4 * numpy.log(t / t0)


with open("disp.log", "r") as file:
    log_3d = json.load(file)

disp_3d = - numpy.array(log_3d["disp"]) * 1e+3

times = numpy.array(log_3d["times"])
print(times[1:])
L = 5.0
xdim = 0.3
zdim = 0.3
I = xdim * (zdim ** 3) / 12  # noqa
E = 1.0 / J(times, 28.0)

q = 2400 * 9.81 * xdim * zdim
disp = q * L ** 4 / (8 * E * I) * 1e+3

plt.figure(figsize=(6, 3))
plt.plot(times - 28, disp, linestyle="dashed", marker="x", label="full B3 1D, Euler-Bernoulli")
plt.plot(times - 28, disp_3d, linestyle="solid", marker="|", label="MPS 3D")


plt.legend()
plt.xlabel("Time [day]")
plt.ylabel("Displacement [mm]")
plt.xscale("log")
plt.tight_layout()
plt.savefig("1d_vs_3d.pdf")
