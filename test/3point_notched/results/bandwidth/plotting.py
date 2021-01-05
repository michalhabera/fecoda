import numpy
import matplotlib.pyplot as plt
import os
plt.style.use(os.path.join(os.path.dirname(__file__), "../../../science.mplstyle"))

plt.figure(figsize=(5, 3))

data = numpy.loadtxt("scaling.log", usecols=[0, 1])

plt.plot(data[::4, 0], data[::4, 1] / 1024, linestyle="solid", marker="o", label="2 x AMD Epyc 7742, DDR4")

plt.legend()
plt.xlabel(r"\# processes")
plt.ylabel(r"Bandwidth [GB/s]")
plt.ylim((0, 250))
plt.tight_layout()
plt.savefig("bandwidth.pdf")
