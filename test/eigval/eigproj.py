import dolfinx
import dolfiny.interpolation
import ufl
import numpy
import numba
from mpi4py import MPI
import fecoda.misc
from cffi import FFI
import time

N = 50
mesh = dolfinx.generation.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
deg = 0
V_el = ufl.TensorElement("DG", mesh.ufl_cell(), deg)
V = dolfinx.FunctionSpace(mesh, V_el)

P_el = ufl.TensorElement("DG", mesh.ufl_cell(), deg)
P = dolfinx.FunctionSpace(mesh, P_el)

num_cells = mesh.topology.index_map(3).size_global
print(f"Num cells {num_cells}")

A = dolfinx.Function(V)
proj = dolfinx.Function(P)

p0 = ufl.diff(fecoda.misc.eig(A)[0], A)
p1 = ufl.diff(fecoda.misc.eig(A)[1], A)
p2 = ufl.diff(fecoda.misc.eig(A)[2], A)

expr = fecoda.misc.eig(A)[0] * p0 + fecoda.misc.eig(A)[1] * p1 + fecoda.misc.eig(A)[2] * p2
cexpr = dolfiny.interpolation.CompiledExpression(expr, P_el)


for i in range(10):

    vals = numpy.random.rand(6)
    arr = numpy.array([[vals[0], vals[1], vals[2]],
                       [vals[1], vals[3], vals[4]],
                       [vals[2], vals[4], vals[5]]]).flatten()
    print(f"Testing on random matrix {i}")

    def expr2(x):
        values = numpy.zeros((9, x.shape[1]))
        for i in range(9):
            values[i] = arr[i]

        return values

    A.interpolate(expr2)
    dolfiny.interpolation.interpolate_cached(cexpr, proj)

    diff = (proj.vector - A.vector).norm()
    print(f"||A_spect - A|| = {diff}")

    assert numpy.isclose(diff, 0.0)
ffi = FFI()


@numba.njit
def run_pure(kernel):
    warr = arr
    Aarr = numpy.zeros(9, dtype=numpy.double)
    geometry = numpy.zeros((3, 2))
    constants = numpy.zeros(1, dtype=numpy.double)
    for i in range(num_cells):
        kernel(ffi.from_buffer(Aarr), ffi.from_buffer(warr), ffi.from_buffer(constants), ffi.from_buffer(geometry))


# Cold run
run_pure(cexpr.module.tabulate_expression)

pure_ffcx_times = []

for i in range(3):
    t0 = time.time()
    run_pure(cexpr.module.tabulate_expression)
    tok = time.time() - t0
    print(f"FFCx {tok}")
    pure_ffcx_times.append(tok)

print(f"FFCx avg: {numpy.average(pure_ffcx_times)}")
