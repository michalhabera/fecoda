import numpy
import numba
import time
import numba.core.typing.cffi_utils as cffi_support

import dolfiny.interpolation
import fecoda.misc
import ufl
import dolfinx
import dolfinx.io
from mpi4py import MPI
from cffi import FFI


N = 50
mesh = dolfinx.generation.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
deg = 0
V_el = ufl.TensorElement("DG", mesh.ufl_cell(), deg)
L_el = ufl.VectorElement("DG", mesh.ufl_cell(), deg, dim=3)
V = dolfinx.FunctionSpace(mesh, V_el)
L = dolfinx.FunctionSpace(mesh, L_el)

num_cells = mesh.topology.index_map(3).size_global
print(f"Num cells {num_cells}")

A = dolfinx.Function(V)
lam = dolfinx.Function(L)
lam_hand = dolfinx.Function(L)


def expr(x):
    values = numpy.zeros((9, x.shape[1]))
    values[0] = 1.0
    values[1] = 2.0
    values[2] = 1.0
    values[3] = 2.0
    values[4] = 1.0
    values[6] = 1.0
    values[8] = 1.0

    return values


A.interpolate(expr)
l_expr = ufl.as_vector(fecoda.misc.eig(A))

cexpr = dolfiny.interpolation.CompiledExpression(l_expr, L_el)

# Cold run
dolfiny.interpolation.interpolate_cached(cexpr, lam)

ffcx_times = []
repeats = 5

# Hot runs
for i in range(repeats):
    t0 = time.time()
    dolfiny.interpolation.interpolate_cached(cexpr, lam)
    tok = time.time() - t0
    print(f"Time {tok}")
    ffcx_times.append(tok)

ffibuilder = FFI()
ffibuilder.set_source("_cffi_kernelA", r"""
#include <math.h>
#include <stdalign.h>

void tabulate_expression(double* restrict A,
                         const double* w,
                         const double* c,
                         const double* restrict coordinate_dofs)
{
    double eps = 1e-12;
    double q = (w[0] + w[4] + w[8]) / 3.0;
    double p1 = 0.5*(pow(w[1], 2) + pow(w[2], 2) + pow(w[5], 2) + pow(w[3], 2) + pow(w[6], 2) + pow(w[7], 2));
    double p2 = pow(w[0] - q, 2) + pow(w[4] - q, 2) + pow(w[8] - q, 2) + 2*p1;
    double p = sqrt(p2 / 6.0);

    double B[9];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double qf = 0.0;
            if (i == j)
                qf = q;
            B[i*3 + j] = (w[i*3 + j] - qf) / p;
        }
    }

    // det(B)
    double r = (B[0]*B[4]*B[8] - B[0]*B[5]*B[7] - B[1]*B[3]*B[8]
                + B[1]*B[5]*B[6] + B[2]*B[3]*B[7] - B[2]*B[4]*B[6]) / 2.0;
    r = fmax(fmin(r, 1.0 - eps), -1.0 + eps);
    double phi = acos(r) / 3.0;
    double l0, l1, l2 = 0.0;
    if (p2 < eps)
    {
        l0 = q;
        l1 = q;
        l2 = q;
    }
    else
    {
        l0 = q + 2*p*cos(phi);
        l2 = q + 2*p*cos(phi + 2.0/3.0 * M_PI);
        l1 = 3*q - l0 - l2;
    }
    A[0] = l0;
    A[1] = l1;
    A[2] = l2;
}
""", extra_compile_args=["-march=skylake", "-Ofast"])
ffibuilder.cdef("""
void tabulate_expression(double* restrict A,
                         const double* w,
                         const double* c,
                         const double* restrict coordinate_dofs);
""")

ffibuilder.compile(verbose=False)

from _cffi_kernelA import ffi, lib  # noqa
import _cffi_kernelA  # noqa

cffi_support.register_module(_cffi_kernelA)


@numba.njit
def run_pure(kernel):
    warr = numpy.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    Aarr = numpy.zeros(3, dtype=numpy.double)
    geometry = numpy.zeros((3, 2))
    constants = numpy.zeros(1, dtype=numpy.double)
    for i in range(num_cells):
        kernel(ffi.from_buffer(Aarr), ffi.from_buffer(warr), ffi.from_buffer(constants), ffi.from_buffer(geometry))


run_pure(cexpr.module.tabulate_expression)

pure_ffcx_times = []

for i in range(repeats):
    t0 = time.time()
    run_pure(cexpr.module.tabulate_expression)
    tok = time.time() - t0
    print(f"Pure loop FFCx kernel {tok}")
    pure_ffcx_times.append(tok)

hand_kernel = ffi.cast("void(*)(double *, double *, double *, double *)", ffi.addressof(lib, "tabulate_expression"))

pure_hand_times = []
for i in range(repeats):
    t0 = time.time()
    run_pure(hand_kernel)
    tok = time.time() - t0
    print(f"Pure loop hand kernel {tok}")
    pure_hand_times.append(tok)

cexpr.module.tabulate_expression = hand_kernel

hand_times = []
for i in range(repeats):
    t0 = time.time()
    dolfiny.interpolation.interpolate_cached(cexpr, lam_hand)
    tok = time.time() - t0
    print(f"Time hand-C {tok}")
    hand_times.append(tok)

assert numpy.isclose((lam.vector - lam_hand.vector).norm(), 0.0)

print(f"Pure loop FFCx {numpy.average(pure_ffcx_times):1.5f}, std: {numpy.std(pure_ffcx_times):1.5f}")
print(f"Pure loop hand C {numpy.average(pure_hand_times):1.5f}, std: {numpy.std(pure_hand_times):1.5f}")
print(f"FFCx {numpy.average(ffcx_times):1.5f}, std: {numpy.std(ffcx_times):1.5f}")
print(f"Hand C {numpy.average(hand_times):1.5f}, std: {numpy.std(hand_times):1.5f}")
