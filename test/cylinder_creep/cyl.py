import argparse
import os
import sys
from time import time
import logging
import json

import petsc4py
petsc4py.init(sys.argv)

import numpy
import ufl
from dolfinx.io import XDMFFile
from dolfinx import Function, Constant
from dolfinx.fem import locate_dofs_topological, DirichletBC
from dolfinx.geometry import BoundingBoxTree, compute_collisions_point, select_colliding_cells
from dolfiny.interpolation import interpolate
from mpi4py import MPI

import fecoda.mps
import fecoda.main

rank = MPI.COMM_WORLD.rank
logger = logging.getLogger("fecoda")

parser = argparse.ArgumentParser(parents=[fecoda.mps.parser, fecoda.mech.parser, fecoda.main.parser])
parser.add_argument("--mesh")
parser.add_argument("--sigma", type=float)
parser.add_argument("--out")
parser.add_argument("--steps", type=int)
parser.add_argument("--end", type=float)
parser.add_argument("--tp", type=float, help="Time at loading")
args = parser.parse_known_args()[0]

#
# Convert and read mesh and tags
#
meshname = args.mesh
filedir = os.path.dirname(__file__)

t0 = time()
infile = XDMFFile(MPI.COMM_WORLD,
                  os.path.join(filedir, "mesh/{}.xdmf".format(meshname)), "r")
mesh = infile.read_mesh(name="Grid")
mesh.topology.create_connectivity_all()
mt_cell = infile.read_meshtags(mesh, "Grid")
infile.close()

infile = XDMFFile(MPI.COMM_WORLD,
                  os.path.join(filedir, "mesh/{}_triangle.xdmf".format(meshname)), "r")
mt_facet = infile.read_meshtags(mesh, "Grid")
infile.close()

# External boundary facets
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt_facet, metadata={"quadrature_degree": 2})
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mt_cell, metadata={"quadrature_degree": 4})

w0, w1, intern_var0, intern_var1 = fecoda.main.initialize_functions(mesh)

with intern_var0["eta_dash"].vector.localForm() as local:
    local.set(args.tp / fecoda.mps.MPS_q4 * 1.0e-6)

with w0["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp)
with w1["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp)

with w0["phi"].vector.localForm() as local:
    local.set(0.95)
with w1["phi"].vector.localForm() as local:
    local.set(0.95)

displ_bc2 = Function(w0["displ"].function_space.sub(2).collapse())

t = 0.1
tp = args.tp

inital_times = numpy.linspace(t, tp, 5)
afterload_times = tp + numpy.logspace(numpy.log10(fecoda.mps.t_begin), numpy.log10(tp + args.end), args.steps)

times = numpy.hstack((inital_times, afterload_times))
dtimes = numpy.diff(times)

dt = Constant(mesh, 0.0)
t = Constant(mesh, 0.0)

bottom_facets = mt_facet.indices[numpy.where(mt_facet.values == 1)[0]]
bottom_dofs = locate_dofs_topological((w0["displ"].function_space.sub(2), displ_bc2.function_space), 2, bottom_facets)

comm = MPI.COMM_WORLD
filename = f"{meshname}_{args.out}"
with XDMFFile(comm, f"{filename}.xdmf", "w") as ofile:
    ofile.write_mesh(mesh)

ksp = [None] * 4
log = {"compl": [], "times": []}

force = Constant(mesh, 0.0)
df = Constant(mesh, 0.0)
f = {ds(2): (force + df) * ufl.as_vector((0, 0, -1.0))}

gravity_force = Constant(mesh, 1.0)
g = {
    dx(3): gravity_force * 1.0E-6 * fecoda.mech.rho_rebar * ufl.as_vector((0, 0, 9.81))
}
water_flux = {}
co2_flux = {}
heat_flux = {}

J, F, expr_compiled =\
    fecoda.main.compile_forms(intern_var0, intern_var1, w0, w1, f, g,
                              heat_flux, water_flux, co2_flux, t, dt,
                              [], [dx(3)])

strain = Function(intern_var0["eps_cr_kel"].function_space)

t0 = time()

# Global time loop
for k in range(len(dtimes)):

    t.value = times[k]
    dt.value = dtimes[k]

    if rank == 0:
        logger.info(79 * "#")
        logger.info("Time: {:2.2} days.".format(t.value))
        logger.info("Step: {}".format(k))
        logger.info(79 * "#")

    bcs_displ = [DirichletBC(displ_bc2, bottom_dofs, w0["displ"].function_space)]
    if t.value == afterload_times[0]:
        df.value = args.sigma
    else:
        df.value = 0.0

    fecoda.main.solve_displ_system(J[0], F[0], intern_var0, intern_var1, expr_compiled,
                                   w0, w1, bcs_displ, df, dt, t, k)

    fecoda.main.post_update(expr_compiled, intern_var0, intern_var1, w0, w1)
    fecoda.main.copyout_state(w0, w1, intern_var0, intern_var1)

    force.value += df.value

    bb_tree = BoundingBoxTree(mesh, 3)
    p = numpy.array([0.0, 0.0, 0.3], dtype=numpy.float64)
    cell_candidates = compute_collisions_point(bb_tree, p)
    cell = select_colliding_cells(mesh, cell_candidates, p, 1)

    interpolate(ufl.sym(ufl.grad(w1["displ"])), strain)

    if len(cell) > 0:
        value = strain.eval(p, cell)
        value = value[-1]
    else:
        value = None
    values = comm.gather(value, root=0)

    if rank == 0:
        value = [x for x in values if x is not None][0]
        compl = value * -1.0e+6 / args.sigma
        log["compl"].append(compl)
        log["times"].append(float(t.value) - tp - fecoda.mps.t_begin)

        # Flush log file
        with open(f"{filename}.log", "w") as file:
            json.dump(log, file)

    with XDMFFile(comm, f"{filename}.xdmf", "a") as ofile:
        ofile.write_function(w1["displ"], t.value)


if rank == 0:
    logger.info(79 * "-")
    logger.info(f"Simulation finished in {time() - t0}")
