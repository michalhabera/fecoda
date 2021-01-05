import logging
import sys
from time import time

import argparse
import numpy as np
import json

import petsc4py
petsc4py.init(sys.argv)

import ufl
from dolfinx import Constant, DirichletBC, Function
from dolfinx.fem import locate_dofs_topological, assemble_scalar
from dolfinx.io import XDMFFile
from dolfinx.mesh import MeshTags, locate_entities_boundary
from dolfinx.generation import BoxMesh
from dolfinx.geometry import BoundingBoxTree, compute_collisions_point, select_colliding_cells
import fecoda.water
import fecoda.mps
import fecoda.mech
import fecoda.main
from mpi4py import MPI
import matplotlib.pyplot as plt

logger = logging.getLogger("fecoda")
t_init = time()

rank = MPI.COMM_WORLD.rank

parser = argparse.ArgumentParser(parents=[fecoda.mps.parser, fecoda.mech.parser,
                                          fecoda.main.parser, fecoda.damage_rankine.parser])
args = parser.parse_known_args()[0]

ydim = 5.0
xdim = 0.3
zdim = 0.3

Nx = 3
Nz = 3
Ny = 20


mesh = BoxMesh(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([xdim, ydim, zdim])],
               [Nx, Ny, Nz])


def right_side(x):
    return np.isclose(x[1], ydim)


def left_side(x):
    return np.isclose(x[1], 0.0)


left_side_facets = locate_entities_boundary(mesh, 2, left_side)
right_side_facets = locate_entities_boundary(mesh, 2, right_side)

mt = MeshTags(mesh, 2, right_side_facets, 1)

# External boundary facets
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
dx = ufl.Measure("dx", domain=mesh)

load_area = mesh.mpi_comm().allreduce(assemble_scalar(1.0 * ds(1, domain=mesh)), MPI.SUM)

if rank == 0:
    logger.info("Mesh hmin={} hmax={}".format(mesh.hmin(), mesh.hmax()))
    logger.info(f"Load area = {load_area}")

assert np.isclose(xdim * zdim, load_area)

w0, w1, intern_var0, intern_var1 = fecoda.main.initialize_functions(mesh)

# Beginning time for simulation
t0 = 28.0  # [day]

# Simulation ends
t1 = 365 * 10

with intern_var0["eta_dash"].vector.localForm() as local:
    local.set(t0 / fecoda.mps.MPS_q4 * 1.0e-6)

with w0["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp)
with w1["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp)

with w0["phi"].vector.localForm() as local:
    local.set(0.9999)
with w1["phi"].vector.localForm() as local:
    local.set(0.9999)

co2_bc = Function(w0["co2"].function_space)
phi_bc = Function(w0["phi"].function_space)
displ_bc = Function(w0["displ"].function_space)
temp_bc = Function(w0["temp"].function_space)

times = t0 + np.logspace(np.log10(fecoda.mps.t_begin), np.log10(t1 - t0), 50)
times = np.hstack(([t0, ], times))
dtimes = np.diff(times)

if rank == 0:
    logger.info("Simulation times: {}".format(times))
    plt.plot(times, range(len(times)), marker="o")
    plt.savefig("simulation_times.pdf")

dt = Constant(mesh, 0.0)
t = Constant(mesh, 0.0)

mesh.topology.create_connectivity_all()

leftW0 = locate_dofs_topological(w0["displ"].function_space, 2, left_side_facets)

comm = MPI.COMM_WORLD
filename = "cantilever"
with XDMFFile(comm, f"{filename}.xdmf", "w") as ofile:
    ofile.write_mesh(mesh)

force = Constant(mesh, 0.0)
dforce = Constant(mesh, 0.0)

f = {}
body_force = Constant(mesh, 1.0)

g = {
    dx: body_force * 1.0E-6 * fecoda.mech.rho_c * ufl.as_vector((0, 0, 9.81)),
}

water_flux = {}
co2_flux = {}
heat_flux = {}

J, F, expr_compiled =\
    fecoda.main.compile_forms(intern_var0, intern_var1, w0, w1, f, g,
                              heat_flux, water_flux, co2_flux, t, dt,
                              [], [dx], damage_off=True)

log = {"times": [], "disp": []}

# Global time loop
for k in range(len(dtimes)):

    # If time is greater than validity of Kelvin chain
    # sampling last time
    if times[k] + dtimes[k] > 0.5 * fecoda.mps.tau[-1]:
        if rank == 0:
            logger.warning("Validity of Kelvin sampling reached in {} days".format(t.value))
        break

    t.value = times[k]
    dt.value = dtimes[k]

    if rank == 0:
        logger.info(79 * "#")
        logger.info(f"Time: {t.value} days.")
        logger.info(f"Dtime: {dt.value}")
        logger.info("Step: {}".format(k))
        logger.info(79 * "#")

    bcs_displ = []
    bcs_temp = []
    bcs_hum = []
    bcs_co2 = []

    bcs_displ.append(DirichletBC(displ_bc, leftW0))

    _scale = fecoda.main.solve_displ_system(J[0], F[0], intern_var0, intern_var1, expr_compiled,
                                            w0, w1, bcs_displ, dforce, dt, t, k)
    fecoda.main.post_update(expr_compiled, intern_var0, intern_var1, w0, w1)
    fecoda.main.copyout_state(w0, w1, intern_var0, intern_var1)

    with XDMFFile(comm, f"{filename}.xdmf", "a") as ofile:
        ofile.write_function(w1["displ"], t.value)

    # Evaluate displacement at midspan bottom
    bb_tree = BoundingBoxTree(mesh, 3)
    p = np.array([xdim / 2, ydim, 0.0], dtype=np.float64)
    cell_candidates = compute_collisions_point(bb_tree, p)
    cell = select_colliding_cells(mesh, cell_candidates, p, 1)

    if len(cell) > 0:
        value = w1["displ"].eval(p, cell)[2]
    else:
        value = None
    values = comm.gather(value, root=0)

    if rank == 0:
        disp = [x for x in values if x is not None][0]
        log["times"].append(float(t.value))
        log["disp"].append(disp)

        with open("disp.log", "w") as file:
            json.dump(log, file)

print(f"Simulation finished in {time() - t_init}")
