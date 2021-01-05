"""A benchmark of a 3-point flexure notched test for a beam

Implemented according to "Malvar, L. J., & Warren, G. E. (1988).
Fracture energy for three-point-bend tests on single-edge-notched beams.
Experimental Mechanics, 28(3), 266–272. doi:10.1007/bf02329022"

"""

import logging
import os
import sys
from time import time

import argparse
import numpy as np
import json

import petsc4py
petsc4py.init(sys.argv)

import ufl
from dolfinx import Constant, DirichletBC, Function, FunctionSpace
from dolfinx.fem import locate_dofs_topological, assemble_vector, assemble_matrix, Form
from dolfinx.io import XDMFFile
import fecoda.water
import fecoda.damage_rankine
import fecoda.mps
import fecoda.mech
import fecoda.main
from mpi4py import MPI
from petsc4py import PETSc

logger = logging.getLogger("fecoda")

rank = MPI.COMM_WORLD.rank
logger.info("Proc name: {}".format(MPI.Get_processor_name()))

parser = argparse.ArgumentParser(parents=[fecoda.mps.parser, fecoda.mech.parser,
                                          fecoda.main.parser, fecoda.damage_rankine.parser], add_help=False)
parser.add_argument("--out")
parser.add_argument("--mesh")
parser.add_argument("--steps", type=int)
parser.add_argument("--displ", type=float)
args, unknown = parser.parse_known_args()

#
# Read mesh and tags
#
filedir = os.path.dirname(__file__)

t0 = time()
infile = XDMFFile(MPI.COMM_WORLD,
                  os.path.join(filedir, "mesh/{}.xdmf".format(args.mesh)), "r")
mesh = infile.read_mesh(name="Grid")
mesh.topology.create_connectivity_all()
infile.close()

infile = XDMFFile(MPI.COMM_WORLD,
                  os.path.join(filedir, "mesh/{}_line.xdmf".format(args.mesh)), "r")
mt_line = infile.read_meshtags(mesh, name="Grid")
infile.close()

top_load_facets = mt_line.indices[np.where(mt_line.values == 4)[0]]

# External boundary facets
dx = ufl.Measure("dx", domain=mesh)

if rank == 0:
    logger.info(f"[Timer] Mesh reading: {time() - t0}")
    logger.info(f"Mesh hmin: {mesh.hmin()} hmax: {mesh.hmax()}")
    logger.info(
        f"Num cells global: {mesh.topology.index_map(3).size_global}, local: {mesh.topology.index_map(3).size_local}")

w0, w1, intern_var0, intern_var1 = fecoda.main.initialize_functions(mesh)

displ_bc_bottom = Function(w0["displ"].function_space)
displ_bc_bottom2 = Function(w0["displ"].function_space.sub(2).collapse())
displ_bc_top = Function(w0["displ"].function_space)
displ_bc_top2 = Function(w0["displ"].function_space.sub(2).collapse())


t = 14.0
tp = 28.0

with intern_var0["eta_dash"].vector.localForm() as local:
    local.set(tp / fecoda.mps.MPS_q4 * 1.0e-6)

with w0["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp)
with w1["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp)

with w0["phi"].vector.localForm() as local:
    local.set(0.95)
with w1["phi"].vector.localForm() as local:
    local.set(0.95)

inital_times = np.linspace(t, tp, 5, endpoint=False)
loading_times = np.linspace(tp, tp + 1.0, args.steps)

times = np.hstack((inital_times, loading_times))
dtimes = np.diff(times)

dt = Constant(mesh, 0.0)
t = Constant(mesh, 0.0)

bottom_left_lines = mt_line.indices[np.where(mt_line.values == 3)[0]]
bottom_left_dofs = locate_dofs_topological(w0["displ"].function_space, 1, bottom_left_lines)

bottom_right_lines = mt_line.indices[np.where(mt_line.values == 2)[0]]
bottom_right_dofs = locate_dofs_topological((w0["displ"].function_space.sub(2),
                                             w0["displ"].function_space.sub(2).collapse()), 1, bottom_right_lines)

top_load_dofs = locate_dofs_topological((w0["displ"].function_space.sub(2),
                                         w0["displ"].function_space.sub(2).collapse()), 1, top_load_facets)

comm = MPI.COMM_WORLD
filename = f"{args.mesh}_{args.out}"
with XDMFFile(comm, f"{filename}.xdmf", "w") as ofile:
    ofile.write_mesh(mesh)


ksp = [None] * 4
log = {"displ": [], "force": []}

h = Constant(mesh, mesh.hmin())
force = Constant(mesh, 0.0)
dforce = Constant(mesh, 0.0)
f = {}

# Gravity direction swapped here to match the experiments
gravity_force = Constant(mesh, -1.0)
g = {
    dx: gravity_force * 1.0E-6 * 9.81 * fecoda.mech.rho_c * ufl.as_vector((0, 0, 1.0))
}
water_flux = {}
co2_flux = {}
heat_flux = {}


def loading_bc(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = - args.displ
    return values


displ_top = 0.0

J, F, expr_compiled =\
    fecoda.main.compile_forms(intern_var0, intern_var1, w0, w1, f, g,
                              heat_flux, water_flux, co2_flux, t, dt,
                              [], [dx])

DMG = FunctionSpace(mesh, ("DG", 0))
dmg0 = Function(DMG, name="dmg")

# Define variational problem for projection
proj_rhs = Form(ufl.inner(intern_var0["dmg"], ufl.TestFunction(DMG)) * dx)
mass_DMG = assemble_matrix(ufl.inner(ufl.TrialFunction(DMG), ufl.TestFunction(DMG)) * dx, [])
mass_DMG.assemble()

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

    if t.value in loading_times:
        bcs_displ = [DirichletBC(displ_bc_bottom, bottom_left_dofs),
                     DirichletBC(displ_bc_bottom2, bottom_right_dofs, w0["displ"].function_space),
                     DirichletBC(displ_bc_top2, top_load_dofs, w0["displ"].function_space)]
        displ_bc_top2.interpolate(loading_bc)
    else:
        bcs_displ = [DirichletBC(displ_bc_bottom, bottom_left_dofs),
                     DirichletBC(displ_bc_bottom2, bottom_right_dofs, w0["displ"].function_space)]

    scale = fecoda.main.solve_displ_system(J[0], F[0], intern_var0, intern_var1, expr_compiled,
                                           w0, w1, bcs_displ, dforce, dt, t, k)

    fecoda.main.post_update(expr_compiled, intern_var0, intern_var1, w0, w1)

    resid = assemble_vector(F[0])
    resid.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    top_load_dofs_owned = top_load_dofs[0][top_load_dofs[0]
                                           < w0["displ"].function_space.dofmap.index_map_bs
                                           * w0["displ"].function_space.dofmap.index_map.size_local]
    top_force = MPI.COMM_WORLD.allreduce(np.sum(resid.array[top_load_dofs_owned]), MPI.SUM)

    fecoda.main.copyout_state(w0, w1, intern_var0, intern_var1)

    if t.value in loading_times:
        displ_top += scale * args.displ

    if rank == 0:
        log["force"].append(top_force)
        log["displ"].append(displ_top)

        # Flush log file
        with open(f"{filename}.log", "w") as file:
            json.dump(log, file)

    # Project damage into visualizable space
    b = assemble_vector(proj_rhs)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    proj_ksp = PETSc.KSP()
    proj_ksp.create(mesh.mpi_comm())

    proj_ksp.setType("cg")
    proj_ksp.getPC().setType("jacobi")

    proj_ksp.setOperators(mass_DMG)

    proj_ksp.setFromOptions()
    proj_ksp.solve(b, dmg0.vector)

    with XDMFFile(comm, f"{filename}.xdmf", "a") as ofile:
        ofile.write_function(w1["displ"], t.value)
        ofile.write_function(dmg0, t.value)
        ofile.write_function(intern_var0["sigma"], t.value)

if rank == 0:
    logger.info(f"Simulation finished: {time() - t0}")
