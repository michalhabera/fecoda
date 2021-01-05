"""A benchmark of a 2-point flexure test for a beam

@Book{gilbert2004an,
 author = {Gilbert, R. I.},
 title = {An experimental study of flexural cracking in reinforced concrete members under sustained loads},
 publisher = {University of New South Wales, School of Civil and Environmental Engineering},
 year = {2004},
 address = {Sydney, Australia},
 isbn = {0858414023}
}

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
from dolfinx import Constant, DirichletBC, Function, FunctionSpace, Form
from dolfinx.fem import locate_dofs_topological, assemble_scalar, assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import MeshTags, locate_entities_boundary
from dolfinx.geometry import BoundingBoxTree
from dolfinx import cpp
import fecoda.water
import fecoda.mps
import fecoda.mech
import fecoda.main
from mpi4py import MPI
import matplotlib.pyplot as plt
from petsc4py import PETSc

logger = logging.getLogger("fecoda")

rank = MPI.COMM_WORLD.rank

parser = argparse.ArgumentParser(parents=[fecoda.mps.parser, fecoda.mech.parser,
                                          fecoda.main.parser, fecoda.damage_rankine.parser])
parser.add_argument("--out")
parser.add_argument("--force", type=float, help="Loading force at the top in kN")

parser.add_argument("--moulding_steps", type=int, default=3)
parser.add_argument("--wetting_steps", type=int, default=3)
parser.add_argument("--afterload_steps", type=int, default=10)
parser.add_argument("--afterload_end", type=int, default=50)
parser.add_argument("--endtime_steps", type=int, default=10)
parser.add_argument("--loading_steps", type=int, default=15)

parser.add_argument("--mesh")
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

ydim = 0.348
xdim = 0.25
zdim = 3.5
lineload1 = zdim / 3
lineload2 = 2.0 * lineload1
linewidth = 0.05


def line_load(x):
    """Line load around z=lineload1 mm and z=lineload2 mm"""
    return np.logical_and(
        np.logical_or(
            np.logical_and(
                np.less_equal(x[2], lineload1 + linewidth),
                np.greater_equal(x[2], lineload1 - linewidth)),
            np.logical_and(
                np.less_equal(x[2], lineload2 + linewidth),
                np.greater_equal(x[2], lineload2 - linewidth))),
        np.isclose(x[1], ydim))


def top_load(x):
    return np.logical_and(np.isclose(x[1], ydim),
                          np.logical_and(np.less_equal(x[2], zdim * 2 / 3), np.greater_equal(x[2], zdim / 3)))


def bottom_left_corner(x):
    return np.logical_and(
        np.isclose(x[1], 0.0),
        np.isclose(x[2], zdim - 0.1))


def bottom_right_corner(x):
    return np.logical_and(
        np.isclose(x[1], 0.0),
        np.isclose(x[2], 0.1))


line_load_facets = locate_entities_boundary(mesh, 2, line_load)
bottom_left_lines = locate_entities_boundary(mesh, 1, bottom_left_corner)
bottom_right_lines = locate_entities_boundary(mesh, 1, bottom_right_corner)
top_load_facets = locate_entities_boundary(mesh, 2, top_load)

mt_line_load = MeshTags(mesh, 2, line_load_facets, 1)

# External boundary facets
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt_line_load)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mt_cell)

load_area = mesh.mpi_comm().allreduce(assemble_scalar(1.0 * ds(1, domain=mesh)), MPI.SUM)

if rank == 0:
    logger.info(f"[Timer] Mesh reading {time() - t0}")
    logger.info(f"Mesh hmin={mesh.hmin()} hmax={mesh.hmax()}")

w0, w1, intern_var0, intern_var1 = fecoda.main.initialize_functions(mesh)

# Beginning time for simulation
t = 0.01  # [day]

with intern_var0["eta_dash"].vector.localForm() as local:
    local.set(t / fecoda.mps.MPS_q4 * 1.0e-6)

with w0["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp - 5)
with w1["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp - 5)

with w0["phi"].vector.localForm() as local:
    local.set(0.95)
with w1["phi"].vector.localForm() as local:
    local.set(0.95)

co2_bc = Function(w0["co2"].function_space)
phi_bc = Function(w0["phi"].function_space)
displ_bc = Function(w0["displ"].function_space)
temp_bc = Function(w0["temp"].function_space)

moulding_end = 3

loading_time = 1.0
moulding_times = np.linspace(t, moulding_end, args.moulding_steps, endpoint=False)
wetting_times = np.linspace(moulding_end, fecoda.water.t_drying_onset, args.wetting_steps, endpoint=False)
loading_times = np.linspace(fecoda.water.t_drying_onset, fecoda.water.t_drying_onset
                            + loading_time, args.loading_steps, endpoint=True)
afterload_times = fecoda.water.t_drying_onset + loading_time + \
    np.logspace(np.log10(fecoda.mps.t_begin), np.log10(
        args.afterload_end - fecoda.water.t_drying_onset), args.afterload_steps, endpoint=False)

endtime_end = args.afterload_end + args.endtime_steps * (args.afterload_end - afterload_times[-1])

endtime_times = np.linspace(args.afterload_end, endtime_end, args.endtime_steps)

times = np.hstack((moulding_times, wetting_times, loading_times, afterload_times, endtime_times))
dtimes = np.diff(times)

if rank == 0:
    logger.info(f"Simulation times: {times}")
    plt.plot(times, range(len(times)), marker="o")
    plt.savefig("simulation_times.pdf")

W0sub1c = w0["displ"].function_space.sub(1).collapse()
W0sub2c = w0["displ"].function_space.sub(2).collapse()
displ_bc_y = Function(W0sub1c)
displ_bc_z = Function(W0sub2c)

displ_bc_top = Function(W0sub1c)

dt = Constant(mesh, 0.0)
t = Constant(mesh, 0.0)

bottom = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[1], 0.0))
bottomW0 = locate_dofs_topological(w0["displ"].function_space, 2, bottom)

top_dofsW0 = locate_dofs_topological((w0["displ"].function_space.sub(1), W0sub1c), 2, top_load_facets)

boundaryf = locate_entities_boundary(mesh, 2, lambda x: [True] * x.shape[1])
boundaryf_dofsW1 = locate_dofs_topological(w0["temp"].function_space, 2, boundaryf)
boundaryf_dofsW2 = locate_dofs_topological(w0["phi"].function_space, 2, boundaryf)
boundaryf_dofsW3 = locate_dofs_topological(w0["co2"].function_space, 2, boundaryf)

left = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], 0.0))
leftW2 = locate_dofs_topological(w0["phi"].function_space, 2, left)
leftW1 = locate_dofs_topological(w0["temp"].function_space, 2, left)

mesh.topology.create_connectivity_all()

leftW0 = locate_dofs_topological(w0["displ"].function_space, 1, bottom_left_lines)
rightW0sub1 = locate_dofs_topological((w0["displ"].function_space.sub(1), W0sub1c), 1, bottom_right_lines)

comm = MPI.COMM_WORLD
filename = f"{args.mesh}_{args.out}"
with XDMFFile(comm, f"{filename}.xdmf", "w") as ofile:
    ofile.write_mesh(mesh)

ksp = [None] * 4
log = {"disp_midspan": [], "times": [], "force": []}

force_max = 2.0 * args.force * -1.e-3 / load_area
dforce0 = force_max / args.loading_steps
force = Constant(mesh, 0.0)
dforce = Constant(mesh, 0.0)

f = {ds(1): (force + dforce) * ufl.as_vector((0, 1.0, 0))}

gravity_force = Constant(mesh, 1.0)
g = {
    dx(10): gravity_force * 1.0E-6 * fecoda.mech.rho_c * ufl.as_vector((0, 9.81, 0)),
    dx(1): gravity_force * 1.0E-6 * fecoda.mech.rho_rebar * ufl.as_vector((0, 9.81, 0)),
    dx(2): gravity_force * 1.0E-6 * fecoda.mech.rho_rebar * ufl.as_vector((0, 9.81, 0))
}
water_flux = {}
co2_flux = {}
heat_flux = {}


def loading_bc(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = - 0.001
    return values


J, F, expr_compiled =\
    fecoda.main.compile_forms(intern_var0, intern_var1, w0, w1, f, g,
                              heat_flux, water_flux, co2_flux, t, dt,
                              [dx(1), dx(2)], [dx(10)], randomize=5.e-2)

DMG = FunctionSpace(mesh, ("DG", 0))
dmg0 = Function(DMG, name="dmg")

# Define variational problem for projection
proj_rhs = Form(ufl.inner(intern_var0["dmg"], ufl.TestFunction(DMG)) * dx)
mass_DMG = assemble_matrix(ufl.inner(ufl.TrialFunction(DMG), ufl.TestFunction(DMG)) * dx, [])
mass_DMG.assemble()

dscale = 1.0
apply_load = True

t0 = time()

# Global time loop
for k in range(len(dtimes)):

    t.value = times[k]
    dt.value = dtimes[k]

    # If time is greater than validity of Kelvin chain
    # sampling last time
    if t.value > 0.5 * fecoda.mps.tau[-1]:
        if rank == 0:
            logger.warning("Validity of Kelvin sampling reached in {} days".format(t.value))
        break

    bcs_displ = []
    bcs_temp = []
    bcs_hum = []
    bcs_co2 = []

    # Moulding phase
    if t.value in moulding_times:
        if rank == 0:
            logger.info("Moulding phase")
        bcs_displ.append(DirichletBC(displ_bc, bottomW0))
    else:
        bcs_displ.append(DirichletBC(displ_bc, leftW0))
        bcs_displ.append(DirichletBC(displ_bc_y, rightW0sub1, w0["displ"].function_space.sub(1)))

    # Wetting phase
    if t.value in wetting_times and apply_load:
        if rank == 0:
            logger.info("Wetting phase")
        with temp_bc.vector.localForm() as local:
            local.set(-5.0 / args.wetting_steps)
        with co2_bc.vector.localForm() as local:
            local.set(0.6 / args.wetting_steps)
    else:
        with temp_bc.vector.localForm() as local:
            local.set(0.0)
        with co2_bc.vector.localForm() as local:
            local.set(0.0)

    if t.value in loading_times and apply_load:
        if rank == 0:
            logger.info("Loading phase")
        with phi_bc.vector.localForm() as local:
            local.set(-0.2 / args.loading_steps)
    else:
        with phi_bc.vector.localForm() as local:
            local.set(0.0)

    bcs_temp.append(DirichletBC(temp_bc, boundaryf_dofsW1))
    bcs_hum.append(DirichletBC(phi_bc, boundaryf_dofsW2))
    bcs_co2.append(DirichletBC(co2_bc, boundaryf_dofsW3))

    # Loading phase
    if t.value in loading_times and apply_load:
        dforce.value = dforce0
    else:
        dforce.value = 0.0

    def io_callback(iv1, w1, t):
        # Project damage into visualizable space
        b = assemble_vector(proj_rhs)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        proj_ksp = PETSc.KSP()
        proj_ksp.create(MPI.COMM_WORLD)
        proj_ksp.setType("cg")
        proj_ksp.getPC().setType("jacobi")
        proj_ksp.setOperators(mass_DMG)
        proj_ksp.setFromOptions()
        proj_ksp.solve(b, dmg0.vector)

        with XDMFFile(MPI.COMM_WORLD, f"{filename}.xdmf", "a") as ofile:
            ofile.write_function(w1["displ"], t)
            ofile.write_function(w1["temp"], t)
            ofile.write_function(w1["phi"], t)
            ofile.write_function(w1["co2"], t)
            ofile.write_function(dmg0, t)

    bcs_thc = bcs_temp + bcs_hum + bcs_co2

    try:
        dscale = fecoda.main.solve_fullstep(J, F, intern_var0, intern_var1, expr_compiled,
                                            w0, w1, bcs_displ, bcs_thc, force, dforce, dt, t, k, dscale,
                                            io_callback=io_callback, rtol=1.e-3, max_its=20)
    except fecoda.main.ConvergenceError:
        if rank == 0:
            logger.info(79 * "!")
            logger.info(f"Stopping at applied force: {force.value}")
        apply_load = False

    # Evaluate displacement at midspan bottom
    bb_tree = BoundingBoxTree(mesh, 3)
    p = np.array([0.0, 0.0, zdim / 2], dtype=np.float64)
    cell_candidates = cpp.geometry.compute_collisions_point(bb_tree, p)
    cell = cpp.geometry.select_colliding_cells(mesh, cell_candidates, p, 1)

    if len(cell) > 0:
        value = w1["displ"].eval(p, cell)[1]
    else:
        value = None
    values = comm.gather(value, root=0)

    if rank == 0:
        value = [x for x in values if x is not None][0]
        log["disp_midspan"] += [value]
        log["times"] += [float(t.value)]
        log["force"] += [float(force.value)]

        # Flush log file
        with open(f"{meshname}.log", "w") as file:
            json.dump(log, file)

if rank == 0:
    logger.info(f"Simulation finished: {time() - t0}")
