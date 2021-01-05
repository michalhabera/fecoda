import logging
import os
from time import time

import argparse
import numpy as np
import json

import ufl
from dolfinx import Constant, DirichletBC, Function
from dolfinx.fem import locate_dofs_topological, assemble_scalar
from dolfinx.io import XDMFFile
from dolfinx.mesh import MeshTags, locate_entities_boundary, locate_entities
from dolfinx.geometry import BoundingBoxTree
from dolfinx import cpp
import dolfiny.interpolation
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
parser.add_argument("--out", default="results.xdmf")
parser.add_argument("--force", type=float, help="Loading force at the top in kN")

parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--loading_steps", type=int, default=10)

parser.add_argument("--mesh")
args = parser.parse_args()

meshname = args.mesh
filedir = os.path.dirname(__file__)

infile = XDMFFile(MPI.COMM_WORLD,
                  os.path.join(filedir, "mesh/{}.xdmf".format(meshname)), "r")
mesh = infile.read_mesh(name="Grid")
mesh.topology.create_connectivity_all()
mt_cell = infile.read_meshtags(mesh, "Grid")
infile.close()

ydim = 0.3
xdim = 0.2
zdim = 8.0

left_support_margin = 0.0

alpha = 0.25
beta = 0.40


def right_support(x):
    return np.logical_and(
        np.isclose(x[1], 0.0),
        np.isclose(x[2], zdim - beta * zdim))


def left_support(x):
    return np.logical_and(
        np.isclose(x[1], 0.0),
        np.isclose(x[2], left_support_margin))


def top_load(x):
    return np.isclose(x[1], ydim)


left_support_lines = locate_entities_boundary(mesh, 1, left_support)
right_support_lines = locate_entities_boundary(mesh, 1, right_support)
top_load_facets = locate_entities_boundary(mesh, 2, top_load)
replaced_part_cells = locate_entities(
    mesh, 3, lambda x: np.greater_equal(x[2], zdim - alpha * zdim))
replaced_part_interface = locate_entities(
    mesh, 2, lambda x: np.isclose(x[2], zdim - alpha * zdim))
right_side_facets = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], zdim))

mt_top_load = MeshTags(mesh, 2, top_load_facets, 1)

# External boundary facets
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt_top_load, metadata={"quadrature_degree": 2})
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mt_cell, metadata={"quadrature_degree": 6})

load_area = mesh.mpi_comm().allreduce(assemble_scalar(1.0 * ds(1, domain=mesh)), MPI.SUM)

if rank == 0:
    logger.info("Mesh hmin={} hmax={}".format(mesh.hmin(), mesh.hmax()))
    logger.info(f"Load area = {load_area}")

assert np.isclose(xdim * zdim, load_area)

w0, w1, intern_var0, intern_var1 = fecoda.main.initialize_functions(mesh)

replaced_part_dofs = locate_dofs_topological(intern_var1["gamma_0"].function_space, 3, replaced_part_cells)
replaced_part_dofs_W0 = locate_dofs_topological(w0["displ"].function_space, 3, replaced_part_cells)
replaced_part_interface_dofs_W0 = locate_dofs_topological(w0["displ"].function_space, 2, replaced_part_interface)
right_side_dofs_W0sub = locate_dofs_topological(w0["displ"].function_space.sub(2), 2, right_side_facets)

# Beginning time for simulation
t0 = 28.0  # [day]
# t0 = 365*10

# Time when replacement is done
t1 = 365.0 * 10

# Simulation ends
t2 = 365 * 25

with intern_var0["eta_dash"].vector.localForm() as local:
    local.set(t0 / fecoda.mps.MPS_q4 * 1.0e-6)

with w0["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp)
with w1["temp"].vector.localForm() as local:
    local.set(fecoda.misc.room_temp)

with w0["phi"].vector.localForm() as local:
    local.set(0.95)
with w1["phi"].vector.localForm() as local:
    local.set(0.95)

co2_bc = Function(w0["co2"].function_space)
phi_bc = Function(w0["phi"].function_space)
displ_bc = Function(w0["displ"].function_space)
temp_bc = Function(w0["temp"].function_space)

ta = 1.0
times = t0 + np.logspace(np.log10(ta), np.log10(t1 - t0 - 0.1), args.steps)
times_replaced = t1 + np.logspace(np.log10(ta), np.log10(t2 - t1), args.steps)
times = np.hstack(((t0, ), times, (t1, ), times_replaced))
dtimes = np.diff(times)

if rank == 0:
    logger.info("Simulation times: {}".format(times))
    plt.plot(times, range(len(times)), marker="o")
    plt.savefig("simulation_times.pdf")

W0sub1c = w0["displ"].function_space.sub(1).collapse()
W0sub2c = w0["displ"].function_space.sub(2).collapse()
displ_bc_y = Function(W0sub1c)
displ_bc_z = Function(W0sub2c)

dt = Constant(mesh, 0.0)
t = Constant(mesh, 0.0)

mesh.topology.create_connectivity_all()

leftW0 = locate_dofs_topological(w0["displ"].function_space, 1, left_support_lines)
rightW0sub1 = locate_dofs_topological(w0["displ"].function_space.sub(1), 1, right_support_lines)

comm = MPI.COMM_WORLD
filename = args.out
with XDMFFile(comm, filename, "w") as ofile:
    ofile.write_mesh(mesh)

force = Constant(mesh, 0.0)
dforce = Constant(mesh, 0.0)

f = {ds(1): (force + dforce) * ufl.as_vector((0, -1.e-6, 0))}


body_force = Constant(mesh, 1.0)

g = {
    dx(10): body_force * 1.0E-6 * fecoda.mech.rho_c * ufl.as_vector((0, 9.81, 0)),
    # Rebars
    dx(2): body_force * 1.0E-6 * fecoda.mech.rho_c * ufl.as_vector((0, 9.81, 0)),
    dx(3): body_force * 1.0E-6 * fecoda.mech.rho_c * ufl.as_vector((0, 9.81, 0)),
    dx(4): body_force * 1.0E-6 * fecoda.mech.rho_c * ufl.as_vector((0, 9.81, 0)),
    dx(5): body_force * 1.0E-6 * fecoda.mech.rho_c * ufl.as_vector((0, 9.81, 0)),
    # dx(6): body_force * 1.0E-6 * fecoda.mech.rho_rebar * ufl.as_vector((0, 9.81, 0)),
    # dx(7): body_force * 1.0E-6 * fecoda.mech.rho_rebar * ufl.as_vector((0, 9.81, 0))
}

water_flux = {}
co2_flux = {}
heat_flux = {}

J, F, expr_compiled =\
    fecoda.main.compile_forms(intern_var0, intern_var1, w0, w1, f, g,
                              heat_flux, water_flux, co2_flux, t, dt,
                              None, dx(10) + dx(2) + dx(3) + dx(4) + dx(5))

Joff, Foff, expr_compiledoff =\
    fecoda.main.compile_forms(intern_var0, intern_var1, w0, w1, f, g,
                              heat_flux, water_flux, co2_flux, t, dt,
                              None, dx(2) + dx(3) + dx(4) + dx(5) + dx(10), damage_off=True)

stress_rebar = Function(intern_var0["sigma"].function_space, name="stress_rebar")
cr_inel = Function(intern_var0["dmg"].function_space, name="cr_inel_magn")

log = {"times": [], "disp": []}

# Global time loop
for k in range(len(dtimes)):

    # Relative scale of already applied loading
    scale = 0.0

    # Scale of last successful call to solver
    dscale = 1.0

    # If time is greater than validity of Kelvin chain
    # sampling last time
    if times[k] + dtimes[k] > 0.5 * fecoda.mps.tau[-1]:
        if rank == 0:
            logger.warning("Validity of Kelvin sampling reached in {} days".format(t.value))
        break

    if k == 0:
        dforce_val = args.force
    elif times[k] == times_replaced[0]:
        force.value = 0.0
        dforce_val = args.force
    else:
        dforce_val = 0.0

    while scale < 1.0:

        dscale = min(1.0 - scale, 2.0 * dscale)

        t.value = times[k] + scale * dtimes[k]
        dt.value = dscale * dtimes[k]
        dforce.value = dscale * dforce_val

        if rank == 0:
            logger.info(79 * "#")
            logger.info(f"Time: {times[k]} + {(scale) * dtimes[k]} days.")
            logger.info(f"Dtime: {dt.value}")
            logger.info(f"Scale: {scale}")
            logger.info(f"Dscale: {dscale}")
            logger.info(f"Force: {force.value}")
            logger.info(f"Dforce: {dforce.value}")
            logger.info("Step: {}".format(k))
            logger.info(79 * "#")

        bcs_displ = []
        bcs_temp = []
        bcs_hum = []
        bcs_co2 = []

        if times[k] == t1:

            if rank == 0:
                logger.info(79 * "!")
                logger.info("Replacement")

            with displ_bc.vector.localForm() as localbc:
                with w0["displ"].vector.localForm() as local:
                    localbc.array[replaced_part_interface_dofs_W0] = - local.array[replaced_part_interface_dofs_W0]

            bcs_displ.append(DirichletBC(displ_bc, replaced_part_interface_dofs_W0))
        else:
            with displ_bc.vector.localForm() as local:
                local.set(0.0)

        bcs_displ.append(DirichletBC(displ_bc, leftW0))
        bcs_displ.append(DirichletBC(displ_bc, rightW0sub1))
        bcs_displ.append(DirichletBC(displ_bc, right_side_dofs_W0sub))

        if times[k] == t1:
            _scale = fecoda.main.solve_displ_system(Joff[0], Foff[0], intern_var0, intern_var1, expr_compiledoff,
                                                    w0, w1, bcs_displ, dforce, dt, t, k)
            fecoda.main.post_update(expr_compiledoff, intern_var0, intern_var1, w0, w1)
        else:
            _scale = fecoda.main.solve_displ_system(J[0], F[0], intern_var0, intern_var1, expr_compiled,
                                                    w0, w1, bcs_displ, dforce, dt, t, k)
            fecoda.main.post_update(expr_compiled, intern_var0, intern_var1, w0, w1)

        dscale *= _scale
        scale += dscale
        force.value += dforce.value
        t.value += dt.value

        fecoda.main.copyout_state(w0, w1, intern_var0, intern_var1)

        inel = intern_var0["eps_cr_kel"] + intern_var0["eps_cr_dash"]
        dolfiny.interpolation.interpolate(ufl.sqrt(ufl.inner(inel, inel)), cr_inel)

        with XDMFFile(comm, filename, "a") as ofile:
            print(t.value)
            ofile.write_function(w1["displ"], t.value)
            ofile.write_function(intern_var0["dmg"], t.value)
            ofile.write_function(cr_inel, t.value)

    # Evaluate displacement at midspan bottom
    bb_tree = BoundingBoxTree(mesh, 3)
    p = np.array([xdim / 2, 0.0, zdim], dtype=np.float64)
    cell_candidates = cpp.geometry.compute_collisions_point(bb_tree, p)
    cell = cpp.geometry.select_colliding_cells(mesh, cell_candidates, p, 1)

    if len(cell) > 0:
        value = w1["displ"].eval(p, cell)[1]
    else:
        value = None
    values = comm.gather(value, root=0)

    if rank == 0:
        disp = [x for x in values if x is not None][0]
        log["times"].append(float(t.value))
        log["disp"].append(disp)

        with open("disp.log", "w") as file:
            json.dump(log, file)

    if times[k] == t1:
        for i in range(fecoda.mps.M):
            with intern_var0[f"gamma_{i}"].vector.localForm() as local:
                local.array[replaced_part_dofs] = 0.0
        with intern_var0["eps_cr_dash"].vector.localForm() as local:
            local.array[replaced_part_dofs] = 0.0
        with intern_var0["eps_cr_kel"].vector.localForm() as local:
            local.array[replaced_part_dofs] = 0.0

        with w0["displ"].vector.localForm() as local:
            local.array[replaced_part_dofs_W0] = 0.0

        with intern_var0["sigma"].vector.localForm() as local:
            local.array[replaced_part_dofs] = 0.0

print(f"Simulation finished in {time() - t_init}")
