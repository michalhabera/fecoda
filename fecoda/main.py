import logging
from collections import OrderedDict
from contextlib import ExitStack
from time import time

import numpy as np
import argparse

import ufl
import ufl.algorithms
from dolfinx import cpp, Function, FunctionSpace
from dolfinx.fem import (Form, apply_lifting, assemble_matrix,
                         assemble_matrix_block, assemble_vector,
                         assemble_vector_block, create_matrix,
                         create_matrix_block, create_vector,
                         create_vector_block, set_bc)
from dolfinx.la import VectorSpaceBasis
from dolfiny.function import extract_blocks
from dolfiny.interpolation import CompiledExpression, interpolate_cached
from fecoda import co2, mech, misc, mps, water, damage_rankine
from mpi4py import MPI
from petsc4py import PETSc

_i, _j, _k, _l, _m, _n, _o, _p = ufl.indices(8)

logger = logging.getLogger("fecoda")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--disp_degree", type=int, default=2)
args = parser.parse_known_args()[0]

temp_degree = 1
hum_degree = 1
co2_degree = 1

quad_degree_stress = 1 if args.disp_degree == 1 else 6
quad_degree_thc = 8


class ConvergenceError(Exception):
    pass


def initialize_functions(mesh):

    STRESS_elem = ufl.TensorElement("DG", mesh.ufl_cell(), args.disp_degree - 1)
    DMG_elem = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=quad_degree_stress, quad_scheme="default")

    # Space for dashpot viscosity should ideally be Quadrature, but this wouldn't
    # work since we use it to update other non-Quadrature functions
    ETA_DASH_elem = ufl.FiniteElement("DG", mesh.ufl_cell(), 2)
    DISPL_elem = ufl.VectorElement("CG", mesh.ufl_cell(), args.disp_degree)
    TEMP_elem = ufl.FiniteElement("CG", mesh.ufl_cell(), temp_degree)
    HUM_elem = ufl.FiniteElement("CG", mesh.ufl_cell(), hum_degree)
    CO2_elem = ufl.FiniteElement("CG", mesh.ufl_cell(), co2_degree)

    STRESS = FunctionSpace(mesh, STRESS_elem)
    DMG = FunctionSpace(mesh, DMG_elem)
    ETA_DASH = FunctionSpace(mesh, ETA_DASH_elem)
    DISPL = FunctionSpace(mesh, DISPL_elem)
    TEMP = FunctionSpace(mesh, TEMP_elem)
    HUM = FunctionSpace(mesh, HUM_elem)
    CO2 = FunctionSpace(mesh, CO2_elem)

    W = []
    W.append(DISPL)
    W.append(TEMP)
    W.append(HUM)
    W.append(CO2)

    # Initial temperature
    temp0 = Function(W[1], name="temp0")
    temp1 = Function(W[1], name="temp1")

    # Initial humidity
    phi0 = Function(W[2], name="phi0")
    phi1 = Function(W[2], name="phi1")

    # Initial CO2 concentration
    co20 = Function(W[3], name="co20")
    co21 = Function(W[3], name="co21")

    # Displacement
    displ0 = Function(W[0], name="displ0")
    displ1 = Function(W[0], name="displ1")

    # Pack all internal variables
    # This is done just for brevity of solve method interface
    intern_var0 = OrderedDict()
    intern_var0["eta_dash"] = Function(ETA_DASH)
    intern_var0["caco3"] = Function(CO2, name="caco3")

    intern_var0["eps_cr_kel"] = Function(STRESS, name="eps_cr_kel")
    intern_var0["eps_cr_dash"] = Function(STRESS, name="eps_cr_dash")
    intern_var0["eps_sh_dr"] = Function(STRESS)
    intern_var0["eps_th"] = Function(STRESS)
    intern_var0["eps_eqv"] = Function(DMG, name="eps_eqv")
    intern_var0["sigma"] = Function(STRESS, name="sigma")
    intern_var0["dmg"] = Function(DMG, name="dmg")

    for i in range(mps.M):
        intern_var0[f"gamma_{i}"] = Function(STRESS)

    intern_var1 = OrderedDict()
    intern_var1["eta_dash"] = Function(intern_var0["eta_dash"].function_space)
    intern_var1["caco3"] = Function(intern_var0["caco3"].function_space)
    intern_var1["eps_cr_kel"] = Function(intern_var0["eps_cr_kel"].function_space)
    intern_var1["eps_cr_dash"] = Function(intern_var0["eps_cr_dash"].function_space)
    intern_var1["eps_sh_dr"] = Function(intern_var0["eps_sh_dr"].function_space)
    intern_var1["eps_th"] = Function(intern_var0["eps_th"].function_space)
    intern_var1["eps_eqv"] = Function(intern_var0["eps_eqv"].function_space)
    intern_var1["sigma"] = Function(intern_var0["sigma"].function_space)
    intern_var1["dmg"] = Function(intern_var0["dmg"].function_space)

    for i in range(mps.M):
        intern_var1[f"gamma_{i}"] = Function(intern_var0[f"gamma_{i}"].function_space)

    # Pack all "solved-for" quantities
    w0 = OrderedDict()
    w0["displ"] = displ0
    w0["temp"] = temp0
    w0["phi"] = phi0
    w0["co2"] = co20

    # Pack all "solved-for" quantities
    w1 = OrderedDict()
    w1["displ"] = displ1
    w1["temp"] = temp1
    w1["phi"] = phi1
    w1["co2"] = co21

    return w0, w1, intern_var0, intern_var1


def compile_forms(iv0, iv1, w0, w1, f, g, heat_flux, water_flux, co2_flux,
                  t, dt, reb_dx, con_dx, damage_off=False, randomize=0.0):
    """Return Jacobian and residual forms"""

    t0 = time()

    mesh = w0["displ"].function_space.mesh

    # Prepare zero initial guesses, test and trial fctions, global
    w_displ_trial = ufl.TrialFunction(w0["displ"].function_space)
    w_displ_test = ufl.TestFunction(w0["displ"].function_space)

    w_temp_trial = ufl.TrialFunction(w0["temp"].function_space)
    w_temp_test = ufl.TestFunction(w0["temp"].function_space)

    w_phi_trial = ufl.TrialFunction(w0["phi"].function_space)
    w_phi_test = ufl.TestFunction(w0["phi"].function_space)

    w_co2_trial = ufl.TrialFunction(w0["co2"].function_space)
    w_co2_test = ufl.TestFunction(w0["co2"].function_space)

    #
    # Creep, shrinkage strains
    #

    # Autogenous shrinkage increment
    deps_sh_au = (mps.eps_sh_au(t + dt) - mps.eps_sh_au(t))

    # Thermal strain increment
    deps_th = (misc.beta_C
               * (w1["temp"] - w0["temp"])
               * ufl.Identity(3))

    # Drying shrinkage increment
    deps_sh_dr = (mps.k_sh
                  * (w1["phi"] - w0["phi"])
                  * ufl.Identity(3))

    eta_dash_mid = mps.eta_dash(iv0["eta_dash"], dt / 2.0, w0["temp"], w0["phi"])

    # Prepare creep factors
    beta_cr = mps.beta_cr(dt)
    lambda_cr = mps.lambda_cr(dt, beta_cr)
    creep_v_mid = mps.creep_v(t + dt / 2.0)

    if randomize > 0.0:
        # Randomize Young's modulus
        # This helps convergence at the point where crack initiation begins
        # Randomized E fluctuates uniformly in [E-eps/2, E+eps/2]
        rnd = Function(iv0["eta_dash"].function_space)
        rnd.vector.array[:] = 1.0 - randomize / 2 + np.random.rand(*rnd.vector.array.shape) * randomize
    else:
        rnd = 1.0

    E_kelv = rnd * mps.E_kelv(creep_v_mid, lambda_cr, dt, t, eta_dash_mid)

    gamma0 = []
    for i in range(mps.M):
        gamma0.append(iv0[f"gamma_{i}"])

    deps_cr_kel = mps.deps_cr_kel(beta_cr, gamma0, creep_v_mid)
    deps_cr_dash = mps.deps_cr_dash(iv0["sigma"], eta_dash_mid, dt)

    deps_el = mech.eps_el(w1["displ"] - w0["displ"], deps_th, deps_cr_kel,
                          deps_cr_dash, deps_sh_dr, deps_sh_au)

    # Water vapour saturation pressure
    p_sat = water.p_sat(0.5 * (w1["temp"] + w0["temp"]))
    water_cont = water.water_cont(0.5 * (w1["phi"] + w0["phi"]))
    dw_dphi = water.dw_dphi(0.5 * (w1["phi"] + w0["phi"]))

    # Rate of CaCO_3 concentration change
    dot_caco3 = co2.dot_caco3(dt * 24 * 60 * 60, iv0["caco3"], w1["phi"], w1["co2"], w1["temp"])

    #
    # Balances residuals
    #

    sigma_eff = iv0["sigma"] + mech.stress(E_kelv, deps_el)
    eps_eqv = ufl.Max(damage_rankine.eps_eqv(sigma_eff, mps.E_static(creep_v_mid)), iv0["eps_eqv"])
    f_c = mech.f_c(t)
    f_t = mech.f_t(f_c)
    G_f = mech.G_f(f_c)
    dmg = damage_rankine.damage(eps_eqv, mesh, mps.E_static(creep_v_mid), f_t, G_f)

    # Prepare stress increments
    if damage_off:
        dmg = 0.0 * dmg
        eps_eqv = iv0["eps_eqv"]

    sigma_rebar = mech.stress_rebar(mech.E_rebar, w1["displ"])
    sigma = (1.0 - dmg) * sigma_eff

    _con_dx = []
    for dx in con_dx:
        _con_dx += [dx(metadata={"quadrature_degree": quad_degree_stress})]
    _con_dx = ufl.classes.MeasureSum(*_con_dx)

    _reb_dx = []
    for dx in reb_dx:
        _reb_dx += [dx(metadata={"quadrature_degree": quad_degree_stress})]
    _reb_dx = ufl.classes.MeasureSum(*_reb_dx)

    # Momentum balance for concrete material
    mom_balance = - ufl.inner(sigma, ufl.grad(w_displ_test)) * _con_dx

    # Momentum balance for rebar material
    if len(reb_dx) > 0:
        mom_balance += - ufl.inner(sigma_rebar, ufl.grad(w_displ_test)) * _reb_dx

    # Add volume body forces
    for measure, force in g.items():
        mom_balance -= ufl.inner(force, w_displ_test) * measure

    # Add surface forces to mom balance
    for measure, force in f.items():
        mom_balance += ufl.inner(force, w_displ_test) * measure

    _thc_dx = []
    for dx in con_dx + reb_dx:
        _thc_dx += [dx(metadata={"quadrature_degree": quad_degree_thc})]
    _thc_dx = ufl.classes.MeasureSum(*_thc_dx)

    # Energy balance = evolution of temperature
    energy_balance = (mech.rho_c * misc.C_pc / (dt * 24 * 60 * 60)
                      * ufl.inner((w1["temp"] - w0["temp"]), w_temp_test) * _thc_dx
                      + misc.lambda_c * ufl.inner(ufl.grad(w1["temp"]), ufl.grad(w_temp_test)) * _thc_dx
                      + water.h_v * water.delta_p
                      * ufl.inner(ufl.grad(w1["phi"] * p_sat), ufl.grad(w_temp_test)) * _thc_dx
                      + co2.alpha3 * dot_caco3 * w_temp_test * _thc_dx)

    # Water balance = evolution of humidity
    water_balance = (
        ufl.inner(dw_dphi * 1.0 / (dt * 24 * 60 * 60) * (w1["phi"] - w0["phi"]), w_phi_test) * _thc_dx
        + ufl.inner(dw_dphi * water.D_ws(water_cont) * ufl.grad(w1["phi"])
                    + water.delta_p * ufl.grad(w1["phi"] * p_sat), ufl.grad(w_phi_test)) * _thc_dx
        + co2.alpha2 * dot_caco3 * w_phi_test * _thc_dx
    )

    for measure, flux in water_flux.items():
        water_balance -= ufl.inner(flux, w_phi_test) * measure

    co2_balance = (
        ufl.inner(1.0 / (dt * 24 * 60 * 60) * (w1["co2"] - w0["co2"]), w_co2_test) * _thc_dx
        + ufl.inner(co2.D_co2 * ufl.grad(w1["co2"]), ufl.grad(w_co2_test)) * _thc_dx
        + co2.alpha4 * dot_caco3 * w_co2_test * _thc_dx
    )

    for measure, flux in co2_flux.items():
        co2_balance -= ufl.inner(flux, w_co2_test) * measure

    J_mom = ufl.derivative(mom_balance, w1["displ"], w_displ_trial)

    J_energy_temp = ufl.derivative(energy_balance, w1["temp"], w_temp_trial)
    J_energy_hum = ufl.derivative(energy_balance, w1["phi"], w_phi_trial)
    J_energy_co2 = ufl.derivative(energy_balance, w1["co2"], w_co2_trial)

    J_energy = J_energy_hum + J_energy_temp + J_energy_co2

    J_water_temp = ufl.derivative(water_balance, w1["temp"], w_temp_trial)
    J_water_hum = ufl.derivative(water_balance, w1["phi"], w_phi_trial)
    J_water_co2 = ufl.derivative(water_balance, w1["co2"], w_co2_trial)

    J_water = J_water_temp + J_water_hum + J_water_co2

    J_co2_temp = ufl.derivative(co2_balance, w1["temp"], w_temp_trial)
    J_co2_hum = ufl.derivative(co2_balance, w1["phi"], w_phi_trial)
    J_co2_co2 = ufl.derivative(co2_balance, w1["co2"], w_co2_trial)

    J_co2 = J_co2_temp + J_co2_hum + J_co2_co2

    # Put all Jacobians together
    J_all = J_mom + J_energy + J_water + J_co2

    # Lower algebra symbols and apply derivatives up to terminals
    # This is needed for the Replacer to work properly
    preserve_geometry_types = (ufl.CellVolume, ufl.FacetArea)
    J_all = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(J_all)
    J_all = ufl.algorithms.apply_derivatives.apply_derivatives(J_all)
    J_all = ufl.algorithms.apply_geometry_lowering.apply_geometry_lowering(J_all, preserve_geometry_types)
    J_all = ufl.algorithms.apply_derivatives.apply_derivatives(J_all)
    J_all = ufl.algorithms.apply_geometry_lowering.apply_geometry_lowering(J_all, preserve_geometry_types)
    J_all = ufl.algorithms.apply_derivatives.apply_derivatives(J_all)

    J = extract_blocks(J_all, [w_displ_test, w_temp_test, w_phi_test, w_co2_test],
                       [w_displ_trial, w_temp_trial, w_phi_trial, w_co2_trial])

    J[0][0]._signature = "full" if not damage_off else "dmgoff"

    # Just make sure these are really empty Forms
    assert len(J[1][0].arguments()) == 0
    assert len(J[2][0].arguments()) == 0
    assert len(J[3][0].arguments()) == 0

    F = [-mom_balance, -energy_balance, -water_balance, -co2_balance]

    rank = MPI.COMM_WORLD.rank

    if rank == 0:
        logger.info("Compiling tangents J...")
    J_compiled = [Form(J[0][0]), [[Form(J[i][j]) for j in range(1, 4)] for i in range(1, 4)]]

    if rank == 0:
        logger.info("Compiling residuals F...")
    F_compiled = [Form(F[0]), [Form(F[i]) for i in range(1, 4)]]

    expr = OrderedDict()

    eta_dash1 = mps.eta_dash(iv0["eta_dash"], dt, w1["temp"], w1["phi"])
    expr["eta_dash"] = (eta_dash1, iv0["eta_dash"].function_space.ufl_element())
    expr["caco3"] = (iv0["caco3"] + dt * 24 * 60 * 60 * dot_caco3, iv0["caco3"].function_space.ufl_element())

    expr["eps_cr_kel"] = (iv0["eps_cr_kel"] + deps_cr_kel, iv0["eps_cr_kel"].function_space.ufl_element())
    expr["eps_cr_dash"] = (iv0["eps_cr_dash"] + deps_cr_dash, iv0["eps_cr_dash"].function_space.ufl_element())

    expr["eps_sh_dr"] = (iv0["eps_sh_dr"] + deps_sh_dr, iv0["eps_sh_dr"].function_space.ufl_element())
    expr["eps_th"] = (iv0["eps_th"] + deps_th, iv0["eps_th"].function_space.ufl_element())

    expr["sigma"] = (iv0["sigma"] + mech.stress(E_kelv, deps_el), iv0["sigma"].function_space.ufl_element())
    expr["eps_eqv"] = (eps_eqv, iv0["eps_eqv"].function_space.ufl_element())
    expr["dmg"] = (dmg, iv0["dmg"].function_space.ufl_element())

    for i in range(mps.M):
        expr[f"gamma_{i}"] = (lambda_cr[i] * (iv1["sigma"] - iv0["sigma"]) + (beta_cr[i])
                              * gamma0[i], gamma0[i].function_space.ufl_element())

    expr_compiled = OrderedDict()
    for name, item in expr.items():
        if rank == 0:
            logger.info(f"Compiling expressions for {name}...")
        expr_compiled[name] = CompiledExpression(item[0], item[1])

    if rank == 0:
        logger.info(f"[Timer] UFL forms setup and compilation: {time() - t0}")

    return J_compiled, F_compiled, expr_compiled


def post_update(expr_compiled, intern_var0, intern_var1, w0, w1):
    """Update internal variables after the solution was found."""
    rank = MPI.COMM_WORLD.rank

    names = ["caco3", "eta_dash", "eps_cr_kel", "eps_cr_dash",
             "eps_sh_dr", "eps_th", "sigma", "eps_eqv", "dmg"]
    for i in range(mps.M):
        names.append(f"gamma_{i}")

    # Interoplate into 1-state and copy back into 0-state
    for name in names:
        t0 = time()
        interpolate_cached(expr_compiled[name], intern_var1[name])
        norm = intern_var1[name].vector.norm()
        if rank == 0:
            logger.info(f"[Timer] ||{name}||: {norm:.1f}, post-update: {time() - t0:.4f}")


def copyout_state(w0, w1, intern_var0, intern_var1):
    rank = MPI.COMM_WORLD.rank

    t0 = time()
    for name in w0.keys():
        with w1[name].vector.localForm() as w1_local, w0[name].vector.localForm() as w0_local:
            w1_local.copy(w0_local)

    for name in intern_var0.keys():
        with intern_var1[name].vector.localForm() as iv1_local, intern_var0[name].vector.localForm() as iv0_local:
            iv1_local.copy(iv0_local)

    if rank == 0:
        logger.info(f"[Timer] Copyout state: {time() - t0:.4f}")


def solve_thc_system(J, F, intern_var0, intern_var1, w0, w1, bcs, rtol=1.e-6, atol=1.e-12, max_its=10):
    """Solve system for temperature-humidity-co2"""

    rank = MPI.COMM_WORLD.rank

    # Preallocate diagonal block matrices
    t0 = time()
    A = create_matrix_block(J)

    if rank == 0:
        logger.info(f"[Timer] Preallocation matrix time: {time() - t0:.3f}")

    t0 = time()

    t0 = time()
    b = create_vector_block(F)
    if rank == 0:
        logger.info(f"[Timer] Preallocation vector time: {time() - t0:.3f}")

    assemble_vector_block(b, F, J, bcs)
    resid_norm0 = b.norm()

    if rank == 0:
        print(f"Initial THC residual: {resid_norm0:.2e}")

    # Newton iteration for THC blocks
    for iter in range(max_its):

        if rank == 0:
            logger.info(f"Newton iteration for THC {iter}")

        if iter > 0:
            for bc in bcs:
                with bc.value.vector.localForm() as locvec:
                    locvec.set(0.0)

        A.zeroEntries()
        with b.localForm() as local:
            local.set(0.0)

        size = A.getSize()[0]
        local_size = A.getLocalSize()[0]
        t0 = time()
        assemble_matrix_block(A, J, bcs)
        A.assemble()
        _time = time() - t0

        if rank == 0:
            logger.info(f"[Timer] A1 size: {size}, local size: {local_size}")
            logger.info(f"[Timer] A1 assembly: {_time:.4f}")
            logger.info(f"[Timer] A1 assembly dofs/s: {size / _time:.1f}")

        t0 = time()
        with b.localForm() as local:
            local.set(0.0)
        assemble_vector_block(b, F, J, bcs)
        _time = time() - t0
        if rank == 0:
            logger.info(f"[Timer] b1 assembly: {_time}")
            logger.info(f"[Timer] b1 assembly dofs/s: {size / _time}")

        t0 = time()

        ksp = PETSc.KSP()
        ksp.create(MPI.COMM_WORLD)
        ksp.setOptionsPrefix("thc")
        ksp.setFromOptions()

        x = A.createVecLeft()
        ksp.setOperators(A)
        ksp.solve(b, x)

        if rank == 0:
            its = ksp.its
            t1 = time() - t0
            dofsps = int(size / t1)

            logger.info(f"[Timer] A1 converged in: {its}")
            logger.info(f"[Timer] A1 solve {t1:1.3}")
            logger.info(f"[Timer] A1 solver dofs/s: {dofsps}")

        # Update all subfunctions
        u = [w1["temp"], w1["phi"], w1["co2"]]
        offset = 0
        for i in range(3):
            size_local = u[i].vector.getLocalSize()
            u[i].vector.array[:] += x.array_r[offset:offset + size_local]
            offset += size_local
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with b.localForm() as local:
            local.set(0.0)
        assemble_vector_block(b, F, J, bcs)

        norm = b.norm()

        if resid_norm0 == 0.0:
            rel_resid_norm = 0.0
        else:
            rel_resid_norm = norm / resid_norm0

        if rank == 0:
            logger.info("---")
            logger.info(f"Abs. resid norm: {norm:.2e}")
            logger.info(f"Rel. resid norm: {rel_resid_norm:.2e}")
            logger.info("---")

        if rel_resid_norm < rtol or norm < atol:
            break


def solve_displ_system(J, F, intern_var0, intern_var1, expr_compiled, w0, w1, bcs, df, dt, t, k,
                       io_callback=None, refinement_callback=None, rtol=1.e-6, atol=1.e-16, max_its=20):
    """Solve system for displacement"""

    rank = MPI.COMM_WORLD.rank

    t0 = time()
    A = create_matrix(J)

    if rank == 0:
        logger.info(f"[Timer] Preallocation matrix time {time() - t0:.3f}")

    t0 = time()

    t0 = time()
    b = create_vector(F)
    if rank == 0:
        logger.info(f"[Timer] Preallocation vector time {time() - t0:.3f}")

    with b.localForm() as local:
        local.set(0.0)
    assemble_vector(b, F)
    apply_lifting(b, [J], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    resid_norm0 = b.norm()
    if rank == 0:
        print(f"Initial DISPL residual: {resid_norm0:.2e}")

    iter = 0
    # Total number of NR iterations including refinement attempts
    iter_total = 0
    converged = False
    refine = 0
    scale = 1.0
    rel_resid_norm = 1.0

    bcs_init = []
    for bc in bcs:
        bcs_init += [bc.value.vector.duplicate()]
        with bcs_init[-1].localForm() as loc0, bc.value.vector.localForm() as loc1:
            loc1.copy(loc0)

    df0 = np.copy(df.value)

    while converged is False:

        if iter > max_its or rel_resid_norm > 1.e+2:
            refine += 1
            iter = 0

            if rank == 0:
                logger.info(79 * "!")
                logger.info(f"Restarting NR with rel. stepsize: {1.0 / (2 ** refine)}")

            with w0["displ"].vector.localForm() as w_local, w1["displ"].vector.localForm() as w1_local:
                w_local.copy(w1_local)

            df.value = df0
            scale = 1.0 / 2 ** refine

            if refine > 10:
                raise ConvergenceError("Inner adaptivity reqiures > 10 refinements.")

            # Reset and scale boundary condition
            for i, bc in enumerate(bcs_init):
                with bcs[i].value.vector.localForm() as bcsi_local, bc.localForm() as bc_local:
                    bc_local.copy(bcsi_local)
                    bcsi_local.scale(scale)

            df.value *= scale
            dt.value *= scale

            if refinement_callback is not None:
                refinement_callback(scale)

        if rank == 0:
            logger.info("Newton iteration for displ {}".format(iter))

        if iter > 0:
            for bc in bcs:
                with bc.value.vector.localForm() as locvec:
                    locvec.set(0.0)

        A.zeroEntries()
        with b.localForm() as local:
            local.set(0.0)

        size = A.getSize()[0]
        local_size = A.getLocalSize()[0]
        t0 = time()
        assemble_matrix(A, J, bcs)
        A.assemble()
        _time = time() - t0

        Anorm = A.norm()
        if rank == 0:
            logger.info(f"[Timer] A0 size: {size}, local size: {local_size}")
            logger.info(f"[Timer] A0 assembly: {_time:.4f}")
            logger.info(f"[Timer] A0 assembly dofs/s: {size / _time:.1f}")
            logger.info(f"A0 norm: {Anorm:.4f}")

        t0 = time()
        assemble_vector(b, F)
        apply_lifting(b, [J], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)
        bnorm = b.norm()
        if rank == 0:
            logger.info(f"[Timer] b0 assembly {time() - t0:.4f}")
            logger.info(f"b norm: {bnorm:.4f}")

        nsp = build_nullspace(w1["displ"].function_space)

        ksp = PETSc.KSP()
        ksp.create(MPI.COMM_WORLD)
        ksp.setOptionsPrefix("disp")
        opts = PETSc.Options()

        A.setNearNullSpace(nsp)
        A.setBlockSize(3)

        ksp.setOperators(A)
        x = A.createVecRight()

        ksp.setFromOptions()
        t0 = time()
        ksp.solve(b, x)
        t1 = time() - t0

        opts.view()

        xnorm = x.norm()

        if rank == 0:
            its = ksp.its
            t1 = time() - t0
            dofsps = int(size / t1)

            logger.info(f"[Timer] A0 converged in: {its}")
            logger.info(f"[Timer] A0 solve: {t1:.4f}")
            logger.info(f"[Timer] A0 solve dofs/s: {dofsps:.1f}")
            logger.info(f"Increment norm: {xnorm}")

        # TODO: Local axpy segfaults, could ghostUpdate be avoided?
        w1["displ"].vector.axpy(1.0, x)
        w1["displ"].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        #
        # Evaluate true residual and check
        #

        with b.localForm() as local:
            local.set(0.0)
        assemble_vector(b, F)
        apply_lifting(b, [J], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        norm = b.norm()
        rel_resid_norm = norm / resid_norm0
        rel_dx_norm = x.norm() / w1["displ"].vector.norm()

        if rank == 0:
            logger.info("---")
            logger.info(f"Abs. resid norm: {norm:.2e}")
            logger.info(f"Rel. dx norm: {rel_dx_norm:.2e}")
            logger.info(f"Rel. resid norm: {rel_resid_norm:.2e}")
            logger.info("---")

        iter += 1
        iter_total += 1

        if rel_resid_norm < rtol or norm < atol:
            if rank == 0:
                logger.info(f"Newton converged in: {iter}, total: {iter_total}")

            if io_callback is not None:
                io_callback(intern_var1, w1, t.value + dt.value)
            return scale


def solve_fullstep(J, F, intern_var0, intern_var1, expr_compiled, w0, w1,
                   bcs_displ, bcs_thc, force, df, dt, t, k, dscale, io_callback=None,
                   rtol=1.e-6, atol=1.e-16, max_its=20,
                   adapt_limit=1.e-3):

    rank = MPI.COMM_WORLD.rank
    scale = 0.0

    dt0 = np.copy(dt.value)
    df0 = np.copy(df.value)
    t0 = np.copy(t.value)

    bcs_displ0 = []
    for bc in bcs_displ:
        bcs_displ0 += [bc.value.vector.duplicate()]
        with bcs_displ0[-1].localForm() as loc0, bc.value.vector.localForm() as loc1:
            loc1.copy(loc0)

    bcs_thc0 = []
    for bc in bcs_thc:
        bcs_thc0 += [bc.value.vector.duplicate()]
        with bcs_thc0[-1].localForm() as loc0, bc.value.vector.localForm() as loc1:
            loc1.copy(loc0)

    bcs_all0 = bcs_displ0 + bcs_thc0
    bcs_all = bcs_displ + bcs_thc

    while scale < 1.0:
        dscale = min(1.0 - scale, 2.0 * dscale)

        t.value = t0 + scale * dt0
        dt.value = dscale * dt0
        df.value = dscale * df0

        # Reset and scale boundary condition
        for i, bc0 in enumerate(bcs_all0):
            with bcs_all[i].value.vector.localForm() as bcsi_local, bc0.localForm() as bc0_local:
                bc0_local.copy(bcsi_local)
                bcsi_local.scale(dscale)

        def refinement_callback(scale):

            for i, bc0 in enumerate(bcs_thc0):
                with bcs_thc[i].value.vector.localForm() as bcsi_local, bc0.localForm() as bc0_local:
                    bc0_local.copy(bcsi_local)
                    bcsi_local.scale(scale * dscale)

            # Reset the failed NR solution to the initial guess
            for name in ["temp", "phi", "co2"]:
                with w0[name].vector.localForm() as w_local, w1[name].vector.localForm() as w1_local:
                    w_local.copy(w1_local)

            solve_thc_system(J[1], F[1], intern_var0, intern_var1, w0, w1, bcs_thc)

        if rank == 0:
            logger.info(79 * "#")
            logger.info(f"Time: {t0} + {scale * dt0} days.")
            logger.info(f"Scale: {scale}")
            logger.info(f"Dscale: {dscale}")
            logger.info(f"Force: {force.value}")
            logger.info(f"Dforce: {df.value}")
            logger.info(f"Step: {k}")

        solve_thc_system(J[1], F[1], intern_var0, intern_var1, w0, w1, bcs_thc)
        _scale = solve_displ_system(J[0], F[0], intern_var0, intern_var1, expr_compiled,
                                    w0, w1, bcs_displ, df, dt, t, k, io_callback=io_callback,
                                    refinement_callback=refinement_callback, rtol=rtol, atol=atol,
                                    max_its=max_its)
        post_update(expr_compiled, intern_var0, intern_var1, w0, w1)

        dscale *= _scale

        if dscale < adapt_limit:
            raise ConvergenceError(f"Outer adaptivity requires too small step, <{adapt_limit}")

        scale += dscale
        force.value += df.value
        t.value += dt.value

        copyout_state(w0, w1, intern_var0, intern_var1)

    return dscale


def build_nullspace(V):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    index_map = V.dofmap.index_map
    nullspace_basis = [cpp.la.create_vector(index_map, V.dofmap.index_map_bs) for i in range(6)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        dofs = [V.sub(i).dofmap.list.array for i in range(3)]

        # Build translational null space basis
        for i in range(3):
            basis[i][dofs[i]] = 1.0

        # Build rotational null space basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        basis[3][dofs[0]] = -x1
        basis[3][dofs[1]] = x0
        basis[4][dofs[0]] = x2
        basis[4][dofs[2]] = -x0
        basis[5][dofs[2]] = x1
        basis[5][dofs[1]] = -x2

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(6)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp
