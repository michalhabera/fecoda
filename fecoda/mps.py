import logging

from mpi4py import MPI
import fecoda.mech
from fecoda.misc import room_temp
import ufl
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--c", type=float, help="Cement content [kg m^-3]")
parser.add_argument("--wc", type=float, help="Water-cement ratio [-]")
parser.add_argument("--ac", type=float, help="Aggregates-cement ratio")
parser.add_argument("--ct", type=str, default="R", help="Cement type, R/RS/SL")

parser.add_argument("--q1", type=float)
parser.add_argument("--q2", type=float)
parser.add_argument("--q3", type=float)
parser.add_argument("--q4", type=float)

args = parser.parse_known_args()[0]

_i, _j, _k, _l = ufl.indices(4)
# Begin time for kelvin chain sampling
# This time is used to construct Kelvin chain sampling
t_begin = 1.0E-2  # [day]

c = args.c
wc = args.wc
ac = args.ac

# Shrinkage coeff
# Taken from OOFEM documentation examples
k_sh = 1.2E-3

w = wc * c
a = ac * c

# Number of Kelvin units in Kelvin creep chain
M = 8

# Microprestress solidification theory parameters
# These parameters are the same as for B3 compliance function
#
# The Book, page 49, formulas (3.25-3.28)
MPS_q1 = 1.0E-12 * 126.77 * fecoda.mech.f_c28 ** (-0.5)  # Pa^{-1}
if args.q1 is not None:
    MPS_q1 = 1.0E-12 * args.q1

MPS_q2 = 1.0E-12 * 185.4 * c ** 0.5 * fecoda.mech.f_c28 ** (-0.9)  # Pa^{-1}
if args.q2 is not None:
    MPS_q2 = 1.0E-12 * args.q2

MPS_q3 = 0.29 * (wc) ** 4 * MPS_q2  # Pa^{-1}
if args.q3 is not None:
    MPS_q3 = 1.0E-12 * args.q3

MPS_q4 = 1.0E-12 * 20.3 * (ac) ** (-0.7)  # Pa^{-1}
if args.q4 is not None:
    MPS_q4 = 1.0E-12 * args.q4

# MPS_q1 = 1.0E-9 * 0.7 / (4.733 * fecoda.mech.f_c28 ** 0.5)
# MPS_q2 = (wc / 0.38) ** 3 * 58.6 * 1.0e-3 * 1.0e-9
# MPS_q3 = 39.3 * 1.0e-3 * MPS_q2 * (ac / 6.0) ** -1.1 * (wc / 0.38) ** 0.4

alpha = MPS_q3 / MPS_q2

logger = logging.getLogger("fecoda")

rank = MPI.COMM_WORLD.rank
if rank == 0:
    logger.info(f"Q1 = {MPS_q1*1.0e12} 10^-12 Pa^-1")
    logger.info(f"Q2 = {MPS_q2*1.0e12} 10^-12 Pa^-1")
    logger.info(f"Q3 = {MPS_q3*1.0e12} 10^-12 Pa^-1")
    logger.info(f"Q4 = {MPS_q4*1.0e12} 10^-12 Pa^-1")


# Non-ageing elastic spring coupled to Kelvin chain
E_0 = 1.0 / MPS_q1 * 1.0e-6  # MPa

#
# Approximation of compliance function with ageing Kelvin chain
# The Book, page 779, formulas (F.37, F.38)
#

# Retardation times
tau = np.array([])

# Logarithmic times scale
for m in range(M):
    tau = np.append(tau, 0.3 * t_begin * 10 ** m)

tau_0 = (2.0 * tau[0] / np.sqrt(10.0)) ** 0.1
# Kelvin chain moduli
# zero-th spring is special, without dashpot
D_cr_0 = 1.0 / (MPS_q3 * (np.log(1 + tau_0)
                          - tau_0 / 10 / (1 + tau_0))) * 1.0e-6

if rank == 0:
    logger.info(f"D_cr_0 = {D_cr_0*1.e-3} GPa")

D_cr = np.array([])
# All other springs are approximated with continuous retard. spectrum
for m in range(M):
    tau_m = (2.0 * tau[m]) ** 0.1
    D_cr = np.append(
        D_cr,
        (1.0 + tau_m) ** 2 / (np.log(10.0) * MPS_q3
                              * 0.1 * tau_m * (0.9 + tau_m)) * 1.0e-6)

    if rank == 0:
        logger.info(f"D_cr_{m+1} = {D_cr[m]*1.e-3} GPa, tau_{m+1} = {tau[m]}")


def creep_v(t):
    """Ageing factor in creep Kelvin unit, i.e. volume growth fction

    Parameters
    ----------
    t: Time [days]

    Note
    ----
    The Book, page 415, formula (9.16), with m = 0.5

    """
    return 1.0 / (1.0 + 1.0 / alpha * (1.0 / t) ** 0.5)


def eta_dash(eta_dash0, dt, temp, hum):
    """Ageing viscosity of dashpot element

    Parameters
    ----------
    eta_dash0: Previous viscosity [MPa day]
    dt: Timestep [days]
    temp: Temperature [K]
    hum: Rel. humidity [-]

    Returns
    -------
    Ageing viscosity [MPa day]

    Note
    ----
    The Book, page 470, 471, 477, formulas (10.33, 10.36, 10.39, 10.60)

    """
    beta_sT = ufl.exp(3000 * (1.0 / room_temp - 1.0 / temp))
    alpha_s = 0.1
    beta_sh = alpha_s + (1 - alpha_s) * hum ** 2
    psi_s = beta_sT * beta_sh

    return eta_dash0 + dt * psi_s / MPS_q4 * 1.0e-6


def beta_cr(dt):
    """Beta creep coefficients

    Parameters
    ----------
    dt: Timestep size [day]

    Returns
    -------
    NumPy array of beta creep coefficients [-]

    Note
    ----
    The Book, page 160, formula (5.36)

    """
    beta = []

    for ta in tau:
        beta.append(ufl.conditional(dt / ta > 30.0, 0.0, ufl.exp(- dt / ta)))
    return beta


def lambda_cr(dt, betas):
    """Lambda creep coefficients

    Parameters
    ----------
    dt: Timestep size [day]

    Returns
    -------
    NumPy array of lambda creep coeffs [-]

    Note
    ----
    The Book, page 163, formula (5.48)

    """
    lambdas = []
    for i, ta in enumerate(tau):
        lambdas.append(ufl.conditional(dt / ta < 0.001, 1.0 - 1 / 2 * dt
                                       / ta + 1 / 6 * (dt / ta) ** 2, ta / dt * (1.0 - betas[i])))

    return lambdas


def E_static(creep_v):
    """Static modulus of elasticity

    Note
    ----
    This value increases with age and depends on the initial sampling time of
    Kelvin chain.
    """
    E_inv = 1.0 / E_0 + 1.0 / (D_cr_0 * creep_v)

    return 1.0 / E_inv


E_28 = E_static(creep_v(28.0))
E_35 = E_static(creep_v(35.0))

if rank == 0:
    logger.info(f"E_0 = {E_0*1.e-3} GPa")
    logger.info(f"E_28 = {E_28*1e-3} GPa")
    logger.info(f"E_35 = {E_35*1e-3} GPa")


def E_kelv(creep_v, lambda_cr, dt, t, eta_dash_mid):
    """ Effective Young modulus of the Kelvin chain

    Parameters
    ----------
    creep_v_mid: UFL expr for ageing factor, in mid-time step [-]
    lambda_cr: NumPy array of lambda creep coeffs [-]
    dt: Timestep size [day]
    t: Time [day]
    eta_dash_mid: Viscosity of ageing dashpot in mid-time step

    Returns
    -------
    UFL expr for effective young modulus [MPa]

    Note
    ----
    The Book, page 173, formula (5.83)

    """
    # Prepare temporary inverse modulus
    E_eff_inv = (1.0 / E_static(creep_v)
                 + dt / (2.0 * eta_dash_mid))

    # Add ageing springs in Kelvin chain with dashpots
    for i in range(M):
        E_eff_inv += ((1.0 - lambda_cr[i])
                      / (D_cr[i] * creep_v))

    return 1.0 / E_eff_inv


def deps_cr_kel(beta_cr, gamma0, creep_v_mid):
    """Strain increment due to creep in kelvin chain

    Parameters
    ----------
    beta_cr:
    gamma0: List of previous step gammas (internal variables)
    creep_v_mid: UFL expr for ageing factor

    Note
    ----
    The Book, page 173, formula (5.84). However, increment due to
    ageing dashpot is not considered here, but as a separate term later.

    """
    deps_cr_kel = []
    for i, beta in enumerate(beta_cr):
        deps_cr_kel.append((1.0 - beta) * gamma0[i] / (D_cr[i] * creep_v_mid))
    deps_cr_kel = sum(deps_cr_kel)
    deps_cr_kel = fecoda.mech.C_map(deps_cr_kel)
    return deps_cr_kel


def eps_sh_au(t):
    """Autogeneous shrinkage strain

    Parameters
    ----------
    t: Time [days]

    Note
    ----
    The Book, page 722, formulas (D.17, D.18, D.19, D.20)
    with parameters for cement type from table D.4

    """

    if args.ct == "R":
        r_alpha = 1.0
        r_eps = -3.5
        eps_aucem = 210 * 1.0e-6
        t_aucem = 1.0
    elif args.ct == "RS":
        r_alpha = 1.40
        r_eps = -3.5
        eps_aucem = -84 * 1.0e-6
        t_aucem = 41.0
    elif args.ct == "SL":
        r_alpha = 1.0
        r_eps = -3.5
        eps_aucem = 0.0
        t_aucem = 1.0

    # Autogeneous shrinkage
    eps_infty_sh_au = (eps_aucem
                       * (ac / 6.0) ** -0.75
                       * (wc / 0.38) ** r_eps)
    tau_au = t_aucem * (wc / 0.38) ** 3.0

    au_alpha = r_alpha * wc / 0.38

    return -eps_infty_sh_au * (1.0 + (tau_au / t) ** au_alpha) ** -4.5 * ufl.Identity(3)


def deps_cr_dash(sigma, eta_dash_mid, dt):
    """Increment in strains due to ageing dashpot

    Parameters
    ----------
    sigma: Stress [MPa]
    eta_dash_mid: Dashpot viscosity at midtime step [Mpa day]
    dt: Timestep [day]

    Note
    ----
    The Book, page 173, formula (5.84). This is the part of the formula,
    the rest is in deps_cr_kel() method.

    """
    return dt * fecoda.mech.C_map(sigma) / eta_dash_mid
