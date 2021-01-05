import ufl

import fecoda.misc
import fecoda.mps
import fecoda.mech
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--damage_off", action="store_true")

args = parser.parse_known_args()[0]

rank = MPI.COMM_WORLD.rank

dmg_eps = 1.0e-6


def eps_eqv(sigma, E):

    e1, e2, e3 = fecoda.misc.eig(sigma)
    eqv = ufl.Max(e1, ufl.Max(e2, e3)) / E

    return eqv


def damage(eps_eqv, mesh, E, f_t, G_f):
    h_cb = ufl.CellVolume(mesh) ** (1 / 3)
    eps0 = f_t / E

    eps_f = G_f / 1.0e+6 / (f_t * h_cb) + eps0 / 2

    dmg = ufl.Min(1.0 - eps0 / eps_eqv * ufl.exp(- (eps_eqv - eps0) / (eps_f - eps0)), 1.0 - dmg_eps)
    return ufl.conditional(eps_eqv <= eps0, 0.0, 0.0 if args.damage_off else dmg)
