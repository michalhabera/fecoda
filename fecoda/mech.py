import ufl
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--fc", type=float, help="28-day mean compressive strength")
parser.add_argument("--ft", type=float, help="Tensile strength [MPa] (if to be forced)")
parser.add_argument("--Gf", type=float, help="Fracture energy [N/m] (if to be forced)")
args = parser.parse_known_args()[0]

_i, _j, _k, _l, _m, _n, _o, _p = ufl.indices(8)

# Elasticity
# Unit Young's modulus
E = 1.0  # MPa
# Poisson ratio
nu = 0.2
# first Lame coeff
lambda_ = E * nu / ((1. + nu) * (1. - 2 * nu))
# shear modulus
mu = E / (2. * (1. + nu))

f_c28 = args.fc


def f_c(t):
    s = 0.25
    return f_c28 * ufl.exp(s * (1.0 - ufl.sqrt(28.0 / t)))


def f_t(f_c):
    if args.ft is not None:
        return args.ft
    else:
        return 0.3 * (f_c - 8) ** (2 / 3)


def G_f(f_c):
    if args.Gf is not None:
        return args.Gf
    else:
        return 73 * f_c ** 0.18


# Densities of cement, aggregates
rho_c = 2400  # kg m^{-3}
rho_a = 3000  # kg m^{-3}


def D_map(A):
    """Unit stiffness tensor mapping strain -> stress."""
    return lambda_ * ufl.tr(A) * ufl.Identity(3) + 2 * mu * A


def C_map(A):
    """unit compliance tensor mapping stress -> strain."""
    return - lambda_ / (2.0 * mu * (3.0 * lambda_ + 2.0 * mu)) * ufl.tr(A) * ufl.Identity(3) + 1.0 / (2 * mu) * A


# Young modulus for concrete reinforcement, ASTM-A36 steel
E_rebar = 200.0 * 1.0E+3  # MPa

# Poisson ratio
nu_rebar = 0.26
# first Lame coeff
lambda_rebar_ = nu_rebar / ((1. + nu_rebar) * (1. - 2 * nu_rebar))
# shear modulus
mu_rebar = 1.0 / (2. * (1. + nu_rebar))


def D_rebar_map(A):
    """Unit stiffness tensor (rebars) mapping strain -> stress."""
    return lambda_rebar_ * ufl.tr(A) * ufl.Identity(3) + 2 * mu_rebar * A


# Steel density
rho_rebar = 7800  # kg m^{-3}


def stress(E, eps_elastic):
    """Stress tensor

    Units: inherited from units of E
    """
    return E * D_map(eps_elastic)


def eps_el(displ, eps_th, eps_cr_kel, eps_cr_dash, eps_sh_dr, eps_sh_au):
    """Elastic strain tensor"""
    return (ufl.sym(ufl.grad(displ))  # total strain
            - eps_th  # thermal strain
            - eps_cr_kel  # creep kelvin
            - eps_cr_dash  # creep dashpot
            - eps_sh_dr  # drying shrinkage
            - eps_sh_au)


def stress_rebar(E, displ):
    return E * D_rebar_map(ufl.sym(ufl.grad(displ)))
