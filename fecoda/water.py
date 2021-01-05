from ufl import conditional, ge, exp

rho_w = 1000  # kg m^{-3}

# Humidity evolution and coupling
# Moisture storage coeffs
b = 1.1  # [-]
w_f = 300.  # kg m^{-3}

# Vapour diffusion resistance factor
mu_vap = 260.
# Water vapour permeability of air
delta = 2.E-7 * 300. ** 0.81 / 101325.0
# Water vapour permeability of building
delta_p = delta / mu_vap

# Evaporation enthalpy of water
h_v = 2257. * 1.E+3  # J kg^{-1}

# Time for the onset of drying = time when wet covering is removed
t_drying_onset = 14  # [day]

# Water absoprtion coeff
A = 0.1


def p_sat(temp):
    """Water vapour saturation pressure

    Parameters
    ----------
    temp0: Previous temp function [K]

    Note
    ----
    Kunzel 1995, page 40, formula (50).

    """
    a = conditional(ge(temp, 273.15), 17.08, 22.44)
    theta0 = conditional(ge(temp, 273.15), 234.18, 272.44)

    return 611. * exp(a * (temp - 273.15) / (theta0 + (temp - 273.15)))


def water_cont(phi):
    """Water content

    Parameters
    ----------
    phi: Relative humidity [-]

    Note
    ----
    Kunzel 1995, page 13, formula (7)

    """
    return w_f * (b - 1.) * phi / (b - phi)


def dw_dphi(phi):
    """Water content derivative wrt humidity"""
    return w_f * (b - 1.0) * b / ((b - phi) ** 2.0)


def D_ws(w):
    """Capillary transport coefficient

    Parameters
    ----------
    w: water content [kg * m^{-3}]

    Note
    ----
    Kunzel 1995, page 25, formula (24)

    """
    return 3.8 * (A / w_f) ** 2 * 1000.0 ** (w / w_f - 1.0)
