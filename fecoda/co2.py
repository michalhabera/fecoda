import ufl

# Model here is based on paper
# CEMENT and CONCRETE RESEARCH. Vol. 23,pp.761-772,1993,
# THE CARBONATION OF CONCRETE AND THE MECHANISM OF MOISTURE,
# HEAT AND CARBON DIOXIDE FLOW THROUGH POROUS MATERIALS
#
# below refred simply as Saetta 1993.

co2_max = 1.0
caco3_max = 1.0
A = 3.0
E0 = 37400  # J mol^-1
R = 8.3144  # J K^-1 mol^-1
D_co2 = 1.0e-10

alpha1 = 1.0
# Effects on humidity evolution
alpha2 = 1.0
# Effects on temperature
alpha3 = 1.0
# Effects on CO_2 concentration
alpha4 = 1.0


def f1(phi):
    """Factor representing humidity effects,
    i.e. carbonation reaction stops for very low humidities.

    Note
    ----
    Saetta 1993, page 764, formula (5)

    """
    mid = 2.5 * (phi - 0.5)
    return ufl.Max(ufl.Min(mid, 1.0), 0.0)


def f2(co2):
    """CO_2 concentration effects,
    i.e. reaction rate is very low if there is very little CO2.

    Note
    ----
    Saetta 1993, page 763, formula (4)

    """
    return co2 / co2_max


def f3(caco3):
    """CaCO_3 concentration effects,
    Zero-th order term in differential equation,
    reaction stops if products reached maximum concentration.
    Also note concentration of products = CaCO_3 is equal to the
    degree of carbonation reaction.

    Note
    ----
    Saetta 1993, page 763, formula (3)

    """
    return 1.0 - (caco3 / caco3_max)


def f4(temp):
    """Arrhenius activation term

    Note
    ----
    Saetta 1993, page 763, formula inlined

    """
    return A * ufl.exp(- E0 / (R * temp))


def caco31(dt, caco30, phi, co2, temp):
    """CaCO_3 concentration at the next time step,
    This formula is backward Euler applied to CaCO_3 ODE.

    The ODE for evolution of CaCO_3 concentration (== degree of carbonation reaction)
    is linear in CaCO_3 which makes it possible to explicitly express the solution
    at the next time step.
    """
    return ((dt * alpha1 * f1(phi) * f2(co2) * f4(temp) + caco30)
            / (1.0 + dt * alpha1 * f1(phi) * f2(co2) * f4(temp) / caco3_max))


def dot_caco3(dt, caco30, phi, co2, temp):
    """Rate of CaCO_3 concentration"""
    return (caco31(dt, caco30, phi, co2, temp) - caco30) / dt
