import ufl
import numpy

_i, _j = ufl.indices(2)

# Heat params
#
# Volumetric expansion
beta_C = 36.0E-6  # K^{-1}
# Specific heat capacity
C_pc = 0.9E+3  # J kg^{-1} K^{-1}
# Heat conduction
lambda_c = 1.13  # W m^{-1} K^{-1}

room_temp = 293


def inv(A):
    """Matrix invariants"""
    return ufl.tr(A), 1. / 2 * A[_i, _j] * A[_i, _j], ufl.det(A)


def eig(A):
    """Eigenvalues of 3x3 tensor"""
    eps = 1.0e-12

    q = ufl.tr(A) / 3.0
    p1 = 0.5 * (A[0, 1] ** 2 + A[1, 0] ** 2 + A[0, 2] ** 2 + A[2, 0] ** 2 + A[1, 2] ** 2 + A[2, 1] ** 2)
    p2 = (A[0, 0] - q) ** 2 + (A[1, 1] - q) ** 2 + (A[2, 2] - q) ** 2 + 2 * p1
    p = ufl.sqrt(p2 / 6)
    B = (A - q * ufl.Identity(3))
    r = ufl.det(B) / (2 * p ** 3)

    r = ufl.Max(ufl.Min(r, 1.0 - eps), -1.0 + eps)
    phi = ufl.acos(r) / 3.0

    eig0 = ufl.conditional(p2 < eps, q, q + 2 * p * ufl.cos(phi))
    eig2 = ufl.conditional(p2 < eps, q, q + 2 * p * ufl.cos(phi + (2 * numpy.pi / 3)))
    eig1 = ufl.conditional(p2 < eps, q, 3 * q - eig0 - eig2)  # since trace(A) = eig1 + eig2 + eig3

    return eig0, eig1, eig2
