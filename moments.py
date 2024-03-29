# moments.py: functions related to moment equations and constraints in Neustadt's formalism of fuel-optimal trajectories
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy as np
import math
from orbital_mechanics import nu_to_dt, rho_func, phi_harmo, puls_oop_LP, exp_HCW


def Y_oop(e, nu):
    """Function returning the moment-function for out-of-plane dynamics satisfied by the control
    law in the 2-body problem or 3-body problem for L4 and 5.

            Args:
                e (float): eccentricity.
                nu (float): true anomaly.

            Returns:
                Y (np.array): moment-function.

    """

    rho = rho_func(e, nu)
    phi = phi_harmo(-nu, 1.0)
    Y = np.zeros((2, 1))
    Y[0, 0] = phi[0, 1]
    Y[1, 0] = phi[1, 1]

    return Y / rho


def Y_oop_LP123(nu, x, mu):
    """Function returning the moment-function for out-of-plane dynamics satisfied by the control
    law in the circular 3-body problem for L1, 2 or 3.

            Args:
                nu (float): true anomaly.
                x (np.array): coordinates of Lagrange Point around which linearization occurs
                mu (float): ratio of minor mass over total mass.

            Returns:
                Y (np.array): moment-function.

    """

    phi = phi_harmo(-nu, puls_oop_LP(x, mu))
    Y = np.zeros((2, 1))
    Y[0, 0] = phi[0, 1]
    Y[1, 0] = phi[1, 1]

    return Y


def Y_ip_circular2bp(nu):
    """Function returning the moment-function for the equation satisfied by the control
    law in the circular 2-body problem.

            Args:
                nu (float): true anomaly.

            Returns:
                Y (np.array): moment-function.

    """

    Y = np.zeros((4, 2))
    phi = exp_HCW(-nu)
    Y[:, 0:2] = phi[:, 2:4]

    return Y


def Y_ip_elliptical2bp(e, n, nu0, nu):
    """Function returning the moment-function for the equation satisfied by the control
    law in the elliptical 2-body problem.

            Args:
                e (float): eccentricity.
                n (float): mean motion.
                nu0 (float): initial true anomaly.
                nu (float): current true anomaly.

            Returns:
                Y (np.array): moment-function.

    """

    Y = np.zeros((4, 2))
    rho = rho_func(e, nu)
    rho_inv = 1.0 / rho
    s = math.sin(nu)
    c = math.cos(nu)
    dt = nu_to_dt(e, n, nu0, nu)
    e_sq = e * e
    factor = 1.0 / (1.0 - e_sq)
    J = n * dt * math.sqrt(factor * factor * factor)
    Js = J * s
    Jrho = J * rho
    inter = 1.0 + rho_inv
    Y[1, 1] = 3.0 * Jrho - s * e * inter
    Y[1, 0] = -(-3.0 * Js * e + (-rho + 2.0) * inter)
    Y[0, 1] = -(-3.0 * Jrho * e + s * inter)
    Y[0, 0] = 3.0 * e_sq * Js + c - 2.0 * e * rho_inv
    Y[3, 1] = c + (c + e) * rho_inv
    Y[3, 0] = s
    Y[2, 1] = rho
    Y[2, 0] = s * e

    return Y * factor


def Y_ip2bp(e, n, nu0, nu):
    """Wrapper returning the moment-function for the equation satisfied by the control
    law in the 2-body problem.

            Args:
                e (float): eccentricity.
                n (float): mean motion.
                nu0 (float): initial true anomaly.
                nu (float): current true anomaly.

            Returns:
                (np.array): moment-function.

    """

    if e == 0.:
        return Y_ip_circular2bp(nu)
    else:  # elliptical case
        return Y_ip_elliptical2bp(e, n, nu0, nu)


def Y_2bp(e, n, nu0, nu, m):
    """Wrapper returning the moment-function involved in the equation satisfied by the control
    law in the 2-body problem.

            Args:
                e (float): eccentricity.
                n (float): mean motion.
                nu0 (float): initial true anomaly.
                nu (float): current true anomaly.
                m (int): half-dimension of state vector

            Returns:
                (np.array): moment-function.

    """

    # sanity check(s)
    if (m != 1) and (m != 2) and (m != 3):
        raise ValueError("Y_2BP: half-dimension must be 1, 2 or 3")

    if m == 1:
        return Y_oop(e, nu)
    elif m == 2:
        return Y_ip2bp(e, n, nu0, nu)
    else:  # complete dynamics
        Y = np.zeros((6, 3))
        Yoop = Y_oop(e, nu)
        Yip = Y_ip2bp(e, n, nu0, nu)
        Y[0:2, 0:2] = Yip[0:2, 0:2]
        Y[2, 2] = Yoop[0, 0]
        Y[3:5, 0:2] = Yip[2:4, 0:2]
        Y[5, 2] = Yoop[1, 0]
        return Y
