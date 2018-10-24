# orbital_mechanics.py: functions specific to astrodynamics
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import math
import numpy
from const_params import *

# pre-computation for rotation matrix between local orbital frames
# (3-BP co-rotating and LVLH used by Yamanaka & Ankersen)
swap = numpy.zeros((4, 4))
swap[0, 1] = -1.
swap[1, 0] = 1.
swap[2, 3] = -1.
swap[3, 2] = 1.
swap_inv = numpy.zeros((4, 4))
swap_inv[0, 1] = 1.
swap_inv[1, 0] = -1.
swap_inv[2, 3] = 1.
swap_inv[3, 2] = -1.


def find_L1(mu):
    """Function that iteratively computes the normalized X-coordinate of Lagrange Point 1.

            Args:
                mu (float): ratio of minor mass over total mass.

            Returns:
                (float): coordinate of LP1 on the X-axis of the rotating frame.

    """

    # initialization
    gamma0 = math.pow(mu * (1.0 - mu), 1.0 / 3.0)
    gamma = gamma0 + 1.0

    iter = 0
    while math.fabs(gamma - gamma0) > other_params["tol_gamma_LP"] and iter < other_params["iter_max_LP"]:
        gamma0 = gamma
        gamma = pow(mu * pow(gamma0 - 1.0, 2) / (3.0 - 2.0 * mu - gamma0 * (3.0 - mu - gamma0)), 1.0 / 3.0)
        iter += 1

    return 1.0 - mu - gamma


def find_L2(mu):
    """Function that iteratively computes the normalized X-coordinate of Lagrange Point 2.

                Args:
                    mu (float): ratio of minor mass over total mass.

                Returns:
                    (float): coordinate of LP2 on the X-axis of the rotating frame.

    """

    # initialization
    gamma0 = math.pow(mu * (1.0 - mu), 1.0 / 3.0)
    gamma = gamma0 + 1.0

    iter = 0
    while math.fabs(gamma - gamma0) > other_params["tol_gamma_LP"] and iter < other_params["iter_max_LP"]:
        gamma0 = gamma
        gamma = pow(mu * pow(gamma0 + 1.0, 2) / (3.0 - 2.0 * mu + gamma0 * (3.0 - mu + gamma0)), 1.0 / 3.0)
        iter += 1

    return 1.0 - mu + gamma


def find_L3(mu):
    """Function that iteratively computes the normalized X-coordinate of Lagrange Point 3.

                Args:
                    mu (float): ratio of minor mass over total mass.

                Returns:
                    (float): coordinate of LP3 on the X-axis of the rotating frame.

    """

    # initialization
    gamma0 = math.pow(mu * (1.0 - mu), 1.0 / 3.0)
    gamma = gamma0 + 1.0

    iter = 0
    while math.fabs(gamma - gamma0) > other_params["tol_gamma_LP"] and iter < other_params["iter_max_LP"]:
        gamma0 = gamma
        gamma = pow((1.0 - mu) * pow(gamma0 + 1.0, 2) / (1.0 + 2.0 * mu + gamma0 * (2.0 + mu + gamma0)), 1.0 / 3.0)
        iter += 1

    return - mu - gamma


def puls_oop_LP(x, mu_ratio):
    """Function that returns the pulsation of linearized out-of-plane motion in the 3-body problem.

                Args:
                    x (numpy.array): normalized position of Lagrange Point.
                    mu_ratio (float): ratio of minor mass over total mass.

                Returns:
                    (float): pulsation of linearized out-of-plane motion.

    """

    r1sq = (x[0] + mu_ratio) * (x[0] + mu_ratio) + x[1] * x[1]
    r1 = math.sqrt(r1sq)
    r1cube = r1 * r1 * r1
    r2sq = (x[0] - 1.0 + mu_ratio) * (x[0] - 1.0 + mu_ratio) + x[1] * x[1]
    r2 = math.sqrt(r2sq)
    r2cube = r2 * r2 * r2

    return math.sqrt((1.0 - mu_ratio) / r1cube + mu_ratio / r2cube)


def inter_L123(puls2):
    """Function computing intermediate parameters for computing the in-plane transition matrix
    for Lagrange Points 1, 2 or 3.

                Args:
                    puls2 (float): squared pulsation of ouf-of-plane dynamics near LP1, 2 or 3.

                Returns:
                    gamma_re (float): real eigenvalue of linear matrix of in-plane dynamics.
                    gamma_im (float): imaginary eigenvalue of linear matrix of in-plane dynamics.
                    c (float): parameter necessary for computing the transition matrix.
                    k (float): parameter necessary for computing the transition matrix.

    """
    beta1 = 1. - puls2 / 2.
    beta2 = math.sqrt(2. * puls2 * puls2 - puls2 - 1.)
    gamma_re = math.sqrt(-beta1 + math.sqrt(beta1 * beta1 + beta2 * beta2))
    gamma_im = math.sqrt(beta1 + math.sqrt(beta1 * beta1 + beta2 * beta2))
    c = (gamma_re * gamma_re - 1. - 2. * puls2) / (2. * gamma_re)
    k = (gamma_im * gamma_im + 1. + 2. * puls2) / (2. * gamma_im)
    return gamma_re, gamma_im, c, k


def inter_L45(mu, kappa):
    """Function computing intermediate parameters for computing the in-plane transition matrix
    for Lagrange Points 4 or 5.

                Args:
                    mu (float): ratio of minor mass over total mass.
                    kappa (float): parameter equal to 1-2mu for L4, -1+2mu for L5.

                Returns:
                    root1 (float): first eigenvalue of linear matrix of in-plane dynamics.
                    root2 (float): second eigenvalue of linear matrix of in-plane dynamics.
                    a1 (float): parameter necessary for computing the transition matrix.
                    a2 (float): parameter necessary for computing the transition matrix.
                    b1 (float): parameter necessary for computing the transition matrix.
                    b2 (float): parameter necessary for computing the transition matrix.
                    c1 (float): parameter necessary for computing the transition matrix.
                    c2 (float): parameter necessary for computing the transition matrix.
                    d1 (float): parameter necessary for computing the transition matrix.
                    d2 (float): parameter necessary for computing the transition matrix.

    """

    root1 = math.sqrt(0.5 * (1.0 - math.sqrt(1.0 - 27.0 * mu * (1.0 - mu))))
    root2 = math.sqrt(0.5 * (1.0 + math.sqrt(1.0 - 27.0 * mu * (1.0 - mu))))
    a1 = -3. * math.sqrt(3.) * kappa / (9. + 4. * root1 * root1)
    a2 = 8. * root1 / (9. + 4. * root1 * root1)
    b1 = -8. * root1 / (9. + 4. * root1 * root1)
    b2 = -3. * math.sqrt(3.) * kappa / (9. + 4. * root1 * root1)
    c1 = -3. * math.sqrt(3.) * kappa / (9. + 4. * root2 * root2)
    c2 = 8. * root2 / (9. + 4. * root2 * root2)
    d1 = -8. * root2 / (9. + 4. * root2 * root2)
    d2 = -3. * math.sqrt(3.) * kappa / (9. + 4. * root2 * root2)
    return root1, root2, a1, a2, b1, b2, c1, c2, d1, d2


def sma_to_period(a, planetary_constant):
    """Function that computes the period from the semi-major axis according to Kepler's third law.

                Args:
                    a (float): semi-major axis.
                    planetary_constant (float): gravitational constant of central mass (unit must be consistent with a).

                Returns:
                    (float): orbital period.

    """

    # sanity check(s)
    if a <= 0.0:
        print('sma_to_period: semi-major axis cannot be negative')

    return 2. * math.pi * math.sqrt(a * a * a / planetary_constant)


def period_to_sma(T, planetary_constant):
    """Function that computes the semi-major axis from the period according to Kepler's third law.

                Args:
                    T (float): orbital period.
                    planetary_constant (float): gravitational constant of central mass (unit must be consistent with a).

                Returns:
                    (float): semi-major axis.

    """
    # sanity check(s)
    if T <= 0.0:
        print('period_to_sma: orbital period cannot be negative')

    return math.pow(T * T * planetary_constant, 1./3.) / (2. * math.pi)


def nu_to_dt(e, n, nu0, nu):
    """Function that computes elapsed time given final and initial true anomalies.

                Args:
                    e (float): eccentricity.
                    n (float) : mean motion.
                    nu0 (float) : initial true anomaly.
                    nu (float) : current true anomaly.

                Returns:
                    dt (float): elapsed time between nu and nu0.

    """

    # sanity check(s)
    if (e > 1.0) or (e < 0.0):
        print('nu_to_dt: eccentricity must be between 0 and 1')
    if n <= 0.0:
        print('nu_to_dt: mean motion cannot be negative')

    if nu < nu0:
        return -nu_to_dt(e, n, nu, nu0)
    else:  # current true anomaly is posterior to initial one
        # convert initial and final true anomalies into eccentric ones (modulo 2 pi)
        inter = math.sqrt((1.0 - e) / (1.0 + e))
        E0 = 2.0 * math.atan(inter * math.tan(nu0 / 2.0))
        if E0 < 0.0:
            E0 += 2.0 * math.pi
        E = 2.0 * math.atan(inter * math.tan(nu / 2.0))
        if E < 0.0:
            E += 2.0 * math.pi
        # compute elapsed time (modulo the period) via Kepler equation
        dt = (E - E0 - e * (math.sin(E) - math.sin(E0))) / n
        if E - E0 < -1.0e-10:
            dt += 2.0 * math.pi / n
        # add revolutions to previous result
        dt += (2.0 * math.pi / n) * math.floor((nu - nu0) / (2.0 * math.pi))

        return dt


def dt_to_nu(e, n, nu0, dt):
    """Function that computes current true anomaly given elapsed time given and initial true anomaly.

                Args:
                    e (float): eccentricity.
                    n (float) : mean motion.
                    nu0 (float) : initial true anomaly.
                    dt (float) : elapsed time since nu0.

                Returns:
                    nu (float): current true anomaly.

    """

    # sanity check(s)
    if (e > 1.0) or (e < 0.0):
        print('dt_to_nu: eccentricity must be between 0 and 1')
    if n <= 0.0:
        print('dt_to_nu: mean motion cannot be negative')

    # convert initial true anomaly into eccentric one (modulo 2 pi)
    inter = math.sqrt((1.0 - e) / (1.0 + e))
    E0 = 2.0 * math.atan(inter * math.tan(nu0 / 2.0))
    if E0 < 0.0:
        E0 += 2.0 * math.pi
    # compute elapsed time modulo the period
    period = 2.0 * math.pi / n
    n_rev = math.floor(math.fabs(dt) / period)
    dt_bis = dt
    if dt >= 0.0:
        dt_bis -= period * n_rev
    else:  # negative time of flight
        dt_bis += period * n_rev
    # compute final eccentric anomaly by solving Kepler equation via Newton-Raphson algorithm
    E = nu0
    E_bis = E + 2.0 * other_params["tol_kepler"]
    count = 0
    while math.fabs(E - E_bis) > other_params["tol_kepler"] and count < other_params["iter_max_kepler"]:
        E_bis = E
        E -= (n * dt_bis - E + E0 + e * math.sin(E) - e * math.sin(E0)) / (-1.0 + e * math.cos(E))
        count += 1
    # compute final true anomaly modulo 2 pi
    nuf = 2.0 * math.atan(math.tan(E / 2.0) / inter)
    if nuf < 1.e-10:  # tolerance for negativity
        nuf += 2.0 * math.pi
    # add revolutions to previous result
    nu = nuf
    if dt >= 0.0:
        nu += 2.0 * math.pi * n_rev
        if nu < nu0:
            nu += 2.0 * math.pi
    else:  # negative time of flight
        nu -= 2.0 * math.pi * n_rev
        if nu > nu0:
            nu -= 2.0 * math.pi

    return nu


def rho_func(e, nu):
    """Function that returns 1 + e cos(nu).

            Args:
                e (float): eccentricity.
                nu (float): true anomaly.

            Returns:
                (float): value of 1 + e cos(nu).

    """

    return 1.0 + e * math.cos(nu)


def phi_harmo(nu, pulsation):
    """Function returning the transition matrix of the harmonic oscillator.

            Args:
                nu (float): true anomaly.
                pulsation (float): pulsation.

            Returns:
                phi (numpy.array): transition matrix of harmonic oscillator.

    """

    phi = numpy.zeros((2, 2))
    c = math.cos(pulsation * nu)
    s = math.sin(pulsation * nu)
    phi[0, 0] = c
    phi[0, 1] = s / pulsation
    phi[1, 0] = - s * pulsation
    phi[1, 1] = c

    return phi


def transition_oop(x1_bar, nu1, nu2):
    """Function propagating the transformed vector if it follows a harmonic dynamics.

            Args:
                x1_bar (numpy.array): initial, transformed state vector
                nu1 (float): initial true anomaly.
                nu2 (float): final true anomaly.

            Returns:
                (): transformed vector at nu2.

    """

    # sanity check(s)
    if len(x1_bar) != 2:
        print('TRANSITION_OOP: out-of-plane initial conditions need to be two-dimensional')

    return phi_harmo(nu2 - nu1, 1.0) . dot(x1_bar)


def exp_HCW(nu):
    """Function computing the exponential of the true anomaly times the matrix involved in the in-plane
        Hill-Clohessly-Wiltshire equations.

            Args:
                nu (float): true anomaly.

            Returns:
                phi (numpy.array): transition matrix of in-plane Hill-Clohessly-Wiltshire system.

    """

    phi = numpy.zeros((4, 4))
    c = math.cos(nu)
    s = math.sin(nu)
    phi[0, 0] = 4.0 - 3.0 * c
    phi[0, 2] = s
    phi[0, 3] = -2.0 * c + 2.0
    phi[1, 0] = -6.0 * nu + 6.0 * s
    phi[1, 1] = 1.0
    phi[1, 2] = 2.0 * c - 2.0
    phi[1, 3] = -3.0 * nu + 4.0 * s
    phi[2, 0] = 3.0 * s
    phi[2, 2] = c
    phi[2, 3] = 2.0 * s
    phi[3, 0] = -6.0 + 6.0 * c
    phi[3, 2] = -2.0 * s
    phi[3, 3] = -3.0 + 4.0 * c

    return phi


def phi_YA(e, n, nu0, nu):
    """Function computing the transition matrix for the in-plane Yamanaka-Ankersen equations.

            Args:
                e (float): eccentricity.
                n (float): mean motion.
                nu0 (float): initial true anomaly.
                nu (float): current true anomaly.

            Returns:
                (numpy.array): transition matrix of in-plane Yamanaka-Ankersen system.

    """

    # sanity check(s)
    if (e >= 1.0) or (e < 0.0):
        print('PHI_YA: eccentricity must be larger or equal to 0 and strictly less than 1')

    # pre-computations
    rho = rho_func(e, nu)
    rho_sq = rho * rho
    s = math.sin(nu)
    c = math.cos(nu)
    c2 = math.cos(2.0 * nu)
    sr = s * rho
    cr = c * rho
    dt = nu_to_dt(e, n, nu0, nu)
    J = dt * n / math.sqrt((1.0 - e * e) * (1.0 - e * e) * (1.0 - e * e))

    phi = numpy.zeros((4, 4))
    phi[0, 0] = 1.0
    phi[0, 1] = -cr * (1.0 + 1.0 / rho)
    phi[0, 2] = sr * (1.0 + 1.0 / rho)
    phi[0, 3] = 3.0 * J * rho_sq
    phi[1, 1] = sr
    phi[1, 2] = cr
    phi[1, 3] = 2.0 - 3.0 * e * sr * J
    phi[2, 1] = 2.0 * sr
    phi[2, 2] = 2.0 * cr - e
    phi[2, 3] = 3.0 * (1.0 - 2.0 * e * sr * J)
    phi[3, 1] = c + e * c2
    phi[3, 2] = -(s + e * math.sin(2.0 * nu))
    phi[3, 3] = -3.0 * e * (J * phi[3, 1] + sr / rho_sq)

    return swap_inv.dot(phi.dot(swap))  # conversion from one local orbital frame to the other


def transition_ip2bp(x1_bar, e, n, nu1, nu2):
    """Function propagating the transformed in-plane vector for the 2-body problem.

            Args:
                x1_bar (numpy.array): initial, transformed, in-plane state vector
                e (float): eccentricity.
                n (float): mean motion.
                nu1 (float): initial true anomaly.
                nu2 (float): final true anomaly.

            Returns:
                (numpy.array): final, transformed, in-plane state vector.

    """

    # sanity check(s)
    if len(x1_bar) != 4:
        print('TRANSITION_IP2BP: in-plane initial conditions need to be four-dimensional')
    if (e >= 1.0) or (e < 0.0):
        print('TRANSITION_IP2BP: eccentricity must be larger or equal to 0 and strictly less than 1')
    if n < 0.0:
        print('TRANSITION_IP2BP: mean motion cannot be smaller than 0')

    if e == 0.:
        phi = exp_HCW(nu2 - nu1)
        return phi . dot(x1_bar)
    else:  # elliptical case
        phi2 = phi_YA(e, n, nu1, nu2)
        rho1 = rho_func(e, nu1)
        s1 = math.sin(nu1)
        c1 = math.cos(nu1)
        sr1 = s1 * rho1
        cr1 = c1 * rho1
        factor = 1.0 / (1.0 - e * e)
        phi_inv1 = numpy.zeros((4, 4))
        phi_inv1[0, 0] = 1.0
        phi_inv1[0, 1] = (3.0 * e * s1 * (1.0 + 1.0 / rho1)) * factor
        phi_inv1[0, 2] = -e * (s1 + sr1) * factor
        phi_inv1[0, 3] = (-e * cr1 + 2.0) * factor
        phi_inv1[1, 1] = (-3.0 * (rho1 + e * e) * s1 / rho1) * factor
        phi_inv1[1, 2] = (s1 + sr1) * factor
        phi_inv1[1, 3] = (cr1 - 2.0 * e) * factor
        phi_inv1[2, 1] = (-3.0 * (e + c1)) * factor
        phi_inv1[2, 2] = (e + c1 + cr1) * factor
        phi_inv1[2, 3] = -sr1 * factor
        phi_inv1[3, 1] = 3.0 * rho1 * factor - 1.0
        phi_inv1[3, 2] = -rho1 * rho1 * factor
        phi_inv1[3, 3] = e * sr1 * factor
        phi_inv1 = swap.dot(phi_inv1.dot(swap_inv))
        Phi = phi2 . dot(phi_inv1)
        return Phi . dot(x1_bar)


def grad(x, mu, slr):
    """Function returning the gravitational acceleration in the co-rotating frame in the 2 (mu=0) or 3-body problem.

            Args:
                x (numpy.array): spacecraft's coordinates
                mu (float): ratio of minor mass over total mass.
                slr (float): semilatus rectum of reference elliptical orbit

            Returns:
                gr (numpy.array): gravitational acceleration.

    """

    r1sq = (x[0] + slr * mu) * (x[0] + slr * mu) + x[1] * x[1]
    r2sq = (x[0] - slr + slr * mu) * (x[0] - slr + slr * mu) + x[1] * x[1]
    if len(x) > 2:
        r1sq += x[2] * x[2]
        r2sq += x[2] * x[2]
    r1 = math.sqrt(r1sq)
    r1cube = r1sq * r1
    r2 = math.sqrt(r2sq)
    r2cube = r2sq * r2
    pcube = slr * slr * slr
    gr = numpy.zeros(len(x))
    gr[0] = -pcube * ((1. - mu) * (x[0] + slr * mu) / r1cube + mu * (x[0] - slr + slr * mu) / r2cube)
    gr[1] = -pcube * ((1. - mu) * x[1] / r1cube + mu * x[1] / r2cube)
    if len(x) > 2:
        gr[2] = -pcube * ((1. - mu) * x[2] / r1cube + mu * x[2] / r2cube)

    return gr


def Hessian_ip2bp(x):
    """Function computing the Hessian matrix of conservative forces' potential (gravity + non-inertial) in in-plane R2BP.

            Args:
                x (numpy.array): spacecraft's coordinates

            Returns:
                H (numpy.array): Hessian (2x2) of conservative forces' potential.

    """

    r1sq = x[0] * x[0] + x[1] * x[1]
    r1 = math.sqrt(r1sq)
    r1cube = r1 * r1 * r1
    H = numpy.zeros((2, 2))
    H[0, 0] = -1.0 + 1.0 / r1cube - 3.0 * x[0] * x[0] / (r1cube * r1sq)
    H[0, 1] = -3.0 * x[1] * x[0] / (r1cube * r1sq)
    H[1, 0] = H[0, 1]
    H[1, 1] = -1.0 + 1.0 / r1cube - 3.0 * x[1] * x[1] / (r1cube * r1sq)

    return H


def Hessian_ip3bp(x, mu):
    """Function computing the Hessian matrix of conservative forces' potential (gravity + non-inertial) in in-plane R3BP.

            Args:
                x (numpy.array): spacecraft's coordinates
                mu (float): ratio of minor mass over total mass.

            Returns:
                H (numpy.array): Hessian (2x2) of conservative forces' potential.

    """

    r1sq = (x[0] + mu) * (x[0] + mu) + x[1] * x[1]
    r1 = math.sqrt(r1sq)
    r1cube = r1 * r1 * r1
    r2sq = (x[0] - 1.0 + mu) * (x[0] - 1.0 + mu) + x[1] * x[1]
    r2 = math.sqrt(r2sq)
    r2cube = r2 * r2 * r2
    H = numpy.zeros((2, 2))
    H[0, 0] = -1.0 + (1.0 - mu) / r1cube - 3.0 * (1.0 - mu) * (x[0] + mu) * (x[0] + mu) / (r1cube * r1sq) + mu / r2cube - 3.0 * mu * (x[0] - 1.0 + mu) * (x[0] - 1.0 + mu) / (r2cube * r2sq)
    H[0, 1] = -3.0 * (1.0 - mu) * x[1] * (x[0] + mu) / (r1cube * r1sq) - 3.0 * mu * x[1] * (x[0] - 1.0 + mu) / (r2cube * r2sq)
    H[1, 0] = H[0, 1]
    H[1, 1] = -1.0 + (1.0 - mu) / r1cube - 3.0 * (1.0 - mu) * x[1] * x[1] / (r1cube * r1sq) + mu / r2cube - 3.0 * mu * x[1] * x[1] / (r2cube * r2sq)

    return H