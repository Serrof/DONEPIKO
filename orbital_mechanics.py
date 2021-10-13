# orbital_mechanics.py: functions specific to astrodynamics
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import math
import numpy as np
import utils
from config import conf

# pre-computation for rotation matrix between local orbital frames
# (3-BP co-rotating a.k.a. Hill and LVLH used by Yamanaka & Ankersen)
swap = np.zeros((4, 4))
swap[0, 1] = swap[2, 3] = -1.
swap[1, 0] = swap[3, 2] = 1.
swap_inv = -swap


def _coord_swap(mat):
    """Function that performs the back and forth transformation between local frames (Hill & Y-A)

            Args:
                mat (np.array): input matrix.

            Returns:
                (np.array): new matrix after coordinates have been swapped (and one direction changed).

    """
    return swap_inv @ mat @ swap


def find_L1(mu):
    """Function that iteratively computes the normalized X-coordinate of Lagrange Point 1.

            Args:
                mu (float): ratio of minor mass over total mass.

            Returns:
                (float): coordinate of LP1 on the X-axis of the rotating frame.

    """

    # initialization
    gamma0 = pow(mu * (1.0 - mu), 1.0 / 3.0)
    gamma = gamma0 + 1.0

    nb_iter = 0
    while math.fabs(gamma - gamma0) > conf.params_other["tol_gamma_LP"] and nb_iter < conf.params_other["iter_max_LP"]:
        gamma0 = gamma
        gamma = pow(mu * pow(gamma0 - 1.0, 2) / (3.0 - 2.0 * mu - gamma0 * (3.0 - mu - gamma0)), 1.0 / 3.0)
        nb_iter += 1

    return 1.0 - mu - gamma


def find_L2(mu):
    """Function that iteratively computes the normalized X-coordinate of Lagrange Point 2.

                Args:
                    mu (float): ratio of minor mass over total mass.

                Returns:
                    (float): coordinate of LP2 on the X-axis of the rotating frame.

    """

    # initialization
    gamma0 = pow(mu * (1.0 - mu), 1.0 / 3.0)
    gamma = gamma0 + 1.0

    nb_iter = 0
    while math.fabs(gamma - gamma0) > conf.params_other["tol_gamma_LP"] and nb_iter < conf.params_other["iter_max_LP"]:
        gamma0 = gamma
        gamma = pow(mu * pow(gamma0 + 1.0, 2) / (3.0 - 2.0 * mu + gamma0 * (3.0 - mu + gamma0)), 1.0 / 3.0)
        nb_iter += 1

    return 1.0 - mu + gamma


def find_L3(mu):
    """Function that iteratively computes the normalized X-coordinate of Lagrange Point 3.

                Args:
                    mu (float): ratio of minor mass over total mass.

                Returns:
                    (float): coordinate of LP3 on the X-axis of the rotating frame.

    """

    # initialization
    gamma0 = pow(mu * (1.0 - mu), 1.0 / 3.0)
    gamma = gamma0 + 1.0

    nb_iter = 0
    while math.fabs(gamma - gamma0) > conf.params_other["tol_gamma_LP"] and nb_iter < conf.params_other["iter_max_LP"]:
        gamma0 = gamma
        gamma = pow((1.0 - mu) * pow(gamma0 + 1.0, 2) / (1.0 + 2.0 * mu + gamma0 * (2.0 + mu + gamma0)), 1.0 / 3.0)
        nb_iter += 1

    return -(mu + gamma)


def puls_oop_LP(x, mu_ratio):
    """Function that returns the pulsation of linearized out-of-plane motion in the 3-body problem.

                Args:
                    x (np.array): normalized position of Lagrange Point.
                    mu_ratio (float): ratio of minor mass over total mass.

                Returns:
                    (float): pulsation of linearized out-of-plane motion.

    """

    r1sq = (x[0] + mu_ratio) * (x[0] + mu_ratio) + x[1] * x[1]
    r1 = math.sqrt(r1sq)
    r1cube = r1sq * r1
    r2sq = (x[0] - 1.0 + mu_ratio) * (x[0] - 1.0 + mu_ratio) + x[1] * x[1]
    r2 = math.sqrt(r2sq)
    r2cube = r2sq * r2

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
    inter = math.sqrt(beta1 * beta1 + beta2 * beta2)
    gamma_re = math.sqrt(-beta1 + inter)
    gamma_im = math.sqrt(beta1 + inter)
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

    inter = math.sqrt(1.0 - 27.0 * mu * (1.0 - mu))
    root1 = math.sqrt(0.5 * (1.0 - inter))
    root2 = math.sqrt(0.5 * (1.0 + inter))
    factor = -3. * math.sqrt(3.) * kappa
    inter1 = 9. + 4. * root1 * root1
    inter2 = 9. + 4. * root2 * root2
    a1 = factor / inter1
    a2 = 8. * root1 / inter1
    b1 = -a2
    b2 = a1
    c1 = factor / inter2
    c2 = 8. * root2 / inter2
    d1 = -c2
    d2 = c1
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
        raise ValueError("sma_to_period: semi-major axis cannot be negative")

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
        raise ValueError("period_to_sma: orbital period cannot be negative")

    return ((T * T * planetary_constant) ** (1./3.)) / (2. * math.pi)


def nu_to_dt(e, n, nu0, nu):
    """Function that computes elapsed time given final and initial true anomalies.

                Args:
                    e (float): eccentricity.
                    n (float) : mean motion.
                    nu0 (float) : initial true anomaly in radians.
                    nu (float) : current true anomaly in radians.

                Returns:
                    dt (float): elapsed time between nu and nu0.

    """

    # sanity check(s)
    if (e > 1.0) or (e < 0.0):
        raise ValueError("nu_to_dt: eccentricity must be between 0 and 1")
    if n <= 0.0:
        raise ValueError("nu_to_dt: mean motion cannot be negative")

    if nu == nu0:
        return 0.
    elif nu < nu0:
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
        period = 2.0 * math.pi / n
        if E - E0 < -1.0e-10:  # < 0 with tolerance
            dt += period
        # add revolutions to previous result
        dt += period * math.floor((nu - nu0) / (2.0 * math.pi))

        return dt


def dt_to_nu(e, n, nu0, dt):
    """Function that computes current true anomaly given elapsed time given and initial true anomaly.

                Args:
                    e (float): eccentricity.
                    n (float) : mean motion.
                    nu0 (float) : initial true anomaly in radians.
                    dt (float) : elapsed time since nu0.

                Returns:
                    nu (float): current true anomaly in radians.

    """

    # sanity check(s)
    if (e > 1.0) or (e < 0.0):
        raise ValueError("dt_to_nu: eccentricity must be between 0 and 1")
    if n <= 0.0:
        raise ValueError("dt_to_nu: mean motion cannot be negative")

    if dt == 0.:
        return nu0
    elif (nu0 < 0.) or (nu0 >= 2.0 * math.pi):
        n_mod = math.floor(nu0 / (2.0 * math.pi))
        return 2.0 * math.pi * n_mod + dt_to_nu(e, n, nu0 - 2.0 * math.pi * n_mod, dt)

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
    E = E0
    E_bis = E + 2.0 * conf.params_other["tol_kepler"]
    count = 0
    while math.fabs(E - E_bis) > conf.params_other["tol_kepler"] and count < conf.params_other["iter_max_kepler"]:
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
        if nu < nu0:
            n_rev += 1.
        nu += 2.0 * math.pi * n_rev
    else:  # negative time of flight
        if nu > nu0:
            n_rev += 1.
        nu -= 2.0 * math.pi * n_rev

    return nu


def kep_to_posvel(kep, planetary_constant=1.):
    """Vectorized function that converts Keplerian elements into position-velocity vector.

                Args:
                    kep (numpy.array): Keplerian elements. Units must be coherent. Angles are in radians.
                    planetary_constant (float) : gravitational parameter associated to central body. Default value is 1.

                Returns:
                    (numpy.array): position-velocity vector(s).

    """

    posvel = np.zeros_like(kep)
    sma = kep[0, :]
    ecc = kep[1, :]
    inc = kep[2, :]
    raan = kep[3, :]
    nu = kep[5, :]
    c_i = np.cos(inc)
    s_i = np.sin(inc)
    c_raan = np.cos(raan)
    s_raan = np.sin(raan)
    rho = rho_func(ecc, nu)
    fpa = np.arctan(ecc * np.sin(nu) / rho)
    r = sma * (1. - ecc**2) / rho
    v = np.sqrt(planetary_constant * (2. / r - 1. / sma))
    ang1 = kep[4, :] + nu
    c_ang1 = np.cos(ang1)
    s_ang1 = np.sin(ang1)
    ang2 = ang1 - fpa
    c_ang2 = np.cos(ang2)
    s_ang2 = np.sin(ang2)
    c_i_c_raan = c_i * c_raan
    c_i_s_raan = c_i * s_raan
    posvel[0, :] = r * (c_ang1 * c_raan - s_ang1 * c_i_s_raan)
    posvel[1, :] = r * (c_ang1 * s_raan + s_ang1 * c_i_c_raan)
    posvel[2, :] = r * s_i * s_ang1
    posvel[3, :] = v * (-s_ang2 * c_raan - c_ang2 * c_i_s_raan)
    posvel[4, :] = v * (-s_ang2 * s_raan + c_ang2 * c_i_c_raan)
    posvel[5, :] = v * s_i * c_ang2
    return posvel


def matrix_inertial_local(ref_posvel):
    """Function computing the 3x3 matrix to transform from the inertial frame to the local orbital frame defined by the
    input, such that the X axis is along the position and the Z along the angular momentum.

        Args:
            ref_posvel (np.ndarray): position-velocity (in inertial frame) defining the local orbital frame.

        Returns:
            (np.ndarray): matrix to go from inertial to local frame.

    """

    ux = ref_posvel[:3] / np.linalg.norm(ref_posvel[:3])
    uz = np.cross(ref_posvel[:3], ref_posvel[3:6])
    uz /= np.linalg.norm(uz)
    uy = np.cross(uz, ux)
    return np.array([ux, uy, uz])


def conv_posvel_inertial_local(posvel_inertial, ref_posvel):
    """Function to convert from inertial to the local orbital frame defined by a reference.

            Args:
                posvel_inertial (np.ndarray): position-velocity vector in inertial frame to be converted.
                ref_posvel (np.ndarray): position-velocity (in inertial frame) defining the local orbital frame.

            Returns:
                (np.ndarray): position-velocity vector in local orbital frame.

    """

    rel_posvel = posvel_inertial - ref_posvel
    omega = np.linalg.norm(np.cross(ref_posvel[:3], ref_posvel[3:6])) / np.linalg.norm(ref_posvel[:3]) ** 2
    mat = matrix_inertial_local(ref_posvel)
    posvel_local = np.array(6)
    posvel_local[:3] = mat.dot(rel_posvel[:3])
    posvel_local[3:6] = mat.dot(rel_posvel[3:6]) - np.cross(np.array([0., 0., omega]), posvel_local[:3])
    return posvel_local


def rho_func(e, nu):
    """Function that returns 1 + e cos(nu).

            Args:
                e (float): eccentricity.
                nu (float): true anomaly in radians.

            Returns:
                (float): value of 1 + e cos(nu).

    """

    return 1.0 + e * np.cos(nu)


def phi_harmo(nu, pulsation):
    """Function returning the transition matrix of the harmonic oscillator.

            Args:
                nu (float): true anomaly.
                pulsation (float): pulsation.

            Returns:
                phi (np.array): transition matrix of harmonic oscillator.

    """

    phi = np.zeros((2, 2))
    angle = pulsation * nu
    c = math.cos(angle)
    s = math.sin(angle)
    phi[0, 0] = phi[1, 1] = c
    phi[0, 1] = s / pulsation
    phi[1, 0] = - s * pulsation

    return phi


def transition_oop(x1_bar, nu1, nu2):
    """Function propagating the transformed vector if it follows a harmonic dynamics.

            Args:
                x1_bar (np.array): initial, transformed state vector
                nu1 (float): initial true anomaly.
                nu2 (float): final true anomaly.

            Returns:
                (): transformed vector at nu2.

    """

    try:
        return phi_harmo(nu2 - nu1, 1.0).dot(x1_bar)
    except ValueError as error:
        if len(x1_bar) != 2:
            raise ValueError("TRANSITION_OOP: out-of-plane initial conditions need to be two-dimensional")
        else:
            raise error


def exp_HCW(nu):
    """Function computing the exponential of the true anomaly times the matrix involved in the in-plane
        Hill-Clohessly-Wiltshire equations.

            Args:
                nu (float): true anomaly.

            Returns:
                phi (np.array): transition matrix of in-plane Hill-Clohessly-Wiltshire system.

    """

    phi = np.zeros((4, 4))
    c = math.cos(nu)
    s = math.sin(nu)
    phi[0, 0] = 4.0 - 3.0 * c
    phi[0, 2] = s
    phi[0, 3] = -2.0 * c + 2.0
    phi[1, 0] = -6.0 * nu + 6.0 * s
    phi[1, 1] = 1.0
    phi[1, 2] = -phi[0, 3]
    phi[1, 3] = -3.0 * nu + 4.0 * s
    phi[2, 0] = 3.0 * s
    phi[2, 2] = c
    phi[2, 3] = 2.0 * s
    phi[3, 0] = -6.0 + 6.0 * c
    phi[3, 2] = -phi[2, 3]
    phi[3, 3] = -3.0 + 4.0 * c

    return phi


def phi_YA(e, nu0, nu):
    """Function computing the transition matrix for the in-plane Yamanaka-Ankersen equations.

            Args:
                e (float): eccentricity.
                nu0 (float): initial true anomaly.
                nu (float): current true anomaly.

            Returns:
                (np.array): transition matrix of in-plane Yamanaka-Ankersen system.

    """

    # pre-computations
    rho = rho_func(e, nu)
    rho_sq = rho * rho
    rho_inv = 1.0 / rho
    s = math.sin(nu)
    c = math.cos(nu)
    c2 = math.cos(2.0 * nu)
    sr = s * rho
    cr = c * rho
    dt = nu_to_dt(e, n=1., nu0=nu0, nu=nu)  # dt is inversely proportional to mean motion so J does not depend on it
    J = dt / math.sqrt((1.0 - e * e) ** 3)  # implicitly multiplied by 1

    phi = np.zeros((4, 4))
    phi[0, 0] = 1.0
    phi[0, 1] = -cr * (1.0 + rho_inv)
    phi[0, 2] = sr * (1.0 + rho_inv)
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

    return _coord_swap(phi)  # conversion from one local orbital frame to the other


def transition_ip2bp(x1_bar, e, n, nu1, nu2):
    """Function propagating the transformed in-plane vector for the 2-body problem.

            Args:
                x1_bar (np.array): initial, transformed, in-plane state vector
                e (float): eccentricity.
                n (float): mean motion.
                nu1 (float): initial true anomaly.
                nu2 (float): final true anomaly.

            Returns:
                (np.array): final, transformed, in-plane state vector.

    """

    try:
        if e == 0.:
            phi = exp_HCW(nu2 - nu1)
            return phi.dot(x1_bar)
        else:  # elliptical case
            phi2 = phi_YA(e, nu1, nu2)
            rho1 = rho_func(e, nu1)
            s1 = math.sin(nu1)
            c1 = math.cos(nu1)
            sr1 = s1 * rho1
            cr1 = c1 * rho1
            esq = e * e
            factor = 1.0 / (1.0 - esq)
            phi_inv1 = np.zeros((4, 4))
            phi_inv1[0, 0] = 1.0 / factor
            phi_inv1[0, 1] = 3.0 * e * s1 * (1.0 + 1.0 / rho1)
            phi_inv1[0, 2] = -e * (s1 + sr1)
            phi_inv1[0, 3] = -e * cr1 + 2.0
            phi_inv1[1, 1] = -3.0 * (rho1 + esq) * s1 / rho1
            phi_inv1[1, 2] = s1 + sr1
            phi_inv1[1, 3] = cr1 - 2.0 * e
            phi_inv1[2, 1] = -3.0 * (e + c1)
            phi_inv1[2, 2] = e + c1 + cr1
            phi_inv1[2, 3] = -sr1
            phi_inv1[3, 1] = 3.0 * rho1 - phi_inv1[0, 0]
            phi_inv1[3, 2] = -rho1 * rho1
            phi_inv1[3, 3] = e * sr1
            phi_inv1 *= factor
            phi_inv1 = _coord_swap(phi_inv1)
            Phi = phi2 @ phi_inv1
            return Phi.dot(x1_bar)
    except ValueError as error:
        if len(x1_bar) != 2:
            raise ValueError("TRANSITION_IP2BP: in-plane initial conditions need to be four-dimensional")
        else:
            raise error


def pot_grad(x, mu, slr):
    """Function returning the gradient of the potential in the co-rotating frame in the 2 (mu=0) or 3-body problem.

            Args:
                x (np.array): spacecraft's transformed coordinates
                mu (float): ratio of minor mass over total mass.
                slr (float): semilatus rectum of reference elliptical orbit

            Returns:
                gr (np.array): gradient of the potential relative to conservative forces.

    """

    r1sq = (x[0] + slr * mu) * (x[0] + slr * mu) + x[1] * x[1]
    r2sq = (x[0] - slr + slr * mu) * (x[0] - slr + slr * mu) + x[1] * x[1]
    if len(x) > 2:
        inter = x[2] * x[2]
        r1sq += inter
        r2sq += inter
    r1 = math.sqrt(r1sq)
    r1cube = r1sq * r1
    r2 = math.sqrt(r2sq)
    r2cube = r2sq * r2
    pcube = slr * slr * slr
    slrmu = slr * mu
    gr = [-pcube * ((1. - mu) * (x[0] + slrmu) / r1cube + mu * (x[0] - slr + slrmu) / r2cube),
          -pcube * ((1. - mu) * x[1] / r1cube + mu * x[1] / r2cube)]
    if len(x) > 2:
        gr.append(-pcube * ((1. - mu) * x[2] / r1cube + mu * x[2] / r2cube))

    return gr


def state_deriv_nonlin(x, nu, ecc, x_eq, mu, slr):
    """Function computing the derivative of the transformed state vector w.r.t. the true anomaly for the non-linear motion.

            Args:
                x (np.array): out-of-plane transformed vector.
                nu (float): true anomaly.
                ecc (float): eccentricity of reference orbit.
                x_eq (np.array): coordinates of equilibrium point.
                mu (float): ratio of minor mass over total mass.
                slr (float): semi-latus rectum of reference orbit.

            Returns:
                (List[float]): state derivative.

    """

    half_dim = int(len(x) / 2)
    Y = x[0:half_dim] + slr * x_eq[0:half_dim]
    grad = pot_grad(Y, mu, slr)
    rho = rho_func(ecc, nu)

    if half_dim == 2:
        return [x[2], x[3], 2. * x[3] + (Y[0] + grad[0]) / rho, -2. * x[2] + (Y[1] + grad[1]) / rho]
    else:  # complete dynamics
        return [x[3], x[4], x[5], 2. * x[4] + (Y[0] + grad[0]) / rho, -2. * x[3] +
                (Y[1] + grad[1]) / rho, (grad[2] - ecc * math.cos(nu) * Y[2]) / rho]


def oop_state_deriv(x, nu, ecc, x_eq, mu):
    """Function computing the derivative of the out-of-plane transformed state vector w.r.t. the true anomaly.

            Args:
                x (List[float]): out-of-plane transformed vector.
                nu (float): true anomaly.
                ecc (float): eccentricity of reference orbit.
                x_eq (np.array): coordinates of equilibrium point.
                mu (float): ratio of minor mass over total mass.

            Returns:
                (List[float]): state derivative.

    """

    if mu != 0.:
        pulsation = puls_oop_LP(x_eq, mu)
        factor = (pulsation * pulsation + ecc * math.cos(nu)) / rho_func(ecc, nu)
        return [x[1], -x[0] * factor]

    else:  # out-of-plane elliptical 2-body problem
        return [x[1], -x[0]]


def Hessian_ip2bp(x):
    """Function computing the Hessian matrix of conservative forces' potential (gravity + non-inertial) in in-plane R2BP.

            Args:
                x (np.array): in-plane coordinates

            Returns:
                H (np.array): Hessian (2x2) of conservative forces' potential.

    """

    x0sq = x[0] * x[0]
    x1sq = x[1] * x[1]
    r1sq = x0sq + x1sq
    r1cube = math.sqrt(r1sq) * r1sq
    r1pow5 = r1cube * r1sq
    H = np.zeros((2, 2))
    H[0, 0] = -1.0 + 1.0 / r1cube - 3.0 * x0sq / r1pow5
    H[0, 1] = -3.0 * x[1] * x[0] / r1pow5
    H[1, 0] = H[0, 1]
    H[1, 1] = -1.0 + 1.0 / r1cube - 3.0 * x1sq / r1pow5

    return H


def Hessian_ip3bp(x, mu):
    """Function computing the Hessian matrix of conservative forces' potential (gravity + non-inertial) in in-plane R3BP.

            Args:
                x (np.array): in-plane coordinates
                mu (float): ratio of minor mass over total mass.

            Returns:
                H (np.array): Hessian (2x2) of conservative forces' potential.

    """

    x1sq = x[1] * x[1]
    r1sq = (x[0] + mu) * (x[0] + mu) + x1sq
    r1cube = math.sqrt(r1sq) * r1sq
    r1pow5 = r1cube * r1sq
    r2sq = (x[0] - 1.0 + mu) * (x[0] - 1.0 + mu) + x1sq
    r2cube = math.sqrt(r2sq) * r2sq
    r2pow5 = r2cube * r2sq
    H = np.zeros((2, 2))
    H[0, 0] = -1.0 + (1.0 - mu) / r1cube - 3.0 * (1.0 - mu) * (x[0] + mu) * (x[0] + mu) / r1pow5 + mu / r2cube - 3.0 * mu * (x[0] - 1.0 + mu) * (x[0] - 1.0 + mu) / r2pow5
    H[0, 1] = -3.0 * (1.0 - mu) * x[1] * (x[0] + mu) / r1pow5 - 3.0 * mu * x[1] * (x[0] - 1.0 + mu) / r2pow5
    H[1, 0] = H[0, 1]
    H[1, 1] = -1.0 + (1.0 - mu) / r1cube - 3.0 * (1.0 - mu) * x1sq / r1pow5 + mu / r2cube - 3.0 * mu * x1sq / r2pow5

    return H


def ip_state_deriv(x, nu, ecc, x_eq, mu):
    """Function computing the derivative of the in-plane transformed state vector w.r.t. the true anomaly.

            Args:
                x (List[float]): in-plane transformed vector.
                nu (float): true anomaly.
                ecc (float): eccentricity of reference orbit.
                x_eq (np.array): coordinates of equilibrium point.
                mu (float): ratio of minor mass over total mass.

            Returns:
                (List[float]): state derivative.

    """
    Hessian = Hessian_ip2bp(x_eq) if mu == 0 else Hessian_ip3bp(x_eq, mu)

    rho = rho_func(ecc, nu)
    return [x[2], x[3], 2. * x[3] - (Hessian[0, 0] * x[0] + Hessian[0, 1] * x[1]) / rho, -2. * x[2] -
            (Hessian[1, 0] * x[0] + Hessian[1, 1] * x[1]) / rho]


def complete_state_deriv(x, nu, ecc, x_eq, mu):
    """Function computing the derivative of the complete (in-plane + out-of-plane) transformed state vector w.r.t.
    the true anomaly.

            Args:
                x (List[float]): complete transformed vector.
                nu (float): true anomaly.
                ecc (float): eccentricity of reference orbit.
                x_eq (np.array): coordinates of equilibrium point.
                mu (float): ratio of minor mass over total mass.

            Returns:
                (np.array): state derivative.

    """
    (x_ip, x_oop) = utils.unstack_state(x)
    ip_deriv = ip_state_deriv(x_ip, nu, ecc, x_eq, mu)
    oop_deriv = oop_state_deriv(x_oop, nu, ecc, x_eq, mu)
    return utils.stack_state(ip_deriv, oop_deriv)
