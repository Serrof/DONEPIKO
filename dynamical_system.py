# dynamical_system.py: class for the dynamical models
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy
from moments import *
from numpy import linalg
import utils
import integrators
import orbital_mechanics


class DynamicalSystem:
    """Class implementing the dynamics of the restricted 2- or 3-body problem.

    """

    def __init__(self, mu, ecc, period, sma, Li=None):
        """Constructor.

                Args:
                    mu (float): ratio of minor mass over total mass.
                    ecc (float): eccentricity.
                    period (float): orbital period.
                    sma (float): semi-major axis (must be consistent with period)
                    Li (int): index of Lagrange Point (used only if mu != 0)

        """
        if mu < 0. or mu >= 1.:
            print('DynamicalSystem: mass ratio must be between 0 and 1')
        if ecc < 0. or ecc >= 1.:
            print('DynamicalSystem: eccentricity must be between 0 and 1')
        if period <= 0.:
            print('DynamicalSystem: orbital period must be non negative')
        if mu != 0. and Li != 1 and Li != 2 and Li != 3 and Li != 4 and Li != 5:
            print('DynamicalSystem: for 3-body problem, valid index of Lagrange Point must be provided')

        self.mu = mu
        self.ecc = ecc
        self.period = period
        self.mean_motion = 2. * math.pi / period
        self.sma = sma
        self.Li = Li

        if mu != 0.:
            # set normalized coordinates of Lagrange point of interest
            if self.Li == 1 or self.Li == 2 or self.Li == 3:
                if self.Li == 1:
                    self.x_eq_normalized = numpy.array((find_L1(self.mu), 0.0, 0.0))
                elif self.Li == 2:
                    self.x_eq_normalized = numpy.array((find_L2(self.mu), 0.0, 0.0))
                elif self.Li == 3:
                    self.x_eq_normalized = numpy.array((find_L3(self.mu), 0.0, 0.0))
                puls = puls_oop_LP(self.x_eq_normalized, self.mu)
                (gamma_re, gamma_im, c, k) = inter_L123(puls * puls)
                A = numpy.array([[1.0, 1.0, 1.0, 0.0], [c, -c, 0.0, k], [gamma_re, -gamma_re, 0.0, gamma_im],
                                 [gamma_re * c, gamma_re * c, - gamma_im * k, 0.0]])
                self._A_inv = numpy.linalg.inv(A)
            else:  # Lagrange Point 4 or 5
                if Li == 4:
                    self.x_eq_normalized = numpy.array((0.5 * (1.0 - 2.0 * self.mu), math.sqrt(3.) / 2., 0., 0.))
                    kappa = 1.0 - 2.0 * self.mu
                else:  # Li = 5
                    self.x_eq_normalized = numpy.array((0.5 * (1.0 - 2.0 * self.mu), -math.sqrt(3.) / 2., 0., 0.))
                    kappa = -1.0 + 2.0 * self.mu
                root1, root2, a1, a2, b1, b2, c1, c2, d1, d2 = inter_L45(self.mu, kappa)
                A = numpy.array([[1.0, 0.0, 1.0, 0.0], [a1, a2, c1, c2],
                                 [0.0, root1, 0.0, root2], [b1 * root1, b2 * root1, d1 * root2, d2 * root2]])
                self._A_inv = numpy.linalg.inv(A)

        else:  # restricted 2-body problem
            self.x_eq_normalized = numpy.array([1.0, 0.0, 0.0])
            self._A_inv = None

    def transformation(self, x, nu):
        """Function converting original state vector to a modified space where propagation is simpler.

                Args:
                    x (numpy.array): state vector.
                    nu (float): current true anomaly.

                Returns:
                    x_bar (numpy.array): transformed state vector.

        """

        half_dim = len(x) / 2
        inter = 1.0 - self.ecc * self.ecc
        inter2 = math.sqrt(inter * inter * inter) / self.mean_motion
        rho = rho_func(self.ecc, nu)
        x_bar = numpy.zeros(2 * half_dim)
        for k in range(0, len(x)):
            if k < half_dim:
                x_bar[k] = rho * x[k]
            else:  # indices for velocity components of state vector
                x_bar[k] = -self.ecc * math.sin(nu) * x[k-half_dim] + inter2 * x[k] / rho

        return x_bar

    def transformation_inv(self, x_bar, nu):
        """Function converting back the transformed state vector in its original, physical space.

                Args:
                    x_bar (numpy.array): transformed state vector.
                    nu (float): current true anomaly.

                Returns:
                    x (numpy.array): original state vector.

        """

        half_dim = len(x_bar) / 2
        inter = 1.0 - self.ecc * self.ecc
        inter2 = math.sqrt(inter * inter * inter) / self.mean_motion
        rho = rho_func(self.ecc, nu)
        x = numpy.zeros(2 * half_dim)
        for k in range(0, len(x)):
            if k < half_dim:
                x[k] = x_bar[k] / rho
            else:  # indices for velocity components of state vector
                x[k] = (self.ecc * math.sin(nu) * x_bar[k-half_dim] + rho * x_bar[k]) / inter2

        return x

    def integrate_phi_inv(self, nus, half_dim):
        """Function integrating over the true anomaly the inverse of the fundamental transition matrix associated to the
        transformed state vector.

                Args:
                    nus (list): grid of true anomalies.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    outputs (list): list of inverse fundamental transition matrices integrated on input grid.

        """

        # sanity check(s)
        if (half_dim != 3) and (half_dim != 2) and (half_dim != 1):
            print('integrate_Y: half-dimension of state vector should be 1, 2 or 3')

        if half_dim == 1:

            if (self.mu != 0.) and (self.Li == 1 or self.Li == 2 or self.Li == 3):
                pulsation = orbital_mechanics.puls_oop_LP(self.x_eq_normalized, self.mu)

                if self.ecc == 0.:

                    def func(nu, x):  # right-hand side function for integration
                        return [x[2] * pulsation * pulsation, x[3] * pulsation * pulsation, -x[0], -x[1]]

                else:  # elliptical case 3-body near L1, 2 or 3

                    def func(nu, x):  # right-hand side function for integration
                        factor = (pulsation * pulsation + self.ecc * math.cos(nu)) / rho_func(self.ecc, nu)
                        return [x[2] * factor, x[3] * factor, -x[0], -x[1]]

            else:  # out-of-plane elliptical 2-body problem or 3-body near L4 and 5

                def func(nu, x):  # right-hand side function for integration
                    return [x[2], x[3], -x[0], -x[1]]

        else:  # in-plane or complete dynamics

            if self.mu == 0.:
                H = orbital_mechanics.Hessian_ip2bp(self.x_eq_normalized)
            else:  # restricted three-body case
                H = orbital_mechanics.Hessian_ip3bp(self.x_eq_normalized, self.mu)

            if half_dim == 2:

                A = numpy.array(
                    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, -2.0, 0.0]])

            else:  # complete dynamics

                pulsation = 1.0
                if self.mu != 0.0 and (self.Li == 1 or self.Li == 2 or self.Li == 3):
                    pulsation = orbital_mechanics.puls_oop_LP(self.x_eq_normalized, self.mu)

                A = numpy.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                                 [0.0, 0.0, 0.0, -2.0, 0.0, 0.0],
                                 [0.0, 0.0, -pulsation * pulsation, 0.0, 0.0, 0.0]])

            def func(nu, x):  # right-hand side function for integration
                rho = orbital_mechanics.rho_func(self.ecc, nu)
                if half_dim == 3:
                    A[5, 2] = -(pulsation * pulsation + self.ecc * math.cos(nu)) / rho
                A[half_dim: half_dim + 2, 0:2] = -H / rho
                x_matrix = utils.vector_to_square_matrix(x, 2 * half_dim)
                f_matrix = -x_matrix.dot(A)  # right-hand side of matrix differential equation satisfied by phi^-1
                return utils.square_matrix_to_vector(f_matrix, 2 * half_dim)

        integ = integrators.RK4(func)
        outputs = []
        IC_matrix = numpy.eye(2 * half_dim)
        outputs.append(IC_matrix)
        IC_vector = utils.square_matrix_to_vector(IC_matrix, 2 * half_dim)  # initial conditions of matrix system turned into a vector for integration
        n_step = int(math.ceil(math.fabs(nus[1] - nus[0]) / plot_params["h_min"]))

        for k in range(0, len(nus)-1):
            (state_hist, nu_hist) = integ.integrate(nus[k], nus[k+1], IC_vector, n_step)

            if half_dim == 1:  # temporary fix for the out-of-plane only case
                outputs.append(numpy.transpose(utils.vector_to_square_matrix(state_hist[-1], 2 * half_dim)))
            else:  # in-plane or complete dynamics
                outputs.append(utils.vector_to_square_matrix(state_hist[-1], 2 * half_dim))

            IC_vector = state_hist[-1]  # old final condition becomes initial one

        return outputs

    def evaluate_Y(self, nu, half_dim):
        """Wrapper returning the moment-function involved in the equation satisfied by the control law.

                Args:
                    nu (float): current true anomaly.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (numpy.array): moment-function.

        """

        # sanity check(s)
        if (half_dim != 3) and (half_dim != 2) and (half_dim != 1):
            print('evaluate_Y: half-dimension of state vector should be 1, 2 or 3')

        if self.mu == 0.:
            return Y_2bp(self.ecc, self.mean_motion, 0., nu, half_dim)
        else:  # restricted 3-body problem
            return self._Y_3bp(self.ecc, self.mean_motion, nu, half_dim)

    def _Y_3bp(self, e, n, nu, half_dim):
        """Wrapper returning the moment-function involved in the equation satisfied by the control
        law in the in-plane 3-body problem.

                Args:
                    e (float): eccentricity.
                    n (float): mean motion
                    nu (float): current true anomaly.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (numpy.array): moment-function.

        """

        # sanity check(s)
        if (half_dim != 1) and (half_dim != 2) and (half_dim != 3):
            print('Y_3BP: half-dimension must be 1, 2 or 3')

        if half_dim == 1:
            if self.Li == 1 or self.Li == 2 or self.Li == 3:
                if e != 0.:
                    print('Y_3BP: analytical case not coded yet')
                else:  # circular case
                    return Y_oop_LP123(nu, self.x_eq_normalized, self.mu)
            else:  # Lagrange Point 4 or 5
                return Y_oop(e, nu)
        elif half_dim == 2:
            return self._Y_ip3bp_ds(nu)
        else:  # complete dynamics
            Y = numpy.zeros((6, 3))
            if self.Li == 1 or self.Li == 2 or self.Li == 3:
                Yoop = Y_oop_LP123(nu, self.x_eq_normalized, self.mu)
            else:   # Lagrange Point 4 or 5
                Yoop = Y_oop(e, nu)
            Yip = self._Y_ip3bp_ds(nu)
            Y[0:2, 0:2] = Yip[0:2, 0:2]
            Y[2, 2] = Yoop[0, 0]
            Y[3:5, 0:2] = Yip[2:4, 0:2]
            Y[5, 2] = Yoop[1, 0]
            return Y

    def _Y_ip3bp_ds(self, nu):
        """Wrapper returning the moment-function for the equation satisfied by the control
        law in the in-plane 3-body problem.

                Args:
                    nu (float): current true anomaly.

                Returns:
                    Y (numpy.array): moment-function.

        """

        if self.ecc == 0.:
            Y = numpy.zeros((4, 2))
            if (self.Li == 1) or (self.Li == 2) or (self.Li == 3):
                phi = self.exp_LP123(-nu)
            elif self.Li == 4:
                phi = self.exp_LP45(-nu, 1. - 2. * self.mu)
            else:  # Li = 5
                phi = self.exp_LP45(-nu, -1. + 2. * self.mu)
            Y[:, 0:2] = phi[:, 2:4]

            return Y
        else:  # elliptical case
            print('_Y_ip3bp_ds: analytical elliptical case not coded yet')

    def transition_ip3bp(self, x1_bar, nu1, nu2):
        """Wrapper for the propagation of the transformed vector in the in-plane 3-body problem.

                Args:
                    x1_bar (numpy.array): initial transformed, in-plane state vector.
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.

                Returns:
                    (numpy.array): final transformed, in-plane state vector.

        """

        # sanity check(s)
        if len(x1_bar) != 4:
            print('TRANSITION_IP3BP: in-plane initial conditions need to be four-dimensional')
        if (self.ecc >= 1.0) or (self.ecc < 0.0):
            print('TRANSITION_IP3BP: eccentricity must be larger or equal to 0 and strictly less than 1')

        if self.ecc == 0.:
            if self.Li == 1 or self.Li == 2 or self.Li == 3:
                phi = self.exp_LP123(nu2 - nu1)
            elif self.Li == 4:
                phi = self.exp_LP45(nu2 - nu1, 1. - 2. * self.mu)
            else:  # Li = 5
                phi = self.exp_LP45(nu2 - nu1, -1. + 2. * self.mu)
            return phi.dot(x1_bar)
        else:  # elliptical case
            print('TRANSITION_IP3BP: analytical elliptical case not coded yet')

    def exp_LP123(self, nu):
        """Function computing the exponential of the true anomaly times the matrix involved in the in-plane
        linearized equations around L1, 2 or 3.

                Args:
                    nu (float): true anomaly.

                Returns:
                    phi (numpy.array): transition matrix.

        """

        puls = puls_oop_LP(self.x_eq_normalized, self.mu)
        (gamma_re, gamma_im, c, k) = inter_L123(puls * puls)
        line1 = [math.exp(gamma_re * nu), math.exp(-gamma_re * nu), math.cos(gamma_im * nu), math.sin(gamma_im * nu)]
        line2 = [c * math.exp(gamma_re * nu), -c * math.exp(-gamma_re * nu),
                 -k * math.sin(gamma_im * nu), k * math.cos(gamma_im * nu)]
        line3 = [gamma_re * math.exp(gamma_re * nu), -gamma_re * math.exp(-gamma_re * nu),
                 -gamma_im * math.sin(gamma_im * nu), gamma_im * math.cos(gamma_im * nu)]
        line4 = [gamma_re * c * math.exp(gamma_re * nu), gamma_re * c * math.exp(-gamma_re * nu),
                 -gamma_im * k * math.cos(gamma_im * nu), -gamma_im * k * math.sin(gamma_im * nu)]
        phi = numpy.array([line1, line2, line3, line4]).dot(self._A_inv)

        return phi

    def exp_LP45(self, nu, kappa):
        """Function computing the exponential of the true anomaly times the matrix involved in the in-plane
        linearized equations around L4 or 5.

                Args:
                    nu (float): true anomaly.
                    kappa (float): parameter equal to 1-2mu for L4, -1+2mu for L5.

                Returns:
                    phi (numpy.array): transition matrix.

        """

        root1, root2, a1, a2, b1, b2, c1, c2, d1, d2 = inter_L45(self.mu, kappa)
        line1 = [math.cos(root1 * nu), math.sin(root1 * nu), math.cos(root2 * nu), math.sin(root2 * nu)]
        line2 = [a1 * math.cos(root1 * nu) + b1 * math.sin(root1 * nu),
                 a2 * math.cos(root1 * nu) + b2 * math.sin(root1 * nu),
                 c1 * math.cos(root2 * nu) + d1 * math.sin(root2 * nu),
                 c2 * math.cos(root2 * nu) + d2 * math.sin(root2 * nu)]
        line3 = [-root1 * math.sin(root1 * nu), root1 * math.cos(root1 * nu),
                 -root2 * math.sin(root2 * nu), root2 * math.cos(root2 * nu)]
        line4 = [-root1 * a1 * math.sin(root1 * nu) + root1 * b1 * math.cos(root1 * nu),
                 -root1 * a2 * math.sin(root1 * nu) + root1 * b2 * math.cos(root1 * nu),
                 -root2 * c1 * math.sin(root2 * nu) + root2 * d1 * math.cos(root2 * nu),
                 -root2 * c2 * math.sin(root2 * nu) + root2 * d2 * math.cos(root2 * nu)]
        phi = numpy.array([line1, line2, line3, line4]).dot(self._A_inv)

        return phi

    def propagate(self, nu1, nu2, x1):
        """Wrapper for the propagation of the state vector.

                Args:
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.
                    x1 (numpy.array): initial state vector.

                Returns:
                    (numpy.array): final state vector.

        """

        half_dim = len(x1) / 2

        if nu1 == nu2:
            x2 = numpy.zeros(half_dim * 2)
            for k in range(0, len(x1)):
                x2[k] = x1[k]
            return x2
        else:  # initial and final true anomalies are different
            if half_dim == 1:
                x1_bar = self.transformation(x1, nu1)
                x2_bar = None
                if self.mu != 0 and (self.Li == 1 or self.Li == 2 or self.Li == 3):
                    if self.ecc == 0.:
                        phi = phi_harmo(nu2 - nu1, puls_oop_LP(self.x_eq_normalized, self.mu))
                        x2_bar = phi . dot(x1_bar)
                    else:  # elliptical case
                        print('PROPAGATE: analytical 3-body elliptical out-of-plane near L1, 2 and 3 not coded yet')
                else:  # restricted 2-body problem or L4/5
                    x2_bar = transition_oop(x1_bar, nu1, nu2)
                return self.transformation_inv(x2_bar, nu2)
            elif half_dim == 2:
                x1_bar = self.transformation(x1, nu1)
                x2_bar = None
                if self.mu == 0.:
                    x2_bar = transition_ip2bp(x1_bar, self.ecc, self.mean_motion, nu1, nu2)
                else:  # in-plane 3-body problem
                    if self.ecc == 0:
                        x2_bar = self.transition_ip3bp(x1_bar, nu1, nu2)
                    else:
                        print('PROPAGATE: analytical 3-body elliptical in-plane case not coded yet')
                return self.transformation_inv(x2_bar, nu2)
            else:  # complete dynamics
                x_ip1, x_oop1 = utils.unstack_state(x1)
                x_oop2 = self.propagate(nu1, nu2, x_oop1)
                x_ip2 = self.propagate(nu1, nu2, x_ip1)
                return utils.stack_state(x_ip2, x_oop2)

    def compute_rhs(self, BC, analytical):
        """Function that computes right-hand side of moment equation.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.
                    analytical (bool): set to true for analytical propagation of motion, false for integration.

                Returns:
                    u (numpy.array): right-hand side of moment equation.

        """

        factor = 1.0 - self.ecc * self.ecc
        multiplier = self.mean_motion / math.sqrt(factor * factor * factor)
        x1 = self.transformation(BC.x0, BC.nu0)
        x2 = self.transformation(BC.xf, BC.nuf)

        if analytical:
            u = numpy.zeros(2 * BC.half_dim)

            if BC.half_dim == 1:
                if (self.mu != 0.) and (self.Li == 1 or self.Li == 2 or self.Li == 3):
                    if self.ecc == 0.:
                        u += phi_harmo(-BC.nuf, puls_oop_LP(self.x_eq_normalized, self.mu)) . dot(x2)
                        u -= phi_harmo(-BC.nu0, puls_oop_LP(self.x_eq_normalized, self.mu)) . dot(x1)
                    else:
                        print('compute_rhs: analytical 3-body elliptical out-of-plane near L1, 2 and 3 not coded yet')
                else:  # out-of-plane elliptical 2-body problem or 3-body near L4 and 5
                    u += phi_harmo(-BC.nuf, 1.0).dot(x2)
                    u -= phi_harmo(-BC.nu0, 1.0).dot(x1)
                u[0] *= multiplier
                u[1] *= multiplier

            elif BC.half_dim == 2:
                if self.mu == 0.:
                    if self.ecc == 0.:
                        u += exp_HCW(-BC.nuf) . dot(x2)
                        u -= exp_HCW(-BC.nu0) . dot(x1)
                    else:  # elliptical case
                        M = phi_YA(self.ecc, self.mean_motion, 0., BC.nuf)
                        u += linalg.inv(M) . dot(x2)
                        M = phi_YA(self.ecc, self.mean_motion, 0., BC.nu0)
                        u -= linalg.inv(M) . dot(x1)
                    for i in range(0, len(u)):
                        u[i] *= multiplier
                else:  # in-plane restricted 3-body problem
                    if self.ecc == 0.:
                        if (self.Li == 1) or (self.Li == 2) or (self.Li == 3):
                            u += self.exp_LP123(-BC.nuf) . dot(x2)
                            u -= self.exp_LP123(-BC.nu0) . dot(x1)
                        elif self.Li == 4:
                            u += self.exp_LP45(-BC.nuf, 1. - 2. * self.mu) . dot(x2)
                            u -= self.exp_LP45(-BC.nu0, 1. - 2. * self.mu) . dot(x1)
                        else:  # Li = 5
                            u += self.exp_LP45(-BC.nuf, -1. + 2. * self.mu) . dot(x2)
                            u -= self.exp_LP45(-BC.nu0, -1. + 2. * self.mu) . dot(x1)
                    else:  # elliptical case
                        print('compute_rhs: analytical elliptical 3-body problem in-plane dynamics case not coded yet')

                    for i in range(0, len(u)):
                        u[i] *= multiplier

            else:  # complete dynamics
                x0_ip, x0_oop = utils.unstack_state(BC.x0)
                xf_ip, xf_oop = utils.unstack_state(BC.xf)
                BC_ip = utils.BoundaryConditions(BC.nu0, BC.nuf, x0_ip, xf_ip)
                BC_oop = utils.BoundaryConditions(BC.nu0, BC.nuf, x0_oop, xf_oop)
                z_oop = self.compute_rhs(BC_oop, analytical=True)
                z_ip = self.compute_rhs(BC_ip, analytical=True)
                u = utils.stack_state(z_ip, z_oop)

        else:  # numerical integration

            matrices = self.integrate_phi_inv([BC.nu0, BC.nuf], BC.half_dim)

            IC_matrix = matrices[0]
            FC_matrix = matrices[-1]

            u = (FC_matrix.dot(x2) - IC_matrix.dot(x1)) * multiplier

        return u
