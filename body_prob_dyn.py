# dynamical_system.py: set of classes for the restricted 2 and 3-body problems
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy as np
import math
from moments import rho_func, Y_2bp, Y_oop, Y_oop_LP123, phi_harmo
from numpy import linalg
import utils
from orbital_mechanics import nu_to_dt, dt_to_nu, puls_oop_LP, exp_HCW, Hessian_ip2bp, Hessian_ip3bp, inter_L45, \
    inter_L123, find_L1, find_L2, find_L3, phi_YA, transition_ip2bp, transition_oop, state_deriv_nonlin
import dynamical_system
from abc import ABCMeta, abstractmethod


class BodyProbParams(dynamical_system.DynParams):
    """Class handling the dynamical parameters of a restricted 2 or 3 body problem.

                Attributes:
                    mu (float): ratio of minor mass over total mass.
                    ecc (float): eccentricity.
                    period (float): orbital period.
                    slr (float): semi-latus rectum.
                    mean_motion(float): mean motion.
                    sma (float): semi-major axis (must be consistent with period)
                    Li (int): index of Lagrange Point (required only if mu != 0)

    """

    def __init__(self, mu, ecc, period, sma, Li=None):
        """Constructor.

                Args:
                    mu (float): ratio of minor mass over total mass.
                    ecc (float): eccentricity.
                    period (float): orbital period.
                    sma (float): semi-major axis (must be consistent with period)
                    Li (int): index of Lagrange Point (relevant only if mu != 0)

        """
        # sanity checks
        if mu < 0. or mu >= 1.:
            raise ValueError('BodyProbParams: mass ratio must be between 0 and 1')
        if ecc < 0. or ecc >= 1.:
            raise ValueError('BodyProbParams: eccentricity must be between 0 and 1')
        if period <= 0.:
            raise ValueError('BodyProbParams: orbital period must be non negative')
        if mu != 0. and Li not in [1, 2, 3, 4, 5]:
            raise ValueError('BodyProbParams: for 3-body problem, valid index of Lagrange Point must be provided')

        self.mu = mu
        self.ecc = ecc
        self.period = period
        self.sma = sma
        self.slr = self.sma * (1. - self.ecc * self.ecc)
        self.mean_motion = 2. * math.pi / period
        self.Li = Li

    def copy(self):
        """Function returning a copy of the object.

                Returns:
                    (BodyProbParams): copied object.

        """

        return BodyProbParams(self.mu, self.ecc, self.period, self.sma, self.Li)


class BodyProbDyn(dynamical_system.DynamicalSystem):
    """Abstract class to implement the dynamics of the restricted 2- or 3-body problem.

                Attributes:
                    params (BodyProbParams): parameters characterizing the dynamical system
                    x_eq_normalized (np.array): normalized coordinates of equilibrium point.

    """

    __metaclass__ = ABCMeta

    def __init__(self, mu, ecc, period, sma, Li=None):
        """Constructor.

                Args:
                    mu (float): ratio of minor mass over total mass.
                    ecc (float): eccentricity.
                    period (float): orbital period.
                    sma (float): semi-major axis (must be consistent with period)
                    Li (int): index of Lagrange Point (required only if mu != 0)

        """

        dynamical_system.DynamicalSystem.__init__(self, "")
        self.params = BodyProbParams(mu, ecc, period, sma, Li)

        def conv(nu0, t0, nu):
            return nu_to_dt(self.params.ecc, self.params.mean_motion, nu0, nu)
        self.convToAlterIndVar = conv

        def convInv(nu0, t0, t):
            return dt_to_nu(self.params.ecc, self.params.mean_motion, nu0, t)
        self.convFromAlterIndVar = convInv

        self.x_eq_normalized = None

    def transformation(self, x, nu):
        """Function converting original state vector to a modified space where propagation is simpler.

                Args:
                    x (np.array): state vector.
                    nu (float): current true anomaly.

                Returns:
                    x_bar (np.array): transformed state vector.

        """

        half_dim = int(len(x) / 2)
        inter = 1.0 - self.params.ecc * self.params.ecc
        inter2 = math.sqrt(inter * inter * inter) / self.params.mean_motion
        rho = rho_func(self.params.ecc, nu)
        es = self.params.ecc * math.sin(nu)
        x_bar = [rho * x[k] for k in range(0, half_dim)]
        x_bar.extend([-es * x[k] + inter2 * x[k + half_dim] / rho for k in range(0, half_dim)])

        return x_bar

    def transformation_inv(self, x_bar, nu):
        """Function converting back the transformed state vector in its original, physical space.

                Args:
                    x_bar (np.array): transformed state vector.
                    nu (float): current true anomaly.

                Returns:
                    x (np.array): original state vector.

        """

        half_dim = int(len(x_bar) / 2)
        inter = 1.0 - self.params.ecc * self.params.ecc
        inter2 = math.sqrt(inter * inter * inter) / self.params.mean_motion
        rho = rho_func(self.params.ecc, nu)
        es = self.params.ecc * math.sin(nu)
        x = [x_bar[k] / rho for k in range(0, half_dim)]
        x.extend([(es * x_bar[k] + rho * x_bar[k + half_dim]) / inter2 for k in range(0, half_dim)])

        return x

    def evaluate_state_deriv_nonlin(self, nu, x):
        """Function returning the derivative of the transformed state vector w.r.t. the independent variable in the
        non-linearized dynamics.

                Args:
                    nu (float): value of independent variable.
                    x (np.array): transformed state vector.

                Returns:
                    (np.array): derivative of transformed state vector in non-linear dynamics.

        """
        return state_deriv_nonlin(x, nu, self.params.ecc, self.x_eq_normalized, self.params.mu, self.params.slr)

    def matrix_linear(self, nu, half_dim):
        """Function returning the matrix appearing in the differential system satisfied by the transformed state vector
         in the linearized dynamics.

                Args:
                    nu (float): value of true anomaly.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): matrix for transformed state equation in linearized dynamics.

        """

        M = np.zeros((2 * half_dim, 2 * half_dim))
        M[0: half_dim, half_dim: 2 * half_dim] = np.eye(half_dim)

        if half_dim == 1:

            if (self.params.mu != 0.) and (self.params.Li in [1, 2, 3]):

                pulsation = puls_oop_LP(self.x_eq_normalized, self.params.mu)
                M[1, 0] = -(pulsation * pulsation + self.params.ecc * math.cos(nu)) / rho_func(self.params.ecc, nu)

            else:  # out-of-plane elliptical 2-body problem or 3-body near L4 and 5

                M[1, 0] = -1.0

        else:  # in-plane or complete dynamics

            M[half_dim: half_dim + 2, half_dim: half_dim + 2] = np.array([[0.0, 2.0], [-2.0, 0.0]])

            if self.params.mu == 0.:
                H = Hessian_ip2bp(self.x_eq_normalized)
            else:  # restricted three-body case
                H = Hessian_ip3bp(self.x_eq_normalized, self.params.mu)

            rho = rho_func(self.params.ecc, nu)
            M[half_dim: half_dim + 2, 0:2] = -H / rho

            if half_dim == 3 and self.params.mu != 0.0 and self.params.Li in [1, 2, 3]:
                pulsation = puls_oop_LP(self.x_eq_normalized, self.params.mu)
                M[5, 2] = -(pulsation * pulsation + self.params.ecc * math.cos(nu)) / rho

        return M

    def integrate_Y(self, nus, half_dim):
        """Function integrating over the true anomaly the moment-function.

                Args:
                    nus (List[float]): grid of true anomalies.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    outputs (np.array): moment-function integrated on input grid.

        """
        matrices = self.integrate_phi_inv(nus, half_dim)
        Ys = np.zeros((2 * half_dim, half_dim * len(nus)))
        for k, matrix in enumerate(matrices):
            Ys[:, half_dim * k: half_dim * (k + 1)] = matrix[:, half_dim: 2 * half_dim] / \
                                                      rho_func(self.params.ecc, nus[k])
        return Ys

    @abstractmethod
    def evaluate_Y(self, nu, half_dim):
        """Wrapper returning the moment-function involved in the equation satisfied by the control law.

                Args:
                    nu (float): current true anomaly.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): moment-function.

        """

        raise NotImplementedError

    @abstractmethod
    def _transition_ip(self, x1, nu1, nu2):
        """Function returning the in-plane initial vector propagated to the final true anomaly.

                Args:
                    x1 (np.array): in-plane initial transformed state vector.
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.

        """
        raise NotImplementedError

    @abstractmethod
    def _rhs_ip(self, nu1, nu2, x1, x2):
        """Wrapper for the right-hand side of the moment-equation in the in-plane dynamics.

                Args:
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.
                    x1 (np.array): initial transformed state vector.
                    x2 (np.array): final transformed state vector.

        """
        raise NotImplementedError

    def propagate(self, nu1, nu2, x1):
        """Wrapper for the propagation of the state vector.

                Args:
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.
                    x1 (np.array): initial state vector.

                Returns:
                    (np.array): final state vector.

        """

        half_dim = int(len(x1) / 2)

        if nu1 == nu2:
            return np.array(x1)
        else:  # initial and final true anomalies are different
            if half_dim == 1:
                x1_bar = self.transformation(x1, nu1)
                if self.params.mu != 0 and self.params.Li in [1, 2, 3]:
                    if self.params.ecc == 0.:
                        phi = phi_harmo(nu2 - nu1, puls_oop_LP(self.x_eq_normalized, self.params.mu))
                        x2_bar = phi.dot(x1_bar)
                    else:  # elliptical case
                        raise NotImplementedError('PROPAGATE: analytical 3-body elliptical out-of-plane near L1, 2 and 3 not coded yet')
                else:  # restricted 2-body problem or L4/5
                    x2_bar = transition_oop(x1_bar, nu1, nu2)
                return self.transformation_inv(x2_bar, nu2)
            elif half_dim == 2:
                x1_bar = self.transformation(x1, nu1)
                if self.params.mu == 0. or self.params.ecc == 0:
                    x2_bar = self._transition_ip(x1_bar, nu1, nu2)
                    return self.transformation_inv(x2_bar, nu2)
                else:
                    raise NotImplementedError('PROPAGATE: analytical 3-body elliptical in-plane case not coded yet')
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
                    u (np.array): right-hand side of moment equation.

        """

        factor = 1.0 - self.params.ecc * self.params.ecc
        multiplier = self.params.mean_motion / math.sqrt(factor * factor * factor)
        x1 = self.transformation(BC.x0, BC.nu0)
        x2 = self.transformation(BC.xf, BC.nuf)

        if analytical:

            if BC.half_dim == 1:
                if self.params.mu != 0. and self.params.Li in [1, 2, 3]:
                    if self.params.ecc == 0.:
                        u = phi_harmo(-BC.nuf, puls_oop_LP(self.x_eq_normalized, self.params.mu)).dot(x2)
                        u -= phi_harmo(-BC.nu0, puls_oop_LP(self.x_eq_normalized, self.params.mu)).dot(x1)
                    else:
                        raise NotImplementedError('compute_rhs: analytical 3-body elliptical out-of-plane near L1, 2 and 3 not coded yet')
                else:  # out-of-plane elliptical 2-body problem or 3-body near L4 and 5
                    u = phi_harmo(-BC.nuf, 1.0).dot(x2)
                    u -= phi_harmo(-BC.nu0, 1.0).dot(x1)
                u *= multiplier

            elif BC.half_dim == 2:
                if self.params.mu == 0. or self.params.ecc == 0.:
                    u = self._rhs_ip(BC.nu0, BC.nuf, x1, x2) * multiplier
                else:  # elliptical in-plane restricted 3-body problem case
                    raise NotImplementedError('compute_rhs: analytical elliptical 3-body problem in-plane dynamics case not coded yet')

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


class RestriTwoBodyProb(BodyProbDyn):
    """Class implementing the dynamics of the restricted 2-body problem.

    """

    def __init__(self, ecc, period, sma):
        """Constructor.

                Args:
                    ecc (float): eccentricity.
                    period (float): orbital period.
                    sma (float): semi-major axis (must be consistent with period)

        """

        # call to parent constructor
        BodyProbDyn.__init__(self, 0., ecc, period, sma)
        self.name = "Restricted 2-body problem"

        self.x_eq_normalized = np.array([1.0, 0.0, 0.0])

    @classmethod
    def init_from_params(cls, params):
        """Method to instantiate the class from an object of its parameters' class.

                Args:
                    params (BodyProbParams): object of parameter's class.

        """

        return cls(params.ecc, params.period, params.sma)

    @classmethod
    def init_with_mean_motion(cls, ecc, mm, sma):
        """Method to instantiate the class from the mean motion instead of the orbital period.

                Args:
                    ecc (float): eccentricity.
                    mm (float): mean motion.
                    sma (float): semi-major axis (must be consistent with period)

        """
        params = BodyProbParams(0., ecc, 2. * math.pi / mm, sma)
        return RestriTwoBodyProb.init_from_params(params)

    def copy(self):
        """Function returning a copy of the object.

                Returns:
                    (RestriTwoBodyProb): copied object.

        """

        return RestriTwoBodyProb.init_from_params(self.params)

    def evaluate_Y(self, nu, half_dim):
        """Function returning the moment-function involved in the equation satisfied by the control law.

                Args:
                    nu (float): current true anomaly.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): moment-function.

        """

        # sanity check(s)
        if (half_dim != 3) and (half_dim != 2) and (half_dim != 1):
            raise ValueError('evaluate_Y: half-dimension of state vector should be 1, 2 or 3')

        return Y_2bp(self.params.ecc, self.params.mean_motion, 0., nu, half_dim)

    def _transition_ip(self, x1, nu1, nu2):
        """Function returning the in-plane initial vector propagated to the final true anomaly.

                Args:
                    x1 (np.array): in-plane initial transformed state vector.
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.

        """

        return transition_ip2bp(x1, self.params.ecc, self.params.mean_motion, nu1, nu2)

    def _rhs_ip(self, nu1, nu2, x1, x2):
        """Wrapper for the right-hand side of the moment-equation in the in-plane restricted 2-body problem.

                Args:
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.
                    x1 (np.array): initial transformed state vector.
                    x2 (np.array): final transformed state vector.

        """

        if self.params.ecc == 0.:
            u = exp_HCW(-nu2).dot(x2)
            u -= exp_HCW(-nu1).dot(x1)
        else:  # elliptical case
            M = phi_YA(self.params.ecc, self.params.mean_motion, 0., nu2)
            u = linalg.inv(M).dot(x2)
            M = phi_YA(self.params.ecc, self.params.mean_motion, 0., nu1)
            u -= linalg.inv(M).dot(x1)
        return u


class RestriThreeBodyProb(BodyProbDyn):
    """Class implementing the dynamics of the restricted 3-body problem.

            Attributes:
                    _A_inv (np.array): intermediate matrix used for some calculations.

    """

    def __init__(self, mu, ecc, period, sma, Li):
        """Constructor.

                Args:
                    mu (float): ratio of minor mass over total mass.
                    ecc (float): eccentricity.
                    period (float): orbital period.
                    sma (float): semi-major axis (must be consistent with period)
                    Li (int): index of Lagrange Point (used only if mu != 0)

        """

        # call to parent constructor
        BodyProbDyn.__init__(self, mu, ecc, period, sma, Li)
        self.name = "Restricted 3-body problem"

        # set normalized coordinates of Lagrange point of interest
        if self.params.Li in [1, 2, 3]:
            if self.params.Li == 1:
                self.x_eq_normalized = np.array((find_L1(self.params.mu), 0.0, 0.0))
            elif self.params.Li == 2:
                self.x_eq_normalized = np.array((find_L2(self.params.mu), 0.0, 0.0))
            else:  # Li = 3
                self.x_eq_normalized = np.array((find_L3(self.params.mu), 0.0, 0.0))
            puls = puls_oop_LP(self.x_eq_normalized, self.params.mu)
            (gamma_re, gamma_im, c, k) = inter_L123(puls * puls)
            A = np.array([[1.0, 1.0, 1.0, 0.0], [c, -c, 0.0, k], [gamma_re, -gamma_re, 0.0, gamma_im],
                             [gamma_re * c, gamma_re * c, - gamma_im * k, 0.0]])
            self._A_inv = np.linalg.inv(A)
        else:  # Lagrange Point 4 or 5
            if Li == 4:
                self.x_eq_normalized = np.array((0.5 * (1.0 - 2.0 * self.params.mu), math.sqrt(3.) / 2., 0., 0.))
                kappa = 1.0 - 2.0 * self.params.mu
            else:  # Li = 5
                self.x_eq_normalized = np.array((0.5 * (1.0 - 2.0 * self.params.mu), -math.sqrt(3.) / 2., 0., 0.))
                kappa = -1.0 + 2.0 * self.params.mu
            root1, root2, a1, a2, b1, b2, c1, c2, d1, d2 = inter_L45(self.params.mu, kappa)
            A = np.array([[1.0, 0.0, 1.0, 0.0], [a1, a2, c1, c2],
                             [0.0, root1, 0.0, root2], [b1 * root1, b2 * root1, d1 * root2, d2 * root2]])
            self._A_inv = np.linalg.inv(A)

    @classmethod
    def init_from_params(cls, params):
        """Method to instantiate the class from an object of its parameters' class.

                Args:
                    params (BodyProbParams): object of parameter's class.

        """

        return cls(params.mu, params.ecc, params.period, params.sma, params.Li)

    @classmethod
    def init_with_mean_motion(cls, mu, ecc, mm, sma, Li):
        """Method to instantiate the class from the mean motion instead of the orbital period.

                Args:
                    mu (float): ratio of minor mass over total mass.
                    ecc (float): eccentricity.
                    mm (float): mean motion.
                    sma (float): semi-major axis (must be consistent with period)
                    Li (int): index of Lagrange Point (used only if mu != 0)

        """

        params = BodyProbParams(mu, ecc, 2. * math.pi / mm, sma, Li)
        return RestriThreeBodyProb.init_from_params(params)

    def copy(self):
        """Function returning a copy of the object.

                Returns:
                    (RestriThreeBodyProb): copied object.

        """

        return RestriThreeBodyProb.init_from_params(self.params)

    def evaluate_Y(self, nu, half_dim):
        """Function returning the moment-function involved in the equation satisfied by the control law.

                Args:
                    nu (float): current true anomaly.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): moment-function.

        """

        # sanity check(s)
        if (half_dim != 3) and (half_dim != 2) and (half_dim != 1):
            raise ValueError('evaluate_Y: half-dimension of state vector should be 1, 2 or 3')

        return self._Y_3bp(self.params.ecc, self.params.mean_motion, nu, half_dim)

    def _Y_3bp(self, e, n, nu, half_dim):
        """Wrapper returning the moment-function involved in the equation satisfied by the control
        law in the in-plane 3-body problem.

                Args:
                    e (float): eccentricity.
                    n (float): mean motion
                    nu (float): current true anomaly.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): moment-function.

        """

        # sanity check(s)
        if (half_dim != 1) and (half_dim != 2) and (half_dim != 3):
            raise ValueError('_Y_3BP: half-dimension must be 1, 2 or 3')

        if half_dim == 1:
            if self.params.Li in [1, 2, 3]:
                if e != 0.:
                    raise NotImplementedError('_Y_3BP: analytical case not coded yet')
                else:  # circular case
                    return Y_oop_LP123(nu, self.x_eq_normalized, self.params.mu)
            else:  # Lagrange Point 4 or 5
                return Y_oop(e, nu)
        elif half_dim == 2:
            return self._Y_ip3bp_ds(nu)
        else:  # complete dynamics
            Y = np.zeros((6, 3))
            if self.params.Li in [1, 2, 3]:
                Yoop = Y_oop_LP123(nu, self.x_eq_normalized, self.params.mu)
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
                    Y (np.array): moment-function evaluated at given anomaly.

        """

        if self.params.ecc == 0.:
            Y = np.zeros((4, 2))
            if self.params.Li in [1, 2, 3]:
                phi = self._exp_LP123(-nu)
            elif self.params.Li == 4:
                phi = self._exp_LP45(-nu, 1. - 2. * self.params.mu)
            else:  # Li = 5
                phi = self._exp_LP45(-nu, -1. + 2. * self.params.mu)
            Y[:, 0:2] = phi[:, 2:4]

            return Y
        else:  # elliptical case
            raise NotImplementedError('_Y_ip3bp_ds: analytical elliptical case not coded yet')

    def transition_ip3bp(self, x1_bar, nu1, nu2):
        """Wrapper for the propagation of the transformed vector in the in-plane 3-body problem.

                Args:
                    x1_bar (np.array): initial transformed, in-plane state vector.
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.

                Returns:
                    (np.array): final transformed, in-plane state vector.

        """

        # sanity check(s)
        if len(x1_bar) != 4:
            raise ValueError('TRANSITION_IP3BP: in-plane initial conditions need to be four-dimensional')
        if (self.params.ecc >= 1.0) or (self.params.ecc < 0.0):
            raise ValueError('TRANSITION_IP3BP: eccentricity must be larger or equal to 0 and strictly less than 1')

        if self.params.ecc == 0.:
            if self.params.Li in [1, 2, 3]:
                phi = self._exp_LP123(nu2 - nu1)
            elif self.params.Li == 4:
                phi = self._exp_LP45(nu2 - nu1, 1. - 2. * self.params.mu)
            else:  # Li = 5
                phi = self._exp_LP45(nu2 - nu1, -1. + 2. * self.params.mu)
            return phi.dot(x1_bar)
        else:  # elliptical case
            raise NotImplementedError('TRANSITION_IP3BP: analytical elliptical case not coded yet')

    def _exp_LP123(self, nu):
        """Function computing the exponential of the true anomaly times the matrix involved in the in-plane
        linearized equations around L1, 2 or 3.

                Args:
                    nu (float): true anomaly.

                Returns:
                    phi (np.array): transition matrix.

        """

        puls = puls_oop_LP(self.x_eq_normalized, self.params.mu)
        (gamma_re, gamma_im, c, k) = inter_L123(puls * puls)
        exp_inter = math.exp(gamma_re * nu)
        inv_exp = 1. / exp_inter
        gamma_im_times_nu = gamma_im * nu
        cos_inter = math.cos(gamma_im_times_nu)
        sin_inter = math.sin(gamma_im_times_nu)
        row1 = [exp_inter, inv_exp, cos_inter, sin_inter]
        row2 = [c * exp_inter, -c * inv_exp,
                 -k * sin_inter, k * cos_inter]
        row3 = [gamma_re * exp_inter, -gamma_re * inv_exp,
                 -gamma_im * sin_inter, gamma_im * cos_inter]
        row4 = [gamma_re * c * exp_inter, gamma_re * c * inv_exp,
                 -gamma_im * k * cos_inter, -gamma_im * k * sin_inter]
        phi = np.array([row1, row2, row3, row4]).dot(self._A_inv)

        return phi

    def _exp_LP45(self, nu, kappa):
        """Function computing the exponential of the true anomaly times the matrix involved in the in-plane
        linearized equations around L4 or 5.

                Args:
                    nu (float): true anomaly.
                    kappa (float): parameter equal to 1-2mu for L4, -1+2mu for L5.

                Returns:
                    phi (np.array): transition matrix.

        """

        root1, root2, a1, a2, b1, b2, c1, c2, d1, d2 = inter_L45(self.params.mu, kappa)
        root1_times_nu = root1 * nu
        root2_times_nu = root2 * nu
        cos1 = math.cos(root1_times_nu)
        sin1 = math.sin(root1_times_nu)
        cos2 = math.cos(root2_times_nu)
        sin2 = math.sin(root2_times_nu)
        row1 = [cos1, sin1, cos2, sin2]
        row2 = [a1 * cos1 + b1 * sin1,
                 a2 * cos1 + b2 * sin1,
                 c1 * cos2 + d1 * sin2,
                 c2 * cos2 + d2 * sin2]
        row3 = [-root1 * sin1, root1 * cos1,
                 -root2 * sin2, root2 * cos2]
        row4 = [-root1 * a1 * sin1 + root1 * b1 * cos1,
                 -root1 * a2 * sin1 + root1 * b2 * cos1,
                 -root2 * c1 * sin2 + root2 * d1 * cos2,
                 -root2 * c2 * sin2 + root2 * d2 * cos2]
        phi = np.array([row1, row2, row3, row4]).dot(self._A_inv)

        return phi

    def _transition_ip(self, x1, nu1, nu2):
        """Function returning the in-plane initial vector propagated to the final true anomaly.

                Args:
                    x1 (np.array): in-plane initial transformed state vector.
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.

        """

        return self.transition_ip3bp(x1, nu1, nu2)

    def _rhs_ip(self, nu1, nu2, x1, x2):
        """Wrapper for the right-hand side of the moment-equation in the in-plane restricted 3-body problem.

                Args:
                    nu1 (float): initial true anomaly.
                    nu2 (float): final true anomaly.
                    x1 (np.array): initial transformed state vector.
                    x2 (np.array): final transformed state vector.

        """

        if self.params.ecc == 0.:
            if self.params.Li in [1, 2, 3]:
                u = self._exp_LP123(-nu2).dot(x2) - self._exp_LP123(-nu1).dot(x1)
            else:  # Li = 4 or 5
                arg2 = 1. - 2. * self.params.mu if self.params.Li == 4 else -1. + 2. * self.params.mu
                u = self._exp_LP45(-nu2, arg2).dot(x2) - self._exp_LP45(-nu1, arg2).dot(x1)
        else:
            raise NotImplementedError
        return u
