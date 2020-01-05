# dynamical_system.py: set of classes for the dynamical models with an example dynamics
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

from abc import ABCMeta, abstractmethod
import numpy as np
import math
import integrators
import utils
from config import conf


class DynParams:
    """Abstract class for the implementation of dynamical parameters.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def copy(self):
        """Function returning a copy of the object.

        """
        pass


class DynamicalSystem:
    """Abstract class for dynamical systems.

        Attributes:
            name (str): name of the dynamics implemented.
            params (DynParams): object dealing with the dynamical parameters.
            convToAlterIndVar (func): conversion from original to alternative independent variable
            convFromAlterIndVar (func): conversion from alternative to original independent variable

    """

    __metaclass__ = ABCMeta

    def __init__(self, name):
        """Constructor.

                Args:
                    name (str): name of implemented dynamics.

        """
        self.name = name
        self.params = None

        # conversion to dummy alternative independent variable
        def identical_var(nu0, t0, nu):
            return nu

        # default is alternative variable identical to original one
        self.convToAlterIndVar = identical_var
        self.convFromAlterIndVar = identical_var

    @abstractmethod
    def matrix_linear(self, nu, half_dim):
        """Function returning the matrix appearing in the differential system satisfied by the transformed state vector
         in the linearized dynamics.

                Args:
                    nu (float): value of independent variable.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): matrix for transformed state equation in linearized dynamics.

        """
        pass

    def evaluate_state_deriv(self, nu, x):
        """Function returning the derivative of the transformed state vector w.r.t. the independent variable in the
        linearized dynamics.

                Args:
                    nu (float): value of independent variable.
                    x (np.array): transformed state vector.

                Returns:
                    (np.array): derivative of transformed state vector in linearized dynamics.

        """
        return self.matrix_linear(nu, int(len(x) / 2)).dot(x)

    def evaluate_state_deriv_nonlin(self, nu, x):
        """Function returning the derivative of the state vector w.r.t. the independent variable in the non-linearized
        dynamics. This default implementation implicitly assumes that the dynamics is already linear. For non-linear
        dynamics, it needs to be overloaded.

                Args:
                    nu (float): value of independent variable.
                    x (np.array): state vector.

                Returns:
                    (np.array): derivative of state vector in non-linear dynamics.

        """
        return self.evaluate_state_deriv(nu, x)

    @abstractmethod
    def propagate(self, nu1, nu2, x1):
        """Function for the propagation of the state vector.

                Args:
                    nu1 (float): initial value of independent variable.
                    nu2 (float): final value of independent variable.
                    x1 (np.array): initial state vector.

                Returns:
                    (np.array): final state vector.

        """
        pass

    @abstractmethod
    def evaluate_Y(self, nu, half_dim):
        """Function returning the moment-function involved in the equation satisfied by the control law.

                Args:
                    nu (float): current value of independent variable.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): moment-function evaluated at nu.

        """
        pass

    @abstractmethod
    def compute_rhs(self, BC, analytical):
        """Function that computes right-hand side of moment equation.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.
                    analytical (bool): set to true for analytical propagation of motion, false for integration.

                Returns:
                    u (np.array): right-hand side of moment equation.

        """
        pass

    @abstractmethod
    def integrate_Y(self, nus, half_dim):
        """Function integrating over the independent variable the moment-function.

                Args:
                    nus (list): grid of values for independent variable.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    outputs (list): moment-function integrated on input grid.

        """
        pass

    @abstractmethod
    def copy(self):
        """Function returning a copy of the object.

        """
        pass

    def transformation(self, x, nu):
        """Method to be overwritten if there is a transformation to be performed to obtain a state vector that has an
        analytical formula to be propagated.

                Args:
                    x (np.array): original state vector.
                    nu (float): independent variable.

                Returns:
                    (np.array): transformed state vector.

        """
        return [el for el in x]

    def transformation_inv(self, x, nu):
        """Method to be overwritten by inverse of transformation if the latter is different from (x, nu) -> x.

                Args:
                    x (np.array): transformed state vector.
                    nu (float): independent variable.

                Returns:
                    (np.array): original state vector.

        """
        return self.transformation(x, nu)

    def integrate_phi_inv(self, nus, half_dim):
        """Function integrating over the independent variable the inverse of the fundamental transition matrix
        associated to the transformed state vector.

                Args:
                    nus (list): grid of values for independent variable.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    outputs (list): list of inverse fundamental transition matrices integrated on input grid.

        """

        # sanity check(s)
        if (half_dim != 3) and (half_dim != 2) and (half_dim != 1):
            return ValueError('integrate_Y: half-dimension of state vector should be 1, 2 or 3')

        def func(nu, x):
            x_matrix = utils.vector_to_square_matrix(x, 2 * half_dim)
            f_matrix = -x_matrix.dot(self.matrix_linear(nu, half_dim))  # right-hand side of matrix differential equation satisfied by phi^-1
            return utils.square_matrix_to_vector(f_matrix, 2 * half_dim)

        integ = integrators.RK4(func)
        outputs = []
        IC_matrix = np.eye(2 * half_dim)
        outputs.append(IC_matrix)
        IC_vector = utils.square_matrix_to_vector(IC_matrix, 2 * half_dim)  # initial conditions of matrix system turned into a vector for integration
        n_step = int(math.ceil(math.fabs(nus[1] - nus[0]) / conf.params_other["max_stepsize"]))

        for k in range(0, len(nus)-1):
            (state_hist, nu_hist) = integ.integrate(nus[k], nus[k+1], IC_vector, n_step)

            outputs.append(utils.vector_to_square_matrix(state_hist[-1], 2 * half_dim))

            IC_vector = state_hist[-1]  # old final condition becomes initial one

        return outputs


class ZeroGravity(DynamicalSystem):
    """Class implementing the dynamics with no acting forces i.e. second order state derivative equal to zero, so that
    uncontrolled trajectories follow a rectilinear motion.


    """

    def __init__(self):
        """Constructor.


        """
        DynamicalSystem.__init__(self, "Zero Gravity")

    def matrix_linear(self, nu, half_dim):
        """Function returning the matrix appearing in the differential system satisfied by the transformed state vector
         in the zero-gravity dynamics.

                Args:
                    nu (float): value of independent variable.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): matrix for transformed state equation in linearized dynamics.

        """
        A = np.zeros((2 * half_dim, 2 * half_dim))
        A[0: half_dim, half_dim: 2 * half_dim] = np.eye(half_dim)
        return A

    def propagate(self, nu1, nu2, x1):
        """Function for the propagation of the state vector in zero-gravity dynamics.

                Args:
                    nu1 (float): initial value of independent variable.
                    nu2 (float): final value of independent variable.
                    x1 (np.array): initial state vector.

                Returns:
                    x2 (np.array): final state vector.

        """
        x2 = np.array(x1[:])
        dnu = nu2 - nu1
        half_dim = int(len(x1) / 2)
        x2[:half_dim] += dnu * x1[half_dim:]
        return x2

    def evaluate_Y(self, nu, half_dim):
        """Function returning the moment-function involved in the equation satisfied by the control law.

                Args:
                    nu (float): current value of independent variable.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (np.array): moment-function evaluated at nu in zero-gravity dynamics.

        """
        Y = np.zeros((2 * half_dim, half_dim))
        Y[0:half_dim, :] = -nu * np.eye(half_dim)
        Y[half_dim: 2 * half_dim, :] = np.eye(half_dim)
        return Y

    def compute_rhs(self, BC, analytical):
        """Function that computes right-hand side of moment equation for the zero-gravity dynamics.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.
                    analytical (bool): set to true for analytical propagation of motion, false for integration.

                Returns:
                    u (np.array): right-hand side of moment equation.

        """
        if analytical:
            M0 = np.eye(2 * BC.half_dim)
            M0[0:BC.half_dim, BC.half_dim: 2 * BC.half_dim] = -BC.nu0 * np.eye(BC.half_dim)
            Mf = np.eye(2 * BC.half_dim)
            Mf[0:BC.half_dim, BC.half_dim: 2 * BC.half_dim] = -BC.nuf * np.eye(BC.half_dim)
            return Mf.dot(BC.xf) - M0.dot(BC.x0)

        else:  # propagation is numerical
            matrices = self.integrate_phi_inv([BC.nu0, BC.nuf], BC.half_dim)
            return matrices[-1].dot(BC.xf) - matrices[0].dot(BC.x0)

    def integrate_Y(self, nus, half_dim):
        """Function integrating over the independent variable the moment-function.

                Args:
                    nus (list): grid of values for independent variable.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    outputs (list): moment-function integrated on input grid.

        """
        matrices = self.integrate_phi_inv(nus, half_dim)
        Ys = np.zeros((2 * half_dim, half_dim * len(nus)))
        for k, matrix in enumerate(matrices):
            Ys[:, half_dim * k: half_dim * (k + 1)] = matrix[:, half_dim: 2 * half_dim]
        return Ys

    def copy(self):
        """Function returning a copy of the object.

        """
        return ZeroGravity()
