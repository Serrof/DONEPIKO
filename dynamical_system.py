# dynamical_system.py: class for the dynamical models
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

from abc import ABCMeta, abstractmethod


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
    def evaluate_state_deriv_nonlin(self, nu, x):
        """Function returning the derivative of the state vector w.r.t. the independent variable in the non-linearized
        dynamics.

                Args:
                    nu (float): value of independent variable.
                    x (numpy.array): state vector.

                Returns:
                    (numpy.array): derivative of state vector in non-linear dynamics.

        """
        pass

    @abstractmethod
    def evaluate_state_deriv(self, nu, x):
        """Function returning the derivative of the transformed state vector w.r.t. the independent variable in the
        linearized dynamics.

                Args:
                    nu (float): value of independent variable.
                    x (numpy.array): transformed state vector.

                Returns:
                    (numpy.array): derivative of transformed state vector in linearized dynamics.

        """

    @abstractmethod
    def propagate(self, nu1, nu2, x1):
        """Function for the propagation of the state vector.

                Args:
                    nu1 (float): initial value of independent variable.
                    nu2 (float): final value of independent variable.
                    x1 (numpy.array): initial state vector.

                Returns:
                    (numpy.array): final state vector.

        """
        pass

    @abstractmethod
    def evaluate_Y(self, nu, half_dim):
        """Function returning the moment-function involved in the equation satisfied by the control law.

                Args:
                    nu (float): current value of independent variable.
                    half_dim (int): half-dimension of state vector.

                Returns:
                    (numpy.array): moment-function evaluated at nu.

        """
        pass

    @abstractmethod
    def copy(self):
        """Function returning a copy of the object.

        """
        pass

    def transformation(self, x, nu):
        """Method to be overwritten is there is a transformation to be performed to obtain a state vector that has an
        analytical formula to be propagated.

                Args:
                    x (numpy.array): original state vector.
                    nu (float): independent variable.

                Returns:
                    (numpy.array): transformed state vector.

        """
        return x.copy()

    def transformation_inv(self, x, nu):
        """Method to be overwritten by inverse of transformation if the latter is different from (x, nu) -> x.

                Args:
                    x (numpy.array): transformed state vector.
                    nu (float): independent variable.

                Returns:
                    (numpy.array): original state vector.

        """
        return self.transformation(x, nu)
