# solver.py: abstract class for optimal solvers
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy as np
import body_prob_dyn
import orbital_mechanics
import utils
import math
from abc import ABCMeta, abstractmethod


class Solver:
    """Abstract class for the implementation of fuel-optimality solvers.

            Attributes:
                dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                p (int): type of norm to be minimized.
                _indirect (bool): set to True for indirect approach and False for direct one.
                prop_ana (bool): set to true for analytical propagation of motion, false for integration.

    """

    __metaclass_ = ABCMeta

    def __init__(self, dyn, p, indirect, prop_ana):
        """Constructor for class Solver.

                Args:
                    dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                    p (int): type of norm to be minimized.
                    indirect (bool): set to True for indirect approach and False for direct one.
                    prop_ana (bool): set to true for analytical propagation of motion, false for integration.

        """

        # sanity check(s)
        if p != 1 and p != 2:
            raise ValueError("solver: type of norm to minimized must be 1 or 2")

        self.dyn = dyn.copy()
        self._indirect = indirect
        self.p = p
        self.prop_ana = prop_ana

    @abstractmethod
    def run(self, BC):
        """Abstract method optimizing a trajectory.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                    (utils.ControlLaw): optimal control law.

        """

        raise NotImplementedError

    def set_dyn(self, dyn):
        """Setter for attribute dyn.

                Args:
                    dyn (dynamical_system.DynamicalSystem): new dynamics to be used for two-boundary value problem.

        """

        self.dyn = dyn.copy()

    def grid_Y(self, grid, half_dim):
        """Function computing the value of the moment-function on the given list of true anomalies.

                Args:
                    grid (List[float]): grid of true anomalies where to compute the moment-function.
                    half_dim (int): half-dimension of state-vector.

                Returns:
                    Ys (np.array): grid of values for moment-function on input grid.

        """

        if self.prop_ana:
            grid_size = len(grid)
            Ys = np.zeros((2 * half_dim, half_dim * grid_size))
            for k, el in enumerate(grid):
                Ys[:, half_dim * k: half_dim * (k + 1)] = self.dyn.evaluate_Y(el, half_dim)
        else:  # numerical integration
            Ys = self.dyn.integrate_Y(grid, half_dim)

        return Ys

    def boundary_impulses(self, BC):
        """Function computing the usually sub-optimal control law consisting in two burns, at initial and final times.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                    (utils.ControlLaw): control achieving rendezvous in position at final time with impulse at BC.nu0.

        """

        if isinstance(self.dyn, body_prob_dyn.BodyProbDyn):

            if ((BC.nuf - BC.nu0) % math.pi) != 0.:
                # build and solve system of equations for two boundary impulses
                mat = np.zeros((2 * BC.half_dim, 2 * BC.half_dim))

                if self.prop_ana:
                    mat[:, 0:BC.half_dim] = self.dyn.evaluate_Y(BC.nu0, BC.half_dim)
                    mat[:, BC.half_dim:2 * BC.half_dim] = self.dyn.evaluate_Y(BC.nuf, BC.half_dim)
                    inv_mat = np.linalg.inv(mat)
                    inter = inv_mat.dot(self.dyn.compute_rhs(BC, analytical=True))

                else:  # numerical propagation

                    matrices = self.dyn.integrate_phi_inv([BC.nu0, BC.nuf], BC.half_dim)

                    IC_matrix = matrices[0]
                    FC_matrix = matrices[-1]

                    rho_nu0 = orbital_mechanics.rho_func(self.dyn.params.ecc, BC.nu0)
                    rho_nuf = orbital_mechanics.rho_func(self.dyn.params.ecc, BC.nuf)
                    mat[:, 0: BC.half_dim] = IC_matrix[:, BC.half_dim: 2 * BC.half_dim] / rho_nu0
                    mat[:, BC.half_dim: 2 * BC.half_dim] = FC_matrix[:, BC.half_dim: 2 * BC.half_dim] / rho_nuf

                    factor = 1.0 - self.dyn.params.ecc * self.dyn.params.ecc
                    multiplier = self.dyn.params.mean_motion / math.sqrt(factor * factor * factor)
                    inv_mat = np.linalg.inv(mat)
                    inter = inv_mat.dot((FC_matrix.dot(self.dyn.transformation(BC.xf, BC.nuf)) -
                                         IC_matrix.dot(self.dyn.transformation(BC.x0, BC.nu0))) * multiplier)

                # retrieve the two Delta-Vs
                DVs = np.zeros((2, BC.half_dim))
                DVs[0, :] = inter[0:BC.half_dim]
                DVs[1, :] = inter[BC.half_dim:2 * BC.half_dim]

                # build and solve system of equations for corresponding primer vector
                if self.p == 2:
                    inter2 = np.zeros(2 * BC.half_dim)
                    inter2[:BC.half_dim] = inter[:BC.half_dim] / np.linalg.norm(DVs[0, :], self.p)
                    inter2[BC.half_dim:] = inter[BC.half_dim:] / np.linalg.norm(DVs[1, :], self.p)
                    lamb = np.transpose(inv_mat).dot(inter2)

                else:  # p = 1
                    indices = inter != 0.
                    if len(indices) == 2 * BC.half_dim:
                        inter2 = np.sign(inter)
                        lamb = np.transpose(inv_mat).dot(inter2)
                    else:  # no unique solution for coefficients of primer vector
                        lamb = None

                return utils.ControlLaw(BC.half_dim, [BC.nu0, BC.nuf], DVs, lamb)

            else:  # there is no unique solution to the two-impulse boundary control
                return Exception("No two-impulse boundary control was found.")

        else:
            raise NotImplementedError("The computation of two-impulse boundary control laws is not implemented "
                                       "for that dynamics.")
