# solver.py: abstract class for optimal solvers
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy
import dynamical_system
import utils
import math


class Solver:
    """Abstract class for the implementation of fuel-optimality solvers.

    """

    def __init__(self, dyn, p, indirect):
        """Constructor for class Solver.

                Args:
                    dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                    p (int): type of norm to be minimized.
                    indirect (bool): set to True for indirect approach and False for direct one.

        """

        # sanity check(s)
        if p != 1 and p != 2:
            print('solver: type of norm to minimized must be 1 or 2')
        if (dyn.mu != 0.) and (dyn.ecc != 0.) and (dyn.Li == 1 or dyn.Li == 2 or dyn.Li == 3):
            print('solver: not coded yet')

        self.dyn = dynamical_system.DynamicalSystem(dyn.mu, dyn.ecc, dyn.period, dyn.sma, dyn.Li)
        self._indirect = indirect
        self.p = p

    def set_norm(self, p):
        """Setter for attribute p.

                Args:
                    p (int): type of norm to be minimized.

        """

        self.p = p

    def set_dyn(self, dyn):
        """Setter for attribute dyn.

                Args:
                    dyn (dynamical_system.DynamicalSystem): new dynamics to be used for two-boundary value problem.

        """

        self.dyn = (dyn.mu, dyn.ecc, dyn.period, dyn.Li)

    def grid_Y(self, grid, half_dim):
        """Function computing the value of the moment-function on the given list of true anomalies.

                Args:
                    grid (list): grid of true anomalies where to compute the moment-function.
                    half_dim (int): half-dimension of state-vector.

                Returns:
                    Ys (numpy.array): grid of values for moment-function on input grid.

        """

        grid_size = len(grid)
        Ys = numpy.zeros((2 * half_dim, half_dim * grid_size))
        for k in range(0, grid_size):
            Ys[:, half_dim * k: half_dim * (k + 1)] = self.dyn.evaluate_Y(grid[k], half_dim)

        return Ys

    def boundary_impulses(self, BC):
        """Function computing the usually sub-optimal control law consisting in two burns, at initial and final times.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                    (utils.ControlLaw): control achieving rendezvous in position at final time with impulse at BC.nu0.

        """

        if ((BC.nuf - BC.nu0) % math.pi) != 0.:
            # build and solve system of equations for two boundary impulses
            mat = numpy.zeros((2 * BC.half_dim, 2 * BC.half_dim))
            mat[:, 0:BC.half_dim] = self.dyn.evaluate_Y(BC.nu0, BC.half_dim)
            mat[:, BC.half_dim:2 * BC.half_dim] = self.dyn.evaluate_Y(BC.nuf, BC.half_dim)
            inv_mat = numpy.linalg.inv(mat)
            inter = inv_mat.dot(self.dyn.compute_rhs(BC))

            # retrieve the two Delta-Vs
            DVs = numpy.zeros((2, BC.half_dim))
            DVs[0, :] = inter[0:BC.half_dim]
            DVs[1, :] = inter[BC.half_dim:2 * BC.half_dim]

            # build and solve system of equations for corresponding primer vector
            if self.p == 2:
                inter2 = numpy.zeros(2 * BC.half_dim)
                for i in range(0, BC.half_dim):
                    inter2[i] = inter[i] / numpy.linalg.norm(inter[0:BC.half_dim], self.p)
                    inter2[i+BC.half_dim] = inter[i+BC.half_dim] / numpy.linalg.norm(inter[BC.half_dim:2*BC.half_dim],
                                                                                     self.p)
                lamb = numpy.transpose(inv_mat).dot(inter2)
            else:  # p = 1
                indices = []
                for i, el in enumerate(inter):
                    if el != 0.:
                        indices.append(i)
                if len(indices) == 2 * BC.half_dim:
                    inter2 = numpy.zeros(2 * BC.half_dim)
                    for i in range(0, 2 * BC.half_dim):
                        inter2[i] = numpy.sign(inter[i])
                    lamb = numpy.transpose(inv_mat).dot(inter2)
                else:  # no unique solution for coefficients of primer vector
                    lamb = None

            return utils.ControlLaw(BC.half_dim, [BC.nu0, BC.nuf], DVs, lamb)
        else:  # there is no unique solution to the two-impulse boundary control
            return utils.NoControl(BC.half_dim)
