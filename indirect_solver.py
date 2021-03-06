# indirect_solver.py: class implementing indirect solvers
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy as np
from numpy import linalg
import utils
from utils import unstack_state
from indirect_num import primal_to_dual, solve_primal
from indirect_ana import solver_ana
import solver
import orbital_mechanics
import body_prob_dyn
from config import conf


class IndirectSolver(solver.Solver):
    """Class implementing indirect solvers to the optimal control problem of fuel minimization.

    """

    def __init__(self, dyn, p, prop_ana):
        """Constructor for class IndirectSolver.

                Args:
                    dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                    p (int): type of norm to be minimized.
                    prop_ana (bool): set to true for analytical propagation of motion, false for integration.


        """

        solver.Solver.__init__(self, dyn, p, indirect=True, prop_ana=prop_ana)  # call to parent constructor

    def run(self, BC):
        """Function to optimize a trajectory.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                     (utils.ControlLaw): optimal control law.

        """

        if isinstance(self.dyn, body_prob_dyn.BodyProbDyn):
            return self.solveBodyProbDyn(BC)
        else:  # not the 2 or 3-body problem
            return self._num_approach(BC)  # solve numerically by default

    def _num_approach(self, BC):
        """Function handling the indirect approach numerically.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                     (utils.ControlLaw): optimal control law obtained with numerical approach.

        """

        z = self.dyn.compute_rhs(BC, self.prop_ana)

        # scaling the right-hand side of the moment equation
        scale = linalg.norm(z)
        if scale != 0.:
            z /= scale

            # building grid for norm checks
            grid_check = list(np.linspace(BC.nu0, BC.nuf, conf.params_indirect["n_check"]))
            Y_grid = self.grid_Y(grid_check, BC.half_dim)

            lamb = solve_primal(grid_check, Y_grid, z, self.p)
            (nus, DVs) = primal_to_dual(grid_check, Y_grid, lamb, z, self.p)

            DVs *= scale

            return utils.ControlLaw(BC.half_dim, nus, DVs, lamb)
        else:
            return utils.NoControl(BC)

    def solveBodyProbDyn(self, BC):
        """Wrapper for all solving strategies (analytical, numerical and both) in case of 2 or 3-body problem.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                     (utils.ControlLaw): optimal control law.

        """

        if BC.half_dim == 1 and self.prop_ana:  # out-of-plane analytical solving
            if self.dyn.params.mu != 0 and self.dyn.params.ecc == 0. and self.dyn.params.Li in [1, 2, 3]:
                # special case of circular out-of-plane L1, 2 or 3 where analytical solution can be obtained from
                # 2-body solution by rescaling anomaly
                return self._circular_oop_L123(BC)
            else:  # out-of-plane for elliptical 2-body problem or elliptical 3-body around L4 and 5 or circular around L1, 2 and 3
                z = self.dyn.compute_rhs(BC, analytical=self.prop_ana)
                (nus, DVs, lamb) = solver_ana(z, self.dyn.params.ecc, self.dyn.params.mean_motion, BC.nu0, BC.nuf)
                return utils.ControlLaw(BC.half_dim, nus, DVs, lamb)

        elif (BC.half_dim == 3) and (self.p == 1):  # merge analytical and numerical solutions in the case of complete
            #  dynamics with the 1-norm
            x0_ip, x0_oop = unstack_state(BC.x0)
            xf_ip, xf_oop = unstack_state(BC.xf)
            BC_oop = utils.BoundaryConditions(BC.nu0, BC.nuf, x0_oop, xf_oop)
            CL_oop = self.solveBodyProbDyn(BC_oop)
            BC_ip = utils.BoundaryConditions(BC.nu0, BC.nuf, x0_ip, xf_ip)
            CL_ip = self.solveBodyProbDyn(BC_ip)
            return utils.merge_control(CL_ip, CL_oop)

        else:  # numerical solving
            return self._num_approach(BC)

    def _circular_oop_L123(self, BC):
        """Function computing the optimal control law analytically in the special case of out-of-plane dynamics
        in the circular L1, 2 or 3.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                     (utils.ControlLaw): analytical optimal control law.

        """

        # getting scaling factor for time
        puls = orbital_mechanics.puls_oop_LP(self.dyn.x_eq_normalized, self.dyn.params.mu)

        # scaling inputs
        BC_rescaled = utils.BoundaryConditions(BC.nu0 * puls, BC.nuf * puls, BC.x0, BC.xf)
        dyn_rescaled = body_prob_dyn.RestriTwoBodyProb(0., self.dyn.params.period / puls, self.dyn.params.sma)
        z_rescaled = dyn_rescaled.compute_rhs(BC_rescaled, analytical=True)

        # 'classic' call to numerical solver
        (nus, DVs, lamb) = solver_ana(z_rescaled, 0., dyn_rescaled.params.mean_motion, BC_rescaled.nu0, BC_rescaled.nuf)

        factor = linalg.norm(self.dyn.compute_rhs(BC, analytical=True)) / linalg.norm(z_rescaled)

        # un-scaling
        nus = np.array(nus) / puls
        lamb /= factor

        return utils.ControlLaw(BC.half_dim, nus, DVs, lamb)
