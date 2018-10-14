# indirect_solver.py: class implementing indirect solvers
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy
from numpy import linalg
import utils
from utils import *
from indirect_num import *
from indirect_ana import *
import solver
import orbital_mechanics
import dynamical_system
import tuning_params


class IndirectSolver(solver.Solver):
    """Class implementing indirect solvers to the optimal control problem of fuel minimization.

    """

    def __init__(self, dyn, p):
        """Constructor for class IndirectSolver.

                Args:
                    dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                    p (int): type of norm to be minimized.


        """

        solver.Solver.__init__(self, dyn, p, indirect=True)

    def run(self, BC):
        """Wrapper for all solving strategies (analytical, numerical and both).

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                     (utils.ControlLaw): optimal control law.

        """

        if BC.half_dim == 1:  # analytical case
            if(self.dyn.mu != 0) and (self.dyn.ecc == 0.) and (self.dyn.Li == 1 or self.dyn.Li == 2 or self.dyn.Li == 3):
                # special case of circular out-of-plane L1, 2 or 3 where analytical solution can be obtained from
                # 2-body solution by rescaling anomaly
                return self._circular_oop_L123(BC)
            else:
                z = self.dyn.compute_rhs(BC)
                (nus, DVs, lamb) = solver_ana(z, self.dyn.ecc, self.dyn.mean_motion, BC.nu0, BC.nuf)
        elif (BC.half_dim == 3) and (self.p == 1):  # merge analytical and numerical solutions in the case of complete
            #  dynamics with the 1-norm
            x0_ip, x0_oop = unstack_state(BC.x0)
            xf_ip, xf_oop = unstack_state(BC.xf)
            BC_ip = utils.BoundaryConditions(BC.nu0, BC.nuf, x0_ip, xf_ip)
            BC_oop = utils.BoundaryConditions(BC.nu0, BC.nuf, x0_oop, xf_oop)
            CL_oop = self.run(BC_oop)
            CL_ip = self.run(BC_ip)
            return utils.merge_control(CL_ip, CL_oop)
        else:  # general numerical case

            return self.indirect_num(BC)

        return utils.ControlLaw(BC.half_dim, nus, DVs, lamb)

    def indirect_num(self, BC):
        """Function handling the indirect approach numerically.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                     (utils.ControlLaw): optimal control law obtained with numerical approach.

        """

        z = self.dyn.compute_rhs(BC)
        # scaling the right-hand side of the moment equation
        scale = linalg.norm(z)
        for i in range(0, 2 * BC.half_dim):
            z[i] /= scale

        # building grid for norm checks
        grid_check = numpy.linspace(BC.nu0, BC.nuf, tuning_params.n_check)
        Y_grid = self.grid_Y(grid_check, BC.half_dim)

        lamb = solve_primal(grid_check, Y_grid, z, self.p)
        (nus, DVs) = primal_to_dual(grid_check, Y_grid, lamb, z, self.p)

        for j in range(0, len(nus)):
            for i in range(0, BC.half_dim):
                DVs[j, i] *= scale

        return utils.ControlLaw(BC.half_dim, nus, DVs, lamb)

    def _circular_oop_L123(self, BC):
        """Function computing the optimal control law analytically in the special case of out-of-plane dynamics
        in the circular L1, 2 or 3.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                     (utils.ControlLaw): analytical optimal control law.

        """

        # getting scaling factor for time
        puls = orbital_mechanics.puls_oop_LP(self.dyn.x_L_normalized, self.dyn.mu)

        # scaling inputs
        BC_rescaled = utils.BoundaryConditions(BC.nu0 * puls, BC.nuf * puls, BC.x0, BC.xf)
        dyn_rescaled = dynamical_system.DynamicalSystem(0., 0., self.dyn.period / puls, self.dyn.sma)
        z_rescaled = dyn_rescaled.compute_rhs(BC_rescaled)

        # 'classic' call to numerical solver
        (nus, DVs, lamb) = solver_ana(z_rescaled, 0., dyn_rescaled.mean_motion, BC_rescaled.nu0, BC_rescaled.nuf)

        # un-scaling
        for k in range(0, len(nus)):
            nus[k] /= puls
        factor = linalg.norm(self.dyn.compute_rhs(BC)) / linalg.norm(dyn_rescaled.compute_rhs(BC_rescaled))
        for k in range(0, len(lamb)):
            lamb[k] /= factor

        return utils.ControlLaw(BC.half_dim, nus, DVs, lamb)
