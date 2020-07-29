# direct_solver.py: class implementing direct solvers
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy as np
from numpy import linalg
from scipy.optimize import linprog
from cvxopt import matrix, solvers
import solver
import math
import utils
from config import conf


class DirectSolver(solver.Solver):
    """Class implementing direct solvers to the optimal control problem of fuel minimization.

    """

    def __init__(self, dyn, p, prop_ana):
        """Constructor for class DirectSolver.

                Args:
                    dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                    p (int): type of norm to be minimized.
                    prop_ana (bool): set to true for analytical propagation of motion, false for integration.

        """

        solver.Solver.__init__(self, dyn, p, indirect=False, prop_ana=prop_ana)  # call to parent constructor

    def run(self, BC):
        """Function to call for optimizing a trajectory by the direct approach.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

                Returns:
                    (utils.ControlLaw): optimal control law.

        """

        # pre-computations
        d = 2 * BC.half_dim
        z = self.dyn.compute_rhs(BC, self.prop_ana)
        # scaling the right-hand side of the moment equation
        scale = linalg.norm(z)
        z /= scale

        # building grid on possible impulses' location
        if self.p == 1:
            n_grid = conf.params_direct["n_grid_1norm"]
        else:  # p = 2
            n_grid = conf.params_direct["n_grid_2norm"]
        grid = np.linspace(BC.nu0, BC.nuf, n_grid)
        Y_grid = self.grid_Y(list(grid), BC.half_dim)

        if self.p == 1:

            # building matrix for linear program
            M = np.zeros((d, d * n_grid))
            for k in range(0, n_grid):
                inter = Y_grid[:, k * BC.half_dim: (k+1) * BC.half_dim]
                M[:, d * k: d * k + BC.half_dim] = inter
                M[:, d * k + BC.half_dim: d * k + d] = -inter

            # solving for slack variables
            res = linprog(np.ones(d * n_grid), A_eq=M, b_eq=z,
                          options={"disp": conf.params_other["verbose"],
                                   "tol": conf.params_direct["tol_lin_prog"]})
            if not res.success:
                raise InterruptedError("Linear Program did not converge.")

            sol = res.x
            if conf.params_other["verbose"]:
                print('direct cost 1-norm: ' + str(res.fun))

            # extracting nus with non-zero impulses
            lost = 0.0  # variable to keep track of cost from deleted impulses
            n_components = 0
            indices = []
            nus = []
            for k, el in enumerate(grid):
                DV = sol[d * k: d * k + BC.half_dim] - sol[d * k + BC.half_dim: d * k + d]
                if linalg.norm(DV, 1) > conf.params_direct["DV_min"]:
                    indices.append(k)
                    nus.append(el)
                    for component in DV:
                        if math.fabs(component) > conf.params_direct["DV_min"]:
                            n_components += 1
                else:  # Delta-V is considered numerically negligible
                    lost += linalg.norm(DV, 1)

            # reconstructing velocity jumps
            DVs = np.zeros((len(nus), BC.half_dim))
            for k, index in enumerate(indices):
                if BC.half_dim == 1:
                    DVs[k, 0] = sol[2 * index] - sol[2 * index + 1]
                else:  # in-plane of complete dynamics
                    for j in range(0, BC.half_dim):
                        aux = sol[d * index + j] - sol[d * index + BC.half_dim + j]
                        if math.fabs(aux) > conf.params_direct["DV_min"]:
                            DVs[k, j] = aux
                        else:  # Delta-V is considered numerically negligible
                            lost += math.fabs(aux)
            if conf.params_other["verbose"]:
                print("lost impulse: " + str(lost))

        else:  # p = 2

            # building matrix for linear constraints
            M = np.zeros((d, BC.half_dim * n_grid))
            for k in range(0, n_grid):
                M[:, BC.half_dim * k: BC.half_dim * (k + 1)] = Y_grid[:, k * BC.half_dim: (k + 1) * BC.half_dim]
            A = np.concatenate((np.zeros((d, n_grid)), M), axis=1)
            A = matrix(A)

            # building matrix for linear cost function
            f = np.concatenate((np.ones(n_grid), np.zeros(BC.half_dim * n_grid)), axis=0)
            f = matrix(f)

            # building matrices for SDP constraints
            G = None
            h = None
            vec = np.zeros(BC.half_dim + 1)
            vec = matrix(vec)
            for j in range(0, n_grid):
                mat = np.zeros((BC.half_dim + 1, n_grid * (BC.half_dim + 1)))
                mat[0, j] = -1.0
                for i in range(0, BC.half_dim):
                    mat[i + 1, n_grid + BC.half_dim * j + i] = 1.0
                if j == 0:
                    G = [matrix(mat)]
                    h = [vec]
                else:  # not first loop
                    G += [matrix(mat)]
                    h += [vec]

            solvers.options["show_progress"] = conf.params_other["verbose"]
            solvers.options["abstol"] = conf.params_direct["tol_cvx"]
            solvers.options["maxiters"] = conf.params_direct["max_iter_cvx"]
            solution = solvers.socp(f, Gq=G, hq=h, A=A, b=matrix(z))

            if solution["status"] is not "optimal":
                raise InterruptedError("Semi-Definite Program did not converge. Set verbose to True to see details.")

            sol = list(solution["x"])
            if conf.params_other["verbose"]:
                print("direct cost 2-norm: " + str(solution["primal objective"]))

            # extracting nus with non-zero impulses
            lost = 0.0  # variable to keep track of cost from deleted impulses
            indices = []
            nus = []
            for k, el in enumerate(grid):
                DV = sol[n_grid + BC.half_dim * k:
                         n_grid + BC.half_dim * k + BC.half_dim]
                if linalg.norm(DV, 2) > conf.params_direct["DV_min"]:
                    indices.append(k)
                    nus.append(el)
                else:  # Delta-V is considered numerically negligible
                    lost += linalg.norm(DV, 2)
            if conf.params_other["verbose"]:
                print("lost impulse: " + str(lost))

            # reconstructing velocity jumps
            DVs = np.zeros((len(nus), BC.half_dim))
            for k, index in enumerate(indices):
                DVs[k, :] = sol[n_grid + BC.half_dim * index:
                                n_grid + BC.half_dim * index + BC.half_dim]

        # un-scaling
        DVs *= scale

        return utils.ControlLaw(BC.half_dim, nus, DVs)
