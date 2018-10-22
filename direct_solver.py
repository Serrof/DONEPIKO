# direct_solver.py: class implementing direct solvers
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy
from numpy import linalg
from scipy.optimize import linprog
from cvxopt import matrix, solvers
import solver
import math
import utils
from tuning_params import *


class DirectSolver(solver.Solver):
    """Class implementing direct solvers to the optimal control problem of fuel minimization.

    """

    def __init__(self, dyn, p, prop_ana):
        """Constructor for class DirectSolver.

                Args:
                    dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                    p (int): type of norm to be minimized.

        """

        solver.Solver.__init__(self, dyn, p, indirect=False, prop_ana=prop_ana)  # call to parent constructor

        self._DV_min = DV_min
        self._n_grid_1norm = n_grid_1norm
        self._n_grid_2norm = n_grid_2norm
        self._tol_linprog_dir = tol_linprog_dir
        self._tol_cvx_dir = tol_cvx_dir

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
        for i in range(0, d):
            z[i] /= scale

        # building grid on possible impulses' location
        if self.p == 1:
            grid = numpy.linspace(BC.nu0, BC.nuf, self._n_grid_1norm)
        else:  # p = 2
            grid = numpy.linspace(BC.nu0, BC.nuf, self._n_grid_2norm)
        Y_grid = self.grid_Y(grid, BC.half_dim)

        if self.p == 1:

            # building matrix for linear program
            M = numpy.zeros((d, d * self._n_grid_1norm))
            for k in range(0, self._n_grid_1norm):
                inter = Y_grid[:, k * BC.half_dim: (k+1) * BC.half_dim]
                M[:, d * k: d * k + BC.half_dim] = inter
                M[:, d * k + BC.half_dim: d * k + d] = -inter

            # solving for slack variables
            res = linprog(numpy.ones(d * self._n_grid_1norm), A_eq=M, b_eq=z, options={"disp": False, "tol": self._tol_linprog_dir})
            sol = res.x
            if verbose:
                print('direct cost 1-norm: ' + str(res.fun))

            # extracting nus with non-zero impulses
            lost = 0.0  # variable to keep track of cost from deleted impulses
            n_components = 0
            indices = []
            nus = []
            for k in range(0, self._n_grid_1norm):
                DV = sol[d * k: d * k + BC.half_dim] - sol[d * k + BC.half_dim: d * k + d]
                if linalg.norm(DV, 1) > self._DV_min:
                    indices.append(k)
                    nus.append(grid[k])
                    for component in DV:
                        if math.fabs(component) > self._DV_min:
                            n_components += 1
                else:  # Delta-V is considered numerically negligible
                    lost += linalg.norm(DV, 1)

            # reconstructing velocity jumps
            DVs = numpy.zeros((len(nus), BC.half_dim))
            for k, index in enumerate(indices):
                if BC.half_dim == 1:
                    DVs[k, 0] = sol[2 * index] - sol[2 * index + 1]
                else:  # in-plane of complete dynamics
                    for j in range(0, BC.half_dim):
                        aux = sol[d * index + j] - sol[d * index + BC.half_dim + j]
                        if math.fabs(aux) > self._DV_min:
                            DVs[k, j] = aux
                        else:  # Delta-V is considered numerically negligible
                            lost += math.fabs(aux)
            if verbose:
                print("lost impulse: " + str(lost))

        else:  # p = 2

            # building matrix for linear constraints
            M = numpy.zeros((d, BC.half_dim * self._n_grid_2norm))
            for k in range(0, self._n_grid_2norm):
                M[:, BC.half_dim * k: BC.half_dim * (k + 1)] = Y_grid[:, k * BC.half_dim: (k + 1) * BC.half_dim]
            A = numpy.concatenate((numpy.zeros((d, self._n_grid_2norm)), M), axis=1)
            A = matrix(A)

            # building matrix for linear cost function
            f = numpy.concatenate((numpy.ones(self._n_grid_2norm), numpy.zeros(BC.half_dim * self._n_grid_2norm)), axis=0)
            f = matrix(f)

            # building matrices for SDP constraints
            G = None
            h = None
            vec = numpy.zeros(BC.half_dim + 1)
            vec = matrix(vec)
            for j in range(0, self._n_grid_2norm):
                mat = numpy.zeros((BC.half_dim + 1, self._n_grid_2norm * (BC.half_dim + 1)))
                mat[0, j] = -1.0
                for i in range(0, BC.half_dim):
                    mat[i + 1, self._n_grid_2norm + BC.half_dim * j + i] = 1.0
                if j == 0:
                    G = [matrix(mat)]
                    h = [vec]
                else:  # not first loop
                    G += [matrix(mat)]
                    h += [vec]

            if not verbose:
                solvers.options['show_progress'] = False  # turn off printed stuff
            solvers.options['abstol'] = self._tol_cvx_dir
            solution = solvers.socp(f, Gq=G, hq=h, A=A, b=matrix(z))
            sol = []
            for el in solution['x']:
                sol.append(el)
            if verbose:
                print("direct cost 2-norm: " + str(solution['primal objective']))

            # extracting nus with non-zero impulses
            lost = 0.0  # variable to keep track of cost from deleted impulses
            indices = []
            nus = []
            for k in range(0, self._n_grid_2norm):
                DV = sol[self._n_grid_2norm + BC.half_dim * k: self._n_grid_2norm + BC.half_dim * k + BC.half_dim]
                if linalg.norm(DV, 2) > self._DV_min:
                    indices.append(k)
                    nus.append(grid[k])
                else:  # Delta-V is considered numerically negligible
                    lost += linalg.norm(DV, 2)
            if verbose:
                print("lost impulse: " + str(lost))

            # reconstructing velocity jumps
            DVs = numpy.zeros((len(nus), BC.half_dim))
            for k in range(0, len(nus)):
                DVs[k, :] = sol[self._n_grid_2norm + BC.half_dim * indices[k]: self._n_grid_2norm + BC.half_dim * indices[k] + BC.half_dim]

        # un-scaling
        for j in range(0, len(nus)):
            for i in range(0, BC.half_dim):
                DVs[j, i] *= scale

        return utils.ControlLaw(BC.half_dim, nus, DVs)
