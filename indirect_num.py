# indirect_num.py: functions involved in the numerical solving by the indirect approach
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import math
import random
import numpy
from numpy import linalg
from scipy.optimize import linprog
from cvxopt import matrix, solvers
from const_params import *


def dual_to_primal_norm_type(p):
    """Function returning the type of primal norm given a dual one.

            Args:
                p (int): type of dual norm.

            Returns:
                q (int): type of primal norm.

    """

    # sanity check(s)
    if (p != 1) and (p != 2):
        print('dual_to_primal_norm_type: dual norm must be 1 or 2')

    q = p
    if p == 1:
        q = numpy.inf
    return q


def initialize_iterative_grid(grid_input):
    """Function initializing the grid for the relaxed primal problem from the thinner grid used for post-treatment.

            Args:
                grid_input (list): grid of true anomalies used for norm checks of primer vector.

            Returns:
                 (list): sub-grid to use for first iteration in the solving of the primal problem (norm constraints).
                 (list): indices of points selected in input grid.

    """

    d_nu = grid_input[-1] - grid_input[0]
    if d_nu % math.pi != 0.:
        # only initial and final points of grid are kept
        indices = [0, len(grid_input)-1]
        grid = [grid_input[indices[0]], grid_input[indices[1]]]
        return grid, indices
    else:  # flight duration (measured in radians) is not equal to 0 modulo pi
        return initialize_iterative_grid_randomly(indirect_params["n_init"], grid_input)


def initialize_iterative_grid_randomly(n_points, grid):
    """Function randomly picking points in the grid for norm checks.

            Args:
                n_points (int): number of points to be selected in input grid.
                grid (list): grid of true anomalies used for norm checks of primer vector.

            Returns:
                points (list): sub-grid whose points were randomly picked among input grid.
                indices (list): indices of points randomly picked in input grid.

    """

    indices = []
    points = []
    indices.append(random.randint(0, len(grid) - 1))
    points.append(grid[indices[0]])

    while len(points) < n_points:
        index_drawn = random.randint(0,len(grid) - 1)
        if index_drawn not in indices:  # true anomaly has not been selected yet
            points.append(grid[index_drawn])
            indices.append(index_drawn)

    return points, indices


def find_max_pv(Y_grid, lamb, q):
    """Function checking if the maximum q-norm of the primer vector on a grid is less than one.

            Args:
                Y_grid (numpy.array): grid of moment-function components for norm evaluation of primer vector.
                lamb (numpy.array): coefficients of primer vector.
                q (int): norm for primer vector.

            Returns:
                unit_norm (bool): True if maximum norm is smaller than one.
                index_max (int): index where maximum norm is reached.

    """

    hd = len(lamb) / 2
    index_max = 0
    val_max = 0.0
    unit_norm = True
    for k in range(0, len(Y_grid[0, :])):
        Y_k = Y_grid[:, hd * k: hd * (k + 1)]
        inter = linalg.norm(numpy.transpose(Y_k).dot(lamb), q)
        if val_max < inter:
            val_max = inter
            index_max = k
    if val_max > 1.0 + indirect_params["tol_unit_norm"]:
        unit_norm = False

    return unit_norm, index_max


def remove_nus(Y_grid, q, grid_work, indices_work, lamb):
    """Function removing from a grid the true anomalies where the candidate primer vector has a norm smaller than one.

            Args:
                Y_grid (numpy.array): grid of moment-function components for norm computation of candidate primer vector.
                q (int): type of norm for primer vector.
                grid_work (list): input grid where to trim true anomalies where norm of primer vector is smaller than 1.
                indices_work (list): indices of elements of grid_work in thinner grid of true anomalies used for post-
                process norm checks.
                lamb (numpy.array): coefficients of candidate primer vector.

            Returns:
                 grid (list): filtered grid w.r.t. input one.
                 indices (list): indices in input grid of true anomalies kept in output grid .

    """

    hd = len(lamb) / 2
    grid = []
    indices = []
    for k, nu in enumerate(grid_work):
        grid.append(nu)
        indices.append(indices_work[k])
    removed_nus = 0
    for k in range(0, len(grid)):
        pv = numpy.transpose(Y_grid[:, hd * indices[k - removed_nus]: hd * (indices[k - removed_nus] + 1)]).dot(lamb)
        pv_norm = linalg.norm(pv, q)
        if pv_norm < 1.0 - indirect_params["tol_unit_norm"]:
            del grid[k - removed_nus]
            del indices[k - removed_nus]
            removed_nus += 1

    return grid, indices


def extract_nus(grid_check, Y_grid, lamb, q):
    """Function selecting points of input grid where primer vector has a norm greater than one.

            Args:
                grid_check (): grid of true anomalies where norm of candidate primer vector is compared to 1 to
                check convergence.
                Y_grid (numpy.array): grid of moment-function components for norm evaluation of candidate primer vector.
                lamb (numpy.array): coefficients of candidate primer vector.
                q (int): norm for primer vector.

            Returns:
                nus (list): points of grid_check where norm of candidate primer vector is greater than one.
                indices (list): indices of input grid corresponding to output n_s

    """

    # pre-computations
    d = len(lamb)
    hd = d / 2
    n_check = len(grid_check)

    # extracting optimal nus from primer vector
    nus = []
    indices = []
    k = 0
    while k < n_check:
        Y_k = Y_grid[:, hd * k: hd * (k + 1)]
        inter = linalg.norm(numpy.transpose(Y_k) . dot(lamb), q)
        if 1.0 - indirect_params["tol_unit_norm"] < inter:
            k += 1
            i_nu = k - 1
            # skip following nus for which norm is almost one
            if k < n_check:
                lap = True
                while lap:
                    Y_k = Y_grid[:, hd * k: hd * (k + 1)]
                    inter2 = linalg.norm(numpy.transpose(Y_k) . dot(lamb), q)
                    if 1.0 - indirect_params["tol_unit_norm"] < inter2:
                        if k == n_check:
                            lap = False
                        k += 1
                        if inter2 > inter:
                            inter = inter2
                            i_nu = k - 1
                    else:  # norm is no more less than one according to tolerance
                        lap = False
            nus.append(grid_check[i_nu])
            indices.append(i_nu)
        else:  # norm is not equal to 1 according to tolerance
            k += 1

    return nus, indices


def solve_alphas(M, z, n_alphas):
    """Function solving linear system of equations satisfied by the impulses' norm.

            Args:
                M (numpy.array): matrix such that M * alphas = z.
                z (numpy.array): right-hand side of moment equation.
                n_alphas (int): number of impulses (p=2) or of non-zero components of impulses (p=1).

            Returns:
                alphas (numpy.array): vector made of magnitudes of impulses (p=2) or impulses' components (p=1).

    """

    d = len(z)
    if n_alphas == d:
        if indirect_params["verbose"]:
            print('square case')
        alphas = linalg.solve(M, z)
    else:  # system of equations is either over or under-determined
        if indirect_params["verbose"]:
            print('non-square case')
        alphas = linalg.lstsq(M, z)[0]

    return alphas


def solve_primal(grid_check, Y_grid, z, p):
    """Wrapper for solver of primal problem.

            Args:
                grid_check (list): grid of true anomalies where norm of candidate primer vector is compared to 1 to
                check convergence.
                Y_grid (numpy.array): grid of moment-function components for norm evaluation of primer vector.
                z (numpy.array): right-hand side of moment equation.
                p (int): type of norm to be minimized.

            Returns:
                 lamb (numpy.array): coefficients of primer vector.

    """

    # sanity check(s)
    if (p != 1) and (p != 2):
        print('SOLVE_PRIMAL: norm in cost function must be 1 or 2')

    if p == 1:
        lamb = solve_primal_1norm(grid_check, Y_grid, z)
    else:  # p = 2
        lamb = solve_primal_2norm(grid_check, Y_grid, z)

    return lamb


def primal_to_dual(grid_check, Y_grid, lamb, z, p):
    """Wrapper for retrieving of dual solution from primal one.

            Args:
                grid_check (list): grid of true anomalies where norm of candidate primer vector is compared to 1 to
                check convergence.
                Y_grid (numpy.array): grid of moment-function components for norm evaluation of primer vector.
                lamb (numpy.array): coefficients of primer vector.
                z (numpy.array): right-hand side of moment equation.
                p (int): type of norm to be minimized.

            Returns:
                nus (list): optimal nus of burn.
                DV (numpy.array): corresponding Delta-Vs.

    """

    # sanity check(s)
    if (p != 1) and (p != 2):
        print('PRIMAL_TO_DUAL: norm in cost function must be 1 or 2')

    if p == 1:
        (nus, DV) = primal_to_dual_1norm(grid_check, Y_grid, lamb, z)
    else:  # p = 2
        (nus, DV) = primal_to_dual_2norm(grid_check, Y_grid, lamb, z)

    return nus, DV


def solve_primal_1norm(grid_check, Y_grid, z):
    """Function solving primal problem with 1-norm.

            Args:
                grid_check (list): grid of true anomalies where norm of candidate primer vector is compared to 1 to
                check convergence.
                Y_grid (numpy.array): grid of moment-function components for norm evaluation of primer vector.
                z (numpy.array): right-hand side of moment equation.

            Returns:
                lamb (numpy.array): coefficients of primer vector.

    """

    # pre-computations
    d = len(z)
    hd = d / 2

    # initializing sparser grid for norm checks within optimization
    (grid_work, indices_work) = initialize_iterative_grid(grid_check)
    n_work = len(grid_work)

    converged = False
    iterations = 1
    lamb = None
    res = None
    while (not converged) and (iterations < indirect_params["max_iter"]):

        # building matrix for linear constraints
        A = numpy.zeros((d * n_work, d))
        for j in range(0, len(grid_work)):
            tY = numpy.transpose(Y_grid[:, hd * indices_work[j]: hd * (indices_work[j] + 1)])
            A[d * j: d * j + hd, :] = tY
            A[d * j + hd: d * (j + 1), :] = -tY

        res = linprog(-z, A_ub=A, b_ub=numpy.ones(d * n_work), bounds=(-numpy.inf, numpy.inf),
                      options={"disp": indirect_params["verbose"], "tol": indirect_params["tol_lin_prog"]})
        lamb = res.x

        (converged, index_max) = find_max_pv(Y_grid, lamb, numpy.inf)
        if not converged:
            iterations += 1
            if indirect_params["exchange"]:
                (grid_work, indices_work) = remove_nus(Y_grid, numpy.inf, grid_work, indices_work, lamb)
            grid_work.append(grid_check[index_max])  # add nu
            indices_work.append(index_max)
            n_work = len(grid_work)

        else:  # algorithm has converged
            if indirect_params["verbose"]:
                print('converged with ' + str(n_work) + ' points at iteration ' + str(iterations))

    if indirect_params["verbose"]:
        print('primal numerical cost 1-norm: ' + str(-res.fun))

    return lamb


def primal_to_dual_1norm(grid_check, Y_grid, lamb, z):
    """Function retrieving solution of dual problem from primal one with 1-norm.

            Args:
                grid_check (list): grid of true anomalies where norm of candidate primer vector is compared to 1 to
                check convergence.
                Y_grid (numpy.array): grid of moment-function components for norm evaluation of primer vector.
                lamb (numpy.array): coefficients of primer vector.
                z (numpy.array): right-hand side of moment equation.

            Returns:
                nus (list): optimal nus of burn.
                DVs (numpy.array): corresponding Delta-Vs.

    """

    # pre-computations
    d = len(z)
    hd = int(d / 2)

    # extracting optimal nus from primer vector
    (nus, indices) = extract_nus(grid_check, Y_grid, lamb, numpy.inf)

    # extracting optimal directions of burn from primer vector
    directions = numpy.zeros((len(nus), hd))
    n_alphas = 0
    for i in range(0, len(nus)):
        inter = numpy.transpose(Y_grid[:, hd * indices[i]: hd * (indices[i] + 1)]) . dot(lamb)
        for j in range(0, hd):
            if math.fabs(inter[j]) > 1.0 - indirect_params["tol_unit_norm"]:
                directions[i, j] = numpy.sign(inter[j])
                n_alphas += 1

        # building matrix for linear system
    M = numpy.zeros((d, n_alphas))
    count = 0
    for i in range(0, len(nus)):
        aux = Y_grid[:, hd * indices[i]: hd * (indices[i] + 1)]
        for j in range(0, hd):
            if math.fabs(directions[i, j]) > 0.0:
                direc = directions[i, j]
                M[:, count] = direc * aux[:, j]
                count += 1

    # solve for the alphas
    alphas = solve_alphas(M, z, n_alphas)
    if indirect_params["verbose"]:
        print('dual numerical cost 1-norm : ' + str(sum(alphas)))

    # reconstructing velocity jumps
    DVs = numpy.zeros((len(nus), hd))
    count = 0
    for i in range(0, len(nus)):
        for j in range(0, hd):
            if math.fabs(directions[i, j]) > 0.0:
                DVs[i, j] = directions[i, j] * alphas[count]
                count += 1

    return nus, DVs


def solve_primal_2norm(grid_check, Y_grid, z):
    """Function solving primal problem with 2-norm.

            Args:
                grid_check (list): grid of true anomalies where norm of candidate primer vector is compared to 1 to
                check convergence.
                Y_grid (numpy.array): grid of moment-function components for norm evaluation of primer vector.
                z (numpy.array): right-hand side of moment equation.

            Returns:
                lamb (numpy.array): coefficients of primer vector.

    """

    # pre-computations
    d = len(z)
    hd = d / 2

    # initializing sparser grid for norm checks within optimization
    (grid_work, indices_work) = initialize_iterative_grid(grid_check)
    n_work = len(grid_work)

    converged = False
    iterations = 1
    solvers.options['show_progress'] = indirect_params["verbose"]
    solvers.options['abstol'] = indirect_params["tol_cvx"]
    lamb = numpy.zeros(d)
    while (not converged) and (iterations < indirect_params["max_iter"]):

        # building matrices for SDP constraints
        A = None
        h = None
        for j in range(0, len(grid_work)):
            Y = Y_grid[:, hd * indices_work[j]: hd * (indices_work[j] + 1)]
            if hd == 1:
                B = [[0.0, Y[0, 0], Y[0, 0], 0.0], [0.0, Y[1, 0], Y[1, 0], 0.0]]
            elif hd == 2:
                B = [[0.0, Y[0, 0], Y[0, 1], Y[0, 0], 0.0, 0.0, Y[0, 1], 0.0, 0.0],
                     [0.0, Y[1, 0], Y[1, 1], Y[1, 0], 0.0, 0.0, Y[1, 1], 0.0, 0.0],
                     [0.0, Y[2, 0], Y[2, 1], Y[2, 0], 0.0, 0.0, Y[2, 1], 0.0, 0.0],
                     [0.0, Y[3, 0], Y[3, 1], Y[3, 0], 0.0, 0.0, Y[3, 1], 0.0, 0.0]]
            else:  # case of complete dynamics
                B = [[0.0, Y[0, 0], Y[0, 1], Y[0, 2], Y[0, 0], 0.0, 0.0, 0.0, Y[0, 1], 0.0, 0.0, 0.0, Y[0, 2], 0.0, 0.0, 0.0],
                     [0.0, Y[1, 0], Y[1, 1], Y[1, 2], Y[1, 0], 0.0, 0.0, 0.0, Y[1, 1], 0.0, 0.0, 0.0, Y[1, 2], 0.0, 0.0, 0.0],
                     [0.0, Y[2, 0], Y[2, 1], Y[2, 2], Y[2, 0], 0.0, 0.0, 0.0, Y[2, 1], 0.0, 0.0, 0.0, Y[2, 2], 0.0, 0.0, 0.0],
                     [0.0, Y[3, 0], Y[3, 1], Y[3, 2], Y[3, 0], 0.0, 0.0, 0.0, Y[3, 1], 0.0, 0.0, 0.0, Y[3, 2], 0.0, 0.0, 0.0],
                     [0.0, Y[4, 0], Y[4, 1], Y[4, 2], Y[4, 0], 0.0, 0.0, 0.0, Y[4, 1], 0.0, 0.0, 0.0, Y[4, 2], 0.0, 0.0, 0.0],
                     [0.0, Y[5, 0], Y[5, 1], Y[5, 2], Y[5, 0], 0.0, 0.0, 0.0, Y[5, 1], 0.0, 0.0, 0.0, Y[5, 2], 0.0, 0.0, 0.0]]
            if j == 0:
                A = [-matrix(B)]
                h = [matrix(numpy.eye(hd + 1))]
            else:  # A and h are already not None
                A += [-matrix(B)]
                h += [matrix(numpy.eye(hd + 1))]

        solution = solvers.sdp(matrix(-z), Gs=A, hs=h)
        x = numpy.array(solution['x'])

        for k, x_k in enumerate(x):
            lamb[k] = x_k[0]

        (converged, index_max) = find_max_pv(Y_grid, lamb, 2)
        if not converged:
            iterations += 1
            if indirect_params["exchange"]:
                (grid_work, indices_work) = remove_nus(Y_grid, 2, grid_work, indices_work, lamb)
            grid_work.append(grid_check[index_max])  # add nu
            indices_work.append(index_max)
            n_work = len(grid_work)
        else:  # algorithm has converged
            if indirect_params["verbose"]:
                print('converged with ' + str(n_work) + ' points at iteration ' + str(iterations))

    if indirect_params["verbose"]:
        print('primal numerical cost 2-norm: ' + str(z.dot(lamb)))

    return lamb


def primal_to_dual_2norm(grid_check, Y_grid, lamb, z):
    """Function retrieving solution of dual problem from primal one with 2-norm.

            Args:
                grid_check (list): grid of true anomalies where norm of candidate primer vector is compared to 1 to
                check convergence.
                Y_grid (numpy.array): grid of moment-function components for norm evaluation of primer vector.
                lamb (numpy.array): coefficients of primer vector.
                z (numpy.array): right-hand side of moment equation.

            Returns:
                nus (list): optimal nus of burn.
                DVs (numpy.array): corresponding Delta-Vs.

    """

    # pre-computations
    d = len(z)
    hd = int(d / 2)

    # extracting optimal nus from primer vector
    p = 2
    (nus, indices) = extract_nus(grid_check, Y_grid, lamb, p)

    # building matrix for linear system
    M = numpy.zeros((d, len(nus)))
    directions = numpy.zeros((len(nus), hd))
    for i in range(0, len(nus)):
        aux = Y_grid[:, hd * indices[i]: hd * (indices[i] + 1)]
        inter = numpy.transpose(aux) . dot(lamb)
        for j in range(0, len(inter)):
            directions[i, j] = inter[j] / linalg.norm(inter, p)
            M[:, i] += aux[:, j] * directions[i, j]

    alphas = solve_alphas(M, z, len(nus))
    if indirect_params["verbose"]:
        print('dual numerical cost 2-norm : ' + str(sum(alphas)))

    # reconstructing velocity jumps
    DVs = numpy.zeros((len(nus), hd))
    for i in range(0, len(nus)):
        for j in range(0, hd):
            DVs[i, j] = directions[i, j] * alphas[i]

    return nus, DVs
