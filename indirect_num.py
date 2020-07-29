# indirect_num.py: functions involved in the numerical solving by the indirect approach
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import math
import random
import numpy as np
from numpy import linalg
from scipy.optimize import linprog
from cvxopt import matrix, solvers
from config import conf


def dual_to_primal_norm_type(p):
    """Function returning the type of primal norm given a dual one.

            Args:
                p (int): type of dual norm.

            Returns:
                (int): type of primal norm.

    """

    if p == 1:
        return np.inf
    elif p == 2:
        return 2
    else:
        raise ValueError('dual_to_primal_norm_type: dual norm must be 1 or 2')


def initialize_iterative_grid(grid_input):
    """Function initializing the grid for the relaxed primal problem from the thinner grid used for post-treatment.

            Args:
                grid_input (List[float]): grid of true anomalies used for norm checks of primer vector.

            Returns:
                 (List[float]): sub-grid to use for first iteration in the solving of the primal problem (norm
                 constraints).
                 (List[int]): indices of points selected in input grid.

    """

    d_nu = grid_input[-1] - grid_input[0]
    if d_nu % math.pi != 0.:
        # only initial and final points of grid are kept
        indices = [0, len(grid_input)-1]
        grid = [grid_input[indices[0]], grid_input[indices[1]]]
        return grid, indices
    else:  # flight duration (measured in radians) is not equal to 0 modulo pi
        return initialize_iterative_grid_randomly(conf.params_indirect["n_init"], grid_input)


def initialize_iterative_grid_randomly(n_points, grid):
    """Function randomly picking points in the grid for norm checks.

            Args:
                n_points (int): number of points to be selected in input grid.
                grid (List[float]): grid of true anomalies used for norm checks of primer vector.

            Returns:
                points (List[float]): sub-grid whose points were randomly picked among input grid.
                indices (List[int]): indices of points randomly picked in input grid.

    """

    indices = [random.randint(0, len(grid) - 1)]
    points = [grid[indices[0]]]

    while len(points) < n_points:
        index_drawn = random.randint(0, len(grid) - 1)
        if index_drawn not in indices:  # true anomaly has not been selected yet
            points.append(grid[index_drawn])
            indices.append(index_drawn)

    return points, indices


def find_max_pv(Y_grid, lamb, q):
    """Function checking if the maximum q-norm of the primer vector on a grid is less than one.

            Args:
                Y_grid (np.array): grid of moment-function components for norm evaluation of primer vector.
                lamb (np.array): coefficients of primer vector.
                q (int): norm for primer vector.

            Returns:
                unit_norm (bool): True if maximum norm is smaller than one.
                index_max (int): index where maximum norm is reached.

    """

    hd = int(len(lamb) / 2)
    n_check = int(len(Y_grid[0, :]) / hd)
    norms = np.zeros(n_check)
    for k in range(0, n_check):
        Y_k = Y_grid[:, hd * k: hd * (k + 1)]
        norms[k] = linalg.norm(np.transpose(Y_k).dot(lamb), q)
    index_max = np.argmax(norms)
    unit_norm = (norms[index_max] <= 1.0 + conf.params_indirect["tol_unit_norm"])

    return unit_norm, index_max


def remove_nus(Y_grid, q, grid_work, indices_work, lamb):
    """Function removing from a grid the true anomalies where the candidate primer vector has a norm smaller than one.

            Args:
                Y_grid (np.array): grid of moment-function components for norm computation of candidate primer vector.
                q (int): type of norm for primer vector.
                grid_work (List[float]): input grid where to trim true anomalies where norm of primer vector is smaller
                than 1.
                indices_work (List[int]): indices of elements of grid_work in thinner grid of true anomalies used for
                post-process norm checks.
                lamb (np.array): coefficients of candidate primer vector.

            Returns:
                 grid (List[float]): filtered grid w.r.t. input one.
                 indices (List[int]): indices in input grid of true anomalies kept in output grid .

    """

    hd = int(len(lamb) / 2)
    grid = list(grid_work)
    indices = list(indices_work)
    removed_nus = 0
    for k in range(0, len(grid)):
        pv = np.transpose(Y_grid[:, hd * indices[k - removed_nus]: hd * (indices[k - removed_nus] + 1)]).dot(lamb)
        pv_norm = linalg.norm(pv, q)
        if pv_norm < 1.0 - conf.params_indirect["tol_unit_norm"]:
            del grid[k - removed_nus]
            del indices[k - removed_nus]
            removed_nus += 1

    return grid, indices


def extract_nus(grid_check, Y_grid, lamb, q):
    """Function selecting points of input grid where primer vector has a norm greater than one.

            Args:
                grid_check (List[float]): grid of true anomalies where norm of candidate primer vector is compared to 1
                to check convergence.
                Y_grid (np.array): grid of moment-function components for norm evaluation of candidate primer vector.
                lamb (np.array): coefficients of candidate primer vector.
                q (int): norm for primer vector.

            Returns:
                nus (List[float]): points of grid_check where norm of candidate primer vector is greater than one.
                indices (List[int]): indices of input grid corresponding to output n_s

    """

    # pre-computations
    d = len(lamb)
    hd = int(d / 2)
    n_check = len(grid_check)

    # extracting optimal nus from primer vector
    nus = []
    indices = []
    k = 0
    while k < n_check:
        Y_k = Y_grid[:, hd * k: hd * (k + 1)]
        inter = linalg.norm(np.transpose(Y_k).dot(lamb), q)
        if 1.0 - conf.params_indirect["tol_unit_norm"] < inter:
            k += 1
            i_nu = k - 1
            # skip following nus for which norm is almost one
            if k < n_check:
                lap = True
                while lap:
                    Y_k = Y_grid[:, hd * k: hd * (k + 1)]
                    inter2 = linalg.norm(np.transpose(Y_k).dot(lamb), q)
                    if 1.0 - conf.params_indirect["tol_unit_norm"] < inter2:
                        if k == n_check - 1:
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
                M (np.array): matrix such that M * alphas = z.
                z (np.array): right-hand side of moment equation.
                n_alphas (int): number of impulses (p=2) or of non-zero components of impulses (p=1).

            Returns:
                alphas (np.array): vector made of magnitudes of impulses (p=2) or impulses' components (p=1).

    """

    d = len(z)
    if n_alphas == d:
        if conf.params_other["verbose"]:
            print('square case')
        alphas = linalg.solve(M, z)
    else:  # system of equations is either over or under-determined
        if conf.params_other["verbose"]:
            print('non-square case')
        alphas = linalg.lstsq(M, z, rcond=None)[0]

    if np.min(alphas) < 0.:
        raise ValueError("A non-positive delta-V norm was found. Try a grid with more points and/or smaller epsilon.")

    return alphas


def solve_primal(grid_check, Y_grid, z, p):
    """Wrapper for solver of primal problem.

            Args:
                grid_check (List[float]): grid of true anomalies where norm of candidate primer vector is compared to 1
                to check convergence.
                Y_grid (np.array): grid of moment-function components for norm evaluation of primer vector.
                z (np.array): right-hand side of moment equation.
                p (int): type of norm to be minimized.

            Returns:
                 lamb (np.array): coefficients of primer vector.

    """

    # sanity check(s)
    if (p != 1) and (p != 2):
        raise ValueError('SOLVE_PRIMAL: norm in cost function must be 1 or 2')

    if p == 1:
        lamb = solve_primal_1norm(grid_check, Y_grid, z)
    else:  # p = 2
        lamb = solve_primal_2norm(grid_check, Y_grid, z)

    return lamb


def primal_to_dual(grid_check, Y_grid, lamb, z, p):
    """Wrapper for retrieving of dual solution from primal one.

            Args:
                grid_check (List[float]): grid of true anomalies where norm of candidate primer vector is compared to 1
                to check convergence.
                Y_grid (np.array): grid of moment-function components for norm evaluation of primer vector.
                lamb (np.array): coefficients of primer vector.
                z (np.array): right-hand side of moment equation.
                p (int): type of norm to be minimized.

            Returns:
                nus (List[float]): optimal nus of burn.
                DV (np.array): corresponding Delta-Vs.

    """

    # sanity check(s)
    if (p != 1) and (p != 2):
        raise ValueError('PRIMAL_TO_DUAL: norm in cost function must be 1 or 2')

    if p == 1:
        (nus, DV) = primal_to_dual_1norm(grid_check, Y_grid, lamb, z)
    else:  # p = 2
        (nus, DV) = primal_to_dual_2norm(grid_check, Y_grid, lamb, z)

    return nus, DV


def solve_primal_1norm(grid_check, Y_grid, z):
    """Function solving primal problem with 1-norm.

            Args:
                grid_check (List[float]): grid of true anomalies where norm of candidate primer vector is compared to 1
                to check convergence.
                Y_grid (np.array): grid of moment-function components for norm evaluation of primer vector.
                z (np.array): right-hand side of moment equation.

            Returns:
                lamb (np.array): coefficients of primer vector.

    """

    # pre-computations
    d = len(z)
    hd = int(d / 2)

    # initializing sparser grid for norm checks within optimization
    (grid_work, indices_work) = initialize_iterative_grid(grid_check)
    n_work = len(grid_work)

    converged = False
    iterations = 1
    lamb = None
    res = None
    while (not converged) and (iterations < conf.params_indirect["max_iter_grid"]):

        # building matrix for linear constraints
        A = np.zeros((d * n_work, d))
        for j, index in enumerate(indices_work):
            tY = np.transpose(Y_grid[:, hd * index: hd * (index + 1)])
            A[d * j: d * j + hd, :] += tY
            A[d * j + hd: d * (j + 1), :] -= tY

        res = linprog(-z, A_ub=A, b_ub=np.ones(d * n_work), bounds=(-np.inf, np.inf),
                      options={"disp": conf.params_other["verbose"], "tol": conf.params_indirect["tol_lin_prog"]})
        lamb = res.x

        (converged, index_max) = find_max_pv(Y_grid, lamb, np.inf)

        if not converged:
            iterations += 1
            if conf.params_indirect["exchange"]:
                (grid_work, indices_work) = remove_nus(Y_grid, np.inf, grid_work, indices_work, lamb)
            grid_work.append(grid_check[index_max])  # add nu
            indices_work.append(index_max)
            n_work = len(grid_work)

        else:  # algorithm has converged
            # check if the solver did converge too
            if not res.success:
                raise InterruptedError("The last iteration on the grid did not lead to a convergent LP. "
                                       "Set verbose to True to see details.")

            else:
                if conf.params_other["verbose"]:
                    print('converged with ' + str(n_work) + ' points at iteration ' + str(iterations))

    if conf.params_other["verbose"]:
        print('primal numerical cost 1-norm: ' + str(-res.fun))

    return lamb


def primal_to_dual_1norm(grid_check, Y_grid, lamb, z):
    """Function retrieving solution of dual problem from primal one with 1-norm.

            Args:
                grid_check (List[float]): grid of true anomalies where norm of candidate primer vector is compared to 1
                to check convergence.
                Y_grid (np.array): grid of moment-function components for norm evaluation of primer vector.
                lamb (np.array): coefficients of primer vector.
                z (np.array): right-hand side of moment equation.

            Returns:
                nus (List[float]): optimal nus of burn.
                DVs (np.array): corresponding Delta-Vs.

    """

    # pre-computations
    d = len(z)
    hd = int(d / 2)

    # extracting optimal nus from primer vector
    (nus, indices) = extract_nus(grid_check, Y_grid, lamb, np.inf)

    # extracting optimal directions of burn from primer vector
    directions = np.zeros((len(nus), hd))
    n_alphas = 0
    for i, index in enumerate(indices):
        for j, el in enumerate(np.transpose(Y_grid[:, hd * index: hd * (index + 1)]).dot(lamb)):
            if math.fabs(el) > 1.0 - conf.params_indirect["tol_unit_norm"]:
                directions[i, j] = np.sign(el)
                n_alphas += 1

    # building matrix for linear system
    M = np.zeros((d, n_alphas))
    count = 0
    for i, index in enumerate(indices):
        aux = Y_grid[:, hd * index: hd * (index + 1)]
        for j in range(0, hd):
            if math.fabs(directions[i, j]) > 0.0:
                M[:, count] = directions[i, j] * aux[:, j]
                count += 1

    # solve for the alphas
    alphas = solve_alphas(M, z, n_alphas)
    if conf.params_other["verbose"]:
        print('dual numerical cost 1-norm : ' + str(sum(alphas)))

    # reconstructing velocity jumps
    DVs = np.zeros((len(nus), hd))
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
                grid_check (List[float]): grid of true anomalies where norm of candidate primer vector is compared to 1
                to check convergence.
                Y_grid (np.array): grid of moment-function components for norm evaluation of primer vector.
                z (np.array): right-hand side of moment equation.

            Returns:
                lamb (np.array): coefficients of primer vector.

    """

    # pre-computations
    d = len(z)
    hd = int(d / 2)

    # initializing sparser grid for norm checks within optimization
    (grid_work, indices_work) = initialize_iterative_grid(grid_check)
    n_work = len(grid_work)

    converged = False
    iterations = 1
    solvers.options["show_progress"] = conf.params_other["verbose"]
    solvers.options["abstol"] = conf.params_indirect["tol_cvx"]
    solvers.options["maxiters"] = conf.params_indirect["max_iter_cvx"]
    lamb = np.zeros(d)
    while (not converged) and (iterations < conf.params_indirect["max_iter_grid"]):

        # building matrices for SDP constraints
        A = None
        h = None
        for j, index in enumerate(indices_work):
            Y = Y_grid[:, hd * index: hd * (index + 1)]
            # construction of matrix in np.array form
            inter = np.zeros((d, (hd + 1) * (hd + 1)))
            inter[:, 1:1+hd] += Y
            for i in range(0, hd):
                inter[:, 1 + hd + i * (hd + 1)] = Y[:, i]
            # conversion to matrix type
            B = matrix([list(el) for el in inter])
            if j == 0:
                A = [-B]
                h = [matrix(np.eye(hd + 1))]
            else:  # A and h are already not None
                A += [-B]
                h += [matrix(np.eye(hd + 1))]

        solution = solvers.sdp(matrix(-z), Gs=A, hs=h)
        x = np.array(solution["x"])
        lamb[:] = x[:, 0]

        (converged, index_max) = find_max_pv(Y_grid, lamb, 2)

        if not converged:
            iterations += 1
            if conf.params_indirect["exchange"]:
                (grid_work, indices_work) = remove_nus(Y_grid, 2, grid_work, indices_work, lamb)
            grid_work.append(grid_check[index_max])  # add nu
            indices_work.append(index_max)
            n_work = len(grid_work)

        else:  # algorithm has converged
            # check if the solver did converge too
            if solution["status"] is not "optimal":
                raise InterruptedError("The last iteration on the grid did not lead to a convergent SDP. "
                                       "Set verbose to True to see details.")
            else:
                if conf.params_other["verbose"]:
                    print('converged with ' + str(n_work) + ' points at iteration ' + str(iterations))

    if conf.params_other["verbose"]:
        print('primal numerical cost 2-norm: ' + str(z.dot(lamb)))

    return lamb


def primal_to_dual_2norm(grid_check, Y_grid, lamb, z):
    """Function retrieving solution of dual problem from primal one with 2-norm.

            Args:
                grid_check (List[float]): grid of true anomalies where norm of candidate primer vector is compared to 1
                to check convergence.
                Y_grid (np.array): grid of moment-function components for norm evaluation of primer vector.
                lamb (np.array): coefficients of primer vector.
                z (np.array): right-hand side of moment equation.

            Returns:
                nus (List[float]): optimal burns' true anomalies.
                DVs (np.array): corresponding Delta-Vs.

    """

    # pre-computations
    d = len(z)
    hd = int(d / 2)

    # extracting optimal nus from primer vector
    p = 2
    (nus, indices) = extract_nus(grid_check, Y_grid, lamb, p)

    # building matrix for linear system
    M = np.zeros((d, len(nus)))
    directions = np.zeros((len(nus), hd))
    for i, index in enumerate(indices):
        aux = Y_grid[:, hd * index: hd * (index + 1)]
        inter = np.transpose(aux).dot(lamb)
        directions[i, :] = inter / linalg.norm(inter, p)
        M[:, i] += aux.dot(directions[i, :])

    alphas = solve_alphas(M, z, len(nus))
    if conf.params_other["verbose"]:
        print('dual numerical cost 2-norm : ' + str(sum(alphas)))

    # reconstructing velocity jumps
    DVs = np.zeros((len(nus), hd))
    for i, alpha in enumerate(alphas):
        DVs[i, :] = directions[i, :] * alpha

    return nus, DVs
