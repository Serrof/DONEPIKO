# tuning_params.py: set of tuning parameters for algorithms and other
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

# general solver
verbose = True  # set to False for no on-screen information

# indirect _solver
max_iter = 30  # maximum number of iterations to solve primal problem
eps = 1.0e-4  # tolerance for unit norm
n_init = 2  # number of points on initial grid if the latter is randomly generated
exchange = True  # set to True to remove dates at each iteration where constraints on primer norm are not saturated
tol_linprog_ind = 1.e-11  # tolerance for solving linear programs
tol_cvx_ind = 1.e-11  # absolute tolerance for solving SDPs
n_check = 10000  # number of points on which _q-norm of primer vector is checked

# direct _solver
DV_min = 1.e-10  # numerical threshold to delete impulses in post-process
n_grid_1norm = 10000  # number of points for direct _solver in 1-norm
n_grid_2norm = 100  # number of points for direct _solver in 2-norm
tol_linprog_dir = 1.e-11  # absolute tolerance for solving linear programs
tol_cvx_dir = 1.e-11  # absolute tolerance for solving SDPs

# plotter
mesh_plot = 2000  # mesh size for plots with linear dynamics
font = 22  # font-size for plots
h_min = 0.001  # minimum step-size in radians for integration of nonlinear dynamics

# orbital mechanics
tol_gamma_LP = 1.e-8  # tolerance for computing L1, 2 and 3
