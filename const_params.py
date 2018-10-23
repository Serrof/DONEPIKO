# py: set of tuning parameters for algorithms and other
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

# general tuning parameters for indirect solvers
indirect_params = {
    "verbose": True,  # set to False for no on-screen information
    "max_iter": 30,  # maximum number of iterations to solve primal problem
    "tol_unit_norm": 1.e-4,  # tolerance for unit norm
    "n_init": 2,  # number of points on initial grid if the latter is randomly generated
    "exchange": True,  # set to true to remove dates at each iteration where constraints on primer norm are not saturated
    "tol_lin_prog": 1.e-11,  # tolerance for solving linear programs
    "tol_cvx": 1.e-11,  # absolute tolerance for solving SDPs
    "n_check": 10000  # number of points on which q-norm of primer vector is checked
    }

# general tuning parameters for direct solvers
direct_params = {
    "verbose": True,  # set to False for no on-screen information
    "DV_min": 1.e-10,  # numerical threshold to delete impulses in post-process
    "n_grid_1norm": 10000,  # number of points for direct _solver in 1-norm
    "n_grid_2norm": 100,  # number of points for direct _solver in 2-norm
    "tol_linprog": 1.e-11,  # absolute tolerance for solving linear programs
    "tol_cvx": 1.e-11  # absolute tolerance for solving SDPs
}

# general tuning parameters for plots
plot_params = {
    "mesh_plot": 2000,  # mesh size for plots with linear dynamics
    "font": 22,  # font-size for plots
    "h_min": 1.e-3  # minimum step-size in radians for integration of nonlinear dynamics
}

# parameters for other methods
other_params = {
    "tol_kepler": 1.e-8,  # tolerance for solving Kepler equation with Newton method
    "iter_max_kepler": 100,  # maximum number of iterations when solving Kepler equation
    "tol_gamma_LP": 1.e-8  # tolerance for numerically computing L1, 2 and 3
}

# physical constants related to distances
dist_const = {
    "radius_Earth": 6371.e3,
    "alt_geo": 35786.e3,
    "dist_Earth_Moon": 384400.e37,
    "astro_unit": 149597870.7e3
}
dist_const["radius_geo"] = dist_const["radius_Earth"] + dist_const["alt_geo"]

# physical constants related to masses
mass_const = {
    "mass_Sun": 1.989e30,
    "mass_Earth": 5.972e24,
    "mass_Moon": 7.342e22
}
mass_const["mass_EM"] = mass_const["mass_Earth"] + mass_const["mass_Moon"]
mass_const["mu_EM"] = mass_const["mass_Moon"] / mass_const["mass_EM"]
mass_const["mu_SE"] = mass_const["mass_EM"] / (mass_const["mass_Sun"] + mass_const["mass_EM"])

# physical constants related to gravity
grav_const = {
    "G": 6.674e-11
}
grav_const["Earth_constant"] = grav_const["G"] * mass_const["mass_Earth"]
grav_const["Moon_constant"] = grav_const["G"] * mass_const["mass_Moon"]
grav_const["EM_constant"] = grav_const["G"] * mass_const["mass_EM"]
grav_const["Sun_constant"] = grav_const["G"] * mass_const["mass_Sun"]
