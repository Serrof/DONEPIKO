# master.py: class for the middle-man between solving and plotting
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import plotter
import utils
import indirect_solver
import direct_solver


class Master:
    """Class implementing a so-called master for solving and plotting fuel-optimal trajectories.

            Attributes:
                BC (utils.BoundaryConditions): boundary conditions.
                plotter (plotter.Plotter): plotter object for plotting.
                CL (utils.ControlLaw): control law.
                _solver (solver.Solver): trajectory solver.

    """

    def __init__(self, indirect, p, plr):
        """Constructor for class master.

                Args:
                    indirect (bool): set to True for indirect approach and False for direct one.
                    p (int): type of norm to be minimized.
                    plr (plotter.Plotter): plotter object for plotting.

        """

        self.BC = plr.BC.copy()
        self.plotter = plr.copy()
        self.CL = utils.NoControl(self.BC)
        if indirect:
            self._solver = indirect_solver.IndirectSolver(self.plotter.dyn, p, plr.analytical)
        else:  # direct approach chosen
            self._solver = direct_solver.DirectSolver(self.plotter.dyn, p, plr.analytical)

    def set_norm_solve(self, p):
        """Function to reset type of norm to be minimized.

                Args:
                    p (int): type of norm to be minimized.

        """

        self._solver.set_norm(p)

    def set_norm_plot(self, p):
        """Function to reset type of norm to be plotted.

                Args:
                    p (int): type of norm to be plotted.

        """

        self.plotter.set_norm(p)

    def set_linearity_plot(self, linearized):
        """Function to reset linear property of dynamics to be plotted.

                Args:
                    linearized (bool): set to True for linearized dynamics, False otherwise.

        """

        self.plotter.set_linearity(linearized)

    def set_propagation(self, analytical):
        """Function to change the propagation type (analytical or numerical) of both plotter and solver.

                Args:
                    analytical (bool): set to true for analytical propagation of motion, false for integration.

        """

        self.plotter.set_propagation(analytical)
        self._solver.set_propagation(analytical)

    def set_boundary_cond(self, BC):
        """Setter for attribute BC.

                Args:
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

        """

        self.BC = BC.copy()
        self.plotter.set_boundary_cond(BC)

    def set_approach(self, indirect):
        """Function to reset type of approach for optimization.

                Args:
                    indirect (bool): set to True for indirect approach and False for direct one.

        """

        if indirect:
            self._solver = indirect_solver.IndirectSolver(self._solver.dyn, self._solver.p, self._solver.prop_ana)
        else:  # direct approach chosen
            self._solver = direct_solver.DirectSolver(self._solver.dyn, self._solver.p, self._solver.prop_ana)

    def set_control(self, CL):
        """Setter for the utils.ControlLaw attribute.

                Args:
                    CL (utils.ControlLaw): control law to be simulated.

        """

        self.CL = CL.copy()
        self.plotter.set_control_law(CL)

    def suboptimize(self):
        """Wrapper to solve for a two-impulses trajectory (at initial and final anomalies).


        """

        self.set_control(self._solver.boundary_impulses(self.BC))

    def solve(self):
        """Function to compute fuel-optimal solution to two-point boundary value problem.

        """

        self.CL = self._solver.run(self.BC)
        self.plotter.set_control_law(self.CL)

    def plot(self, selected_plots=None):
        """Function to prepare plots of states and trajectory as well as primer vector in case of an indirect approach.

                Args:
                    selected_plots (List[int]): figures to plot (optional, default is all). If 1 is in the list, it
                    will plot the position and velocity history. If 2 is there, it will plot the trajectory. If 3 is
                    there and that the approach is indirect, it will plot the primer vector.

        """

        if selected_plots is None:
            self.plotter.plot_states()
            self.plotter.plot_traj()
            if len(self.CL.lamb) != 0:
                self.plotter.plot_pv()
        else:
            if (1 not in selected_plots) and (2 not in selected_plots) and (3 not in selected_plots):
                raise ValueError("Plot selection is done with a list including 1, 2 or 3.")
            if 1 in selected_plots:
                self.plotter.plot_states()
            if 2 in selected_plots:
                self.plotter.plot_traj()
            if 3 in selected_plots and len(self.CL.lamb) != 0:
                self.plotter.plot_pv()

    @staticmethod
    def show():
        """Function to show all the pre-computed plots.

        """
        plotter.Plotter.show()

    @staticmethod
    def close():
        """Function to close all the plots.

        """
        plotter.Plotter.close()

    def write_control_law(self, file_path):
        """Wrapper for writer-method of ControlLaw attribute.

                Args:
                    file_path (str): path of file where to write control law.

        """

        self.CL.write_to_file(file_path)

    def write_boundary_cond(self, file_path):
        """Wrapper for writer-method of BoundaryConditions attribute.

                Args:
                    file_path (str): path of file where to write boundary conditions.

        """

        self.BC.write_to_file(file_path)

    def write_states(self, file_path):
        """Wrapper for writer-method of Plotter attribute.

                Args:
                    file_path (str): path of file where to write states.

        """

        self.plotter.write_states_to_file(file_path)
