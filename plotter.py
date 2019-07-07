# plotter.py: class handling the plots
# Copyright(C) 2019 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy
from numpy import linalg
import utils
import dynamical_system
import body_prob_dyn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import integrators
import math
import indirect_num
from config import conf

plt.rcParams.update({'font.size': conf.params_plot["font"]})


class Plotter:
    """Class dealing with the plotting capacities.

                Attributes:
                    dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.
                    p (int): type of norm for control law to be plotted.
                    anomaly (bool): set to True if independent variable is the true anomaly and to False if it is time.
                    linearized (bool): set to True to plot linearized dynamics, False otherwise
                    analytical (bool): set to True to propagate the state vector analytically if possible, False otherwise
                    CL (utils.ControlLaw): control law to be simulated.
                    _q (int): type of norm for primer vector.
                    _nb (int): number of points to be plotted.
                    _nus (list): anomalies to be plotted.
                    _times (list): instants to be plotted.
                    _pts (list): values of independent variable to be plotted (depending on boolean 'anomaly')
                    _states (numpy.array): states to be plotted.

    """

    def __init__(self, dyn, BC, p, anomaly, linearized, analytical, CL=None):
        """Constructor for class plotter.

                Args:
                    dyn (dynamical_system.DynamicalSystem): dynamics to be used for two-boundary value problem.
                    BC (utils.BoundaryConditions): constraints for two-point boundary value problem.
                    p (int): type of norm for control law to be plotted.
                    anomaly (bool): set to True if independent variable is the true anomaly and to False if it is time.
                    linearized (bool): set to True to plot linearized dynamics, False otherwise
                    analytical (bool): set to True to propagate the state vector analytically if possible, False otherwise
                    CL (utils.ControlLaw): control law to be simulated.

        """

        self.p = p
        self._q = indirect_num.dual_to_primal_norm_type(p)
        self.dyn = dyn.copy()
        self.BC = BC.copy()
        if CL is None:
            self.CL = utils.NoControl(BC.half_dim)
        self.linearized = linearized
        self.anomaly = anomaly
        self.analytical = analytical
        if isinstance(dyn, body_prob_dyn.BodyProbDyn):
            # propagation has to be numerical for elliptical out-of-plane L1, 2 or 3 or elliptical in-plane of any LP
            if analytical and dyn.params.mu != 0. and dyn.params.ecc != 0. and \
                (dyn.params.Li == 1 or dyn.params.Li == 2 or dyn.params.Li == 3 or BC.half_dim > 1):
                print('WARNING: propagation type within plotter changed to numerical')
                self.analytical = False

        self._nb = conf.params_plot["mesh_plot"]
        self._nus = None
        self._times = None
        self._pts = None
        self._compute_points()

        self._states = None

    def copy(self):
        """Function returning a copy of the object.

        """

        return Plotter(self.dyn, self.BC, self.p, self.anomaly, self.linearized, self.analytical, self.CL)

    def _compute_points(self):
        """Function generating subsequent values of independent variables used in history of state vector.

        """

        if self.anomaly:    
            self._nus = numpy.linspace(self.BC.nu0, self.BC.nuf, self._nb)
            self._pts = self._nus
        else:  # the independent variable is time
            self._times = numpy.linspace(0., self.dyn.convToAlterIndVar(self.BC.nu0, 0., self.BC.nuf), self._nb)
            self._pts = self._times
            self._nus = numpy.zeros(self._nb)
            for k in range(0, self._nb):
                self._nus[k] = self.dyn.convFromAlterIndVar(self.BC.nu0, 0., self._times[k])

    def set_ind_var(self, anomaly):
        """Setter for attribute anomaly.

                Args:
                     anomaly (bool): set to True if independent variable is the true anomaly and to False if it is time.

        """

        self.anomaly = anomaly
        self._compute_points()

    def set_linearity(self, linearized):
        """Setter for attribute linearized.

                Args:
                     linearized (bool): set to True to plot linearized dynamics, False otherwise

        """

        if linearized != self.linearized:
            self.linearized = linearized
            self._states = None

    def set_norm(self, p):
        """Setter for attribute p.

                Args:
                     p (int): type of norm to be plotted for control law.

        """

        self.p = p
        self._q = indirect_num.dual_to_primal_norm_type(p)

    def set_propagation(self, analytical):
        """Setter for attribute prop_ana.

                Args:
                     analytical (bool): set to true for analytical propagation of motion, false for integration.

        """

        self.analytical = analytical

    def set_boundary_cond(self, BC):
        """Setter for attribute BC.

                Args:
                     BC (utils.BoundaryConditions): constraints for two-point boundary value problem.

        """

        self.BC = BC.copy()
        self._states = None

    def set_control_law(self, CL):
        """Setter for attribute CL.

                Args:
                     CL (utils.ControlLaw): control law to be simulated.

        """

        self.CL = CL.copy()
        if self._states is not None:
            self._compute_states()

    def _compute_states(self):
        """Function computing the history of state variables according to control law.

        """

        dim = self.BC.half_dim * 2
        
        if self.linearized and self.analytical:
            states = numpy.zeros((dim, self._nb))
            inters = numpy.zeros((self.CL.N, dim))
            for i in range(0, self.CL.N):
                if self.BC.half_dim == 1:
                    inters[i, 1] = self.CL.DVs[i]
                else:  # in-plane or complete dynamics
                    inters[i, self.BC.half_dim:dim] = self.CL.DVs[i, :]
            for k, nu in enumerate(self._nus):
                states[:, k] = self.dyn.propagate(self.BC.nu0, nu, self.BC.x0)         
                for i, date in enumerate(self.CL.nus):
                    if nu > date:
                        states[:, k] += self.dyn.propagate(date, nu, inters[i, :])
                    elif nu == date:
                        if self.BC.half_dim == 1:
                            states[1, k] += self.CL.DVs[i]
                        else:  # in-plane or complete dynamics
                            states[self.BC.half_dim:dim, k] += self.CL.DVs[i, :]
            self._states = states

        else:  # dynamics for plots has to be numerically simulated
            if self.BC.half_dim == 1 and not self.linearized:
                return ValueError('_compute_states: non-linear dynamics cannot be only out-of-plane')

            else:  # linearized dynamics or in-plane or complete non-linear dynamics

                if self.linearized:

                    def func(nu, x):  # right-hand side function for integration
                        return self.dyn.evaluate_state_deriv(nu, x)

                else:  # non-linear dynamics

                    def func(nu, x):  # right-hand side function for integration
                        return numpy.array(self.dyn.evaluate_state_deriv_nonlin(nu, x))

                integrator = integrators.ABM8(func)

                def propagate_num(nu1, nu2, IC):
                    if nu1 == nu2:
                        pts_inter = [nu1]
                        state0 = numpy.array(IC[:])
                        states_inter = [state0]
                        return states_inter, pts_inter
                    else:  # initial and final true anomaly are different
                        IC_transformed = self.dyn.transformation(IC, nu1)
                        n_int = int(math.ceil((nu2 - nu1) / conf.params_plot["h_min"]))
                        (states_transformed, pts_inter) = integrator.integrate(nu1, nu2, IC_transformed, n_int)
                        states_inter = []
                        for i, state in enumerate(states_transformed):
                            states_inter.append(self.dyn.transformation_inv(state, pts_inter[i]))
                        return states_inter, pts_inter

                states = []
                pts = []
                for k in range(0, self.CL.N):
                    if k == 0:
                        state0 = numpy.array(self.BC.x0[:])
                        date0 = self.BC.nu0
                        datef = self.CL.nus[0]
                    else:  # not first loop
                        state0 = numpy.array(states[-1])
                        date0 = pts[-1]
                        datef = self.CL.nus[k]

                    (states_inter, pts_inter) = propagate_num(date0, datef, state0)
                    states_inter[-1][self.BC.half_dim:2*self.BC.half_dim] += self.CL.DVs[k, :]
                    
                    if len(pts_inter) == 1:
                        pts.append(pts_inter[0])
                        states.append(states_inter[0])
                    else:  # not first date
                        for i in range(1, len(pts_inter)):
                            pts.append(pts_inter[i])
                            states.append(states_inter[i])

                if self.BC.nuf != self.CL.nus[-1]:
                    (states_inter, pts_inter) = propagate_num(self.CL.nus[-1], self.BC.nuf, states[-1])

                    for i in range(1, len(pts_inter)):
                        pts.append(pts_inter[i])
                        states.append(states_inter[i])

                self._nus = pts
                self._nb = len(self._nus)
                if self.anomaly:
                    self._pts = pts
                else:  # the independent variable is time
                    self._pts = []
                    for nu in self._nus:
                        self._pts.append(self.dyn.convToAlterIndVar(self.BC.nu0, 0., nu))
                self._states = numpy.array(states[:]).transpose()

    def plot_pv(self):
        """Function plotting primer vector's components and norm as functions of the independent variable.

        """

        # plotting position and velocity as functions of the independent variable
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # generating primer vector 
        pv = numpy.zeros((self.BC.half_dim, self._nb))
        pv_norm = []
        if self.analytical:
            for k in range(0, self._nb):
                pv[:, k] = numpy.transpose(self.dyn.evaluate_Y(self._nus[k], self.BC.half_dim)).dot(self.CL.lamb)
        else:
            Ys = self.dyn.integrate_Y(self._nus, self.BC.half_dim)
            for k in range(0, self._nb):
                pv[:, k] = numpy.transpose(Ys[:, k * self.BC.half_dim: (k + 1) * self.BC.half_dim]).dot(self.CL.lamb)

        for k in range(0, self._nb):
            pv_norm.append(linalg.norm(pv[:, k], self._q))

        # plotting primer vector 
        min_pv = numpy.min(pv[0, :])
        max_pv = numpy.max(pv[0, :])    
        if self.BC.half_dim == 1:
            ax1.plot(self._pts, pv[0, :], color='black', ls='dashed', label='$\delta z$-axis', linewidth=2)
        else:  # in-plane or complete dynamics
            min_pv = min(numpy.min(pv[1, :]), min_pv)
            max_pv = max(numpy.max(pv[1, :]), max_pv)
            if self.BC.half_dim == 2:
                ax1.plot(self._pts, pv[0, :], color='blue', ls='dashed', label='$\delta x$-axis', linewidth=2)
                ax1.plot(self._pts, pv[1, :], color='red', ls='dashed', label='$\delta y$-axis', linewidth=2)
            else:  # complete dynamics
                ax1.plot(self._pts, pv[0, :], color='blue', ls='dashed', label='$\delta x$-axis', linewidth=2)
                ax1.plot(self._pts, pv[1, :], color='red', ls='dashed', label='$\delta y$-axis', linewidth=2)
                ax1.plot(self._pts, pv[2, :], color='black', ls='dashed', label='$\delta z$-axis', linewidth=2)
                min_pv = min(numpy.min(pv[2, :]), min_pv)
                max_pv = max(numpy.max(pv[2, :]), max_pv)

        ax2.plot(self._pts, pv_norm, color='green', linewidth=2)

        ax1.set_ylabel('components')
        ax1.set_xlim([self._pts[0], self._pts[-1]])
        ax1.set_ylim([min_pv, max_pv])
        ax1.grid()
        ax1.legend()
        ax1.set_title('primer vector')

        if self.BC.half_dim == 1:
            ax2.set_ylabel('norm')
        else:  # in-plane or complete dynamics
            ax2.set_ylabel(str(self._q) + '-norm')
        ax2.set_xlim([self._pts[0], self._pts[-1]])
        ax2.set_ylim([0.0, numpy.max(pv_norm)]) 
        if self.anomaly:
            ax2.set_xlabel('true anomaly (rad)')
        else:  # the independent variable is time
            ax2.set_xlabel('time (s)')
        ax2.grid()

    def plot_states(self):
        """Function plotting the history of state variables and fuel-consumption.

        """

        # generating position and velocity 
        if self._states is None:
            self._compute_states()

        # plotting position and velocity as functions of the independent variable
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        index = 0
        color_plot = None
        while index < self.BC.half_dim:
            if self.BC.half_dim == 1:
                color_plot = 'black'    
            elif self.BC.half_dim == 2:
                if index == 0:
                    color_plot = 'blue'
                elif index == 1:
                    color_plot = 'red'
            else:  # complete dynamics
                if index == 0:
                    color_plot = 'blue'
                elif index == 1:
                    color_plot = 'red'
                else:  # index = 2
                    color_plot = 'black'
            ax1.plot([self._pts[0]], [self.BC.x0[index]], marker='o', color=color_plot, markersize=12)
            ax2.plot([self._pts[0]], [self.BC.x0[index + self.BC.half_dim]], marker='o', color=color_plot,
                     markersize=12)
            ax1.plot([self._pts[-1]], [self.BC.xf[index]], marker='o', color=color_plot, markersize=12)
            ax2.plot([self._pts[-1]], [self.BC.xf[index + self.BC.half_dim]], marker='o', color=color_plot,
                     markersize=12)
            ax1.plot(self._pts, self._states[index, :], color_plot, linewidth=2)
            ax2.plot(self._pts, self._states[index + self.BC.half_dim, :], color_plot, linewidth=2)
            index += 1

        if self.linearized:
            ax1.set_title('state vector under linearized dynamics')
        else:  # simulated dynamics is non-linear
            ax1.set_title('state vector under non-linear dynamics')
        ax1.set_xlim([self._pts[0], self._pts[-1]])
        ax2.set_xlim([self._pts[0], self._pts[-1]])
        ax1.grid()
        ax2.grid()
        ax1.set_ylabel('position (m)')
        ax2.set_ylabel('velocity (m/s)')

        cost = [0.]
        if self.CL.nus[0] == self.BC.nu0:
            dates_cost = [self._pts[0] - 1.]
        else:  # first maneuver is not at initial true anomaly
            dates_cost = [self._pts[0]]
        for k in range(0, len(self.CL.nus)):
            if self.BC.half_dim == 1:
                jump = math.fabs(self.CL.DVs[k]) 
            else:  # in-plane or complete dynamics
                jump = linalg.norm(self.CL.DVs[k, :], self.p) 
            cost.append(jump + cost[-1])
            if self.anomaly:
                dates_cost.append(self.CL.nus[k])
            else:  # the independent variable is time
                dates_cost.append(self.dyn.convToAlterIndVar(self.BC.nu0, 0., self.CL.nus[k]))
        if dates_cost[-1] != self._pts[-1]:
            cost.append(cost[-1])
            dates_cost.append(self._pts[-1])

        ax3.step(dates_cost, cost, where='post', color='green', linewidth=2) 
        ax3.set_xlim([self._pts[0], self._pts[-1]])
        ax3.set_ylim([0., cost[-1]])
        if self.anomaly:
            ax3.set_xlabel('true anomaly (rad)')
        else:  # the independent variable is time
            ax3.set_xlabel('time (s)')
        ax3.set_ylabel('cost (m/s)')
        ax3.grid()

    def plot_traj(self):
        """Function plotting the trajectory. In case of pure out-of-plane dynamics, the trajectory is in the phase plane.

        """

        # generating position and velocity 
        if self._states is None:
            self._compute_states()

        # plotting position and velocity as functions of the independent variable
        if self.BC.half_dim == 3:
            fig = plt.figure()
            color_plot = 'green'
            ax = fig.add_subplot(111, projection='3d')
            ax.plot([self.BC.x0[0]], [self.BC.x0[1]], [self.BC.x0[2]], marker='+', color=color_plot)
            ax.plot([self.BC.xf[0]], [self.BC.xf[1]], [self.BC.xf[2]], marker='x', color=color_plot)
            ax.plot(self._states[0, :], self._states[1, :], self._states[2, :], color=color_plot)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')  
        else:  # in-plane or ouf-of-plane dynamics only
            fig, ax = plt.subplots(1, 1)
            if self.BC.half_dim == 1: 
                color_plot = 'black'
            else:   # in-plane dynamics
                color_plot = 'purple'
            ax.plot([self.BC.x0[0]], [self.BC.x0[1]], marker='+', color=color_plot)
            ax.plot([self.BC.xf[0]], [self.BC.xf[1]], marker='x', color=color_plot)
            ax.plot(self._states[0, :], self._states[1, :], color=color_plot)
            ax.grid()

            if self.BC.half_dim == 1:
                # enforce first jump to be plotted
                ax.plot([self.BC.x0[0], self.BC.x0[0]], [self.BC.x0[1], self._states[1, 0]], color=color_plot)

        if self.BC.half_dim == 1:
            ax.set_title('phase plane')
        else:  # in-plane or complete dynamics
            ax.set_title('trajectory')

    def show(self):
        """Function to show all the pre-computed plots.

        """
        plt.show()

    def close(self):
        """Function to close all the plots.

        """
        plt.close("all")

    def write_states_to_file(self, file_path):
        """Function writing in a file the history of the states' variables.

            Args:
                file_path (str): The path to create/overwrite the state history.

        """

        if self._states is not None:
            numpy.savetxt(file_path, self._states)
        else:  # state history has not been computed yet
            self._compute_states()
            numpy.savetxt(file_path, self._states)
