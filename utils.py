# utils.py: set of various functions and classes used by the toolbox
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy
from numpy import linalg
import math


def stack_state(x_ip, x_oop):
    """Function that stacks in-plane and out-of-plane state vectors.

        Args:
            x_ip (numpy.array): The in-plane state vector.
            x_oop (numpy.array): The out-of-plane state vector.

        Returns:
            x (numpy.array): The state vector for the corresponding complete dynamics.

    """

    # sanity check(s)
    if len(x_ip) != 4:
        print('STACK_STATE: in-plane vector must be 4-dimensional')
    if len(x_oop) != 2:
        print('STACK_STATE: out-of-plane vector must be 2-dimensional')

    x = numpy.zeros(6)
    x[0:2] = x_ip[0:2]
    x[2] = x_oop[0]
    x[3:5] = x_ip[2:4]
    x[5] = x_oop[1]

    return x


def unstack_state(x):
    """Function that unstacks state vector into in-plane and out-of-plane vectors.

        Args:
            x (numpy.array): The complete state vector.

        Returns:
            x_ip (numpy.array): The in-plane state vector.
            x_oop (numpy.array): The out-of-plane state vector.

    """

    # sanity check(s)
    if len(x) != 6:
        print('UNSTACK_STATE: complete state vector must be 6-dimensional')

    # out-of-plane part
    x_oop = numpy.zeros(2)
    x_oop[0] = x[2]
    x_oop[1] = x[5]
    # in-plane part
    x_ip = numpy.zeros(4)
    x_ip[0:2] = x[0:2]
    x_ip[2:4] = x[3:5]

    return x_ip, x_oop


class BoundaryConditions:
    """Class to manage boundary conditions.

    """

    def __init__(self, nu0, nuf, x0, xf):
        """Constructor for the class BoundaryConditions.

                Args:
                    nu0 (float): initial true anomaly.
                    nuf (float): final true anomaly.
                    x0 (numpy.array): initial state vector.
                    xf (numpy.array): final state vector.

        """

        if len(x0) != len(xf):
            print('BoundaryConditions: miss-match between size of initial and final state vectors')

        self.nu0 = nu0
        self.nuf = nuf
        self.half_dim = int(len(x0) / 2)
        self.x0 = numpy.zeros(2 * self.half_dim)
        self.xf = numpy.zeros(2 * self.half_dim)
        for i in range(0, len(x0)):
            self.x0[i] = x0[i]
            self.xf[i] = xf[i]

    def write_to_file(self, file_path):
        """Function that writes the boundary conditions in a file.

            Args:
                file_path (str): The path to create/overwrite the boundary conditions.

        """

        file_object = open(file_path, 'w')
        file_object.write("Initial true anomaly \n")
        file_object.write(str(self.nu0) + '\n')
        file_object.write("Final true anomaly \n")
        file_object.write(str(self.nuf) + '\n')
        file_object.write("Initial state vector \n")
        for k in range(0, len(self.x0)):
            file_object.write(str(self.x0[k]) + " ")
        file_object.write('\n')
        file_object.write("Final state vector \n")
        for k in range(0, len(self.xf)):
            file_object.write(str(self.xf[k]) + " ")
        file_object.close()


class ControlLaw:
    """Class to manage control laws.

    """

    def __init__(self, half_dim, nus, DVs, lamb=None):
        """Constructor of class ControlLaw.

                Args:
                    half_dim (int): dimension of control vector.
                    nus (numpy.array): true anomalies where burns cccur.
                    DVs (numpy.array): Delta-Vs.
                    lamb (numpy.array): coefficients for primer vector.

        """

        self.N = len(nus)
        self.half_dim = half_dim
        self.nus = numpy.zeros(self.N)
        for i in range(0, self.N):
            self.nus[i] = nus[i]
        self.DVs = numpy.zeros((self.N, half_dim))
        for i in range(0, self.N):
            if self.half_dim == 1:
                self.DVs[i, :] = DVs[i]
            else:  # in-plane or complete dynamics
                self.DVs[i, :] = DVs[i, :]

        if lamb is not None:
            self.lamb = numpy.zeros(len(lamb))
            for i in range(0, len(lamb)):
                self.lamb[i] = lamb[i]
        else:  # no coefficients of primer vector were provided as inputs
            self.lamb = []

    def compute_cost(self, p):
        """Function returning the p-norm of a control law.

        """

        consumption = 0.
        for k in range(0, self.N):
            consumption += linalg.norm(self.DVs[k, :], p)
        return consumption

    def write_to_file(self, file_path):
        """Function that writes the control law in a file.

            Args:
                file_path (str): The path to create/overwrite the control law.

        """

        file_object = open(file_path, 'w')
        file_object.write("True anomalies of burn \n")
        for k in range(0, self.N):
            file_object.write(str(self.nus[k]) + " ")
        file_object.write('\n')
        for k in range(0, self.N):
            file_object.write("Delta-V #" + str(k+1) + "\n")
            for i in range(0, self.half_dim):
                file_object.write(str(self.DVs[k, i]) + " ")
            file_object.write('\n')
        file_object.close()


class NoControl(ControlLaw):
    """ Class for dummy control law.

    """

    def __init__(self, half_dim):
        """Constructor for class NoControl.

                Args:
                    half_dim (int): half-dimension of state vector.

        """
        DVs = numpy.zeros((1, half_dim))
        # call to parent constructor
        ControlLaw.__init__(self, half_dim, [0.], DVs)


def merge_control(CL_ip, CL_oop):
    """Function merging in-plane and out-of-plane control laws into a single one for the complete dynamics.

            Args:
                CL_ip (ControlLaw): in-plane control law.
                CL_oop (ControlLaw): out-of-plane control law.

            Returns:
                (ControlLaw): merged control law for complete dynamics.

    """

    # sanity check(s)
    if CL_ip.half_dim != 2:
        print('merge_control: in-plane control vector must have 2 components')
    if CL_oop.half_dim != 1:
        print('merge_control: out-of-plane control vector must have 1 component')

    # merge nus and corresponding impulses
    nus_unsorted = numpy.concatenate((CL_ip.nus, CL_oop.nus), axis=0)
    DV_unsorted = numpy.zeros((len(nus_unsorted), 3))
    for k in range(0, len(nus_unsorted)):
        if k < len(CL_ip.nus):
            DV_unsorted[k, 0:2] = CL_ip.DVs[k, 0:2]
        else:  # last loop
            DV_unsorted[k, 2] = CL_oop.DVs[k - len(CL_ip.nus)]

    # sort nus and corresponding impulses
    indices_sorting = numpy.argsort(nus_unsorted)
    nus_conc = sorted(nus_unsorted)
    DV_conc = numpy.zeros((len(nus_conc), 3))
    for k, index in enumerate(indices_sorting):
        DV_conc[k, :] = DV_unsorted[index, :]

    # remove duplicated nus and merge impulses accordingly
    nus = []
    indices_nodupli = []
    for k, date in enumerate(nus_conc):
        nus.append(date)
        indices_nodupli.append(k)
    removed = 0
    for k in range(0, len(nus_conc) - 1):
        if nus_conc[k] == nus_conc[k+1]:
            del nus[k - removed]
            del indices_nodupli[k - removed]
            DV_conc[k, :] += DV_conc[k+1, :]
            removed += 1
    DVs = numpy.zeros((len(nus), 3))
    for k in range(0, len(nus)):
        DVs[k, :] = DV_conc[indices_nodupli[k], :]

    if CL_ip.lamb != [] and CL_oop.lamb != []:
        lamb = stack_state(CL_ip.lamb, CL_oop.lamb)
    else:  # coefficients of sub-primer vectors are not all provided
        lamb = None

    return ControlLaw(3, nus, DVs, lamb)
