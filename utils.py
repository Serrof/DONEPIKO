# utils.py: set of various functions and classes used by the toolbox
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import numpy as np
from numpy import linalg


def stack_state(x_ip, x_oop):
    """Function that stacks in-plane and out-of-plane state vectors.

        Args:
            x_ip (np.array): The in-plane state vector.
            x_oop (np.array): The out-of-plane state vector.

        Returns:
            x (np.array): The state vector for the corresponding complete dynamics.

    """

    # sanity check(s)
    if len(x_ip) != 4:
        raise ValueError('STACK_STATE: in-plane vector must be 4-dimensional')
    if len(x_oop) != 2:
        raise ValueError('STACK_STATE: out-of-plane vector must be 2-dimensional')

    x = np.zeros(6)
    x[0:2] = x_ip[0:2]
    x[2] = x_oop[0]
    x[3:5] = x_ip[2:4]
    x[5] = x_oop[1]

    return x


def unstack_state(x):
    """Function that unstacks state vector into in-plane and out-of-plane vectors.

        Args:
            x (np.array): The complete state vector.

        Returns:
            x_ip (np.array): The in-plane state vector.
            x_oop (np.array): The out-of-plane state vector.

    """

    # sanity check(s)
    if len(x) != 6:
        raise ValueError('UNSTACK_STATE: complete state vector must be 6-dimensional')

    # out-of-plane part
    x_oop = np.zeros(2)
    x_oop[0] = x[2]
    x_oop[1] = x[5]
    # in-plane part
    x_ip = np.zeros(4)
    x_ip[0:2] = x[0:2]
    x_ip[2:4] = x[3:5]

    return x_ip, x_oop


def vector_to_square_matrix(x_vector, n):
    """Function turning a N^2-dimensional vector into a N*N matrix column-wise.

        Args:
            x_vector (np.array): vector.
            n (int): size of output square matrix.

        Returns:
            x_matrix (np.array): matrix whose concatenated columns would make up the input vector.

    """
    x_matrix = np.zeros((n, n))
    for i in range(0, n):
        x_matrix[i, :] = x_vector[i * n: (i + 1) * n]

    return x_matrix


def square_matrix_to_vector(x_matrix):
    """Function turning a NxN matrix into a N^2-dimensional vector.

        Args:
            x_matrix (np.array): square matrix.

        Returns:
            x_vector (np.array): vector composed of the concatenated columns of the input matrix.

    """
    return np.array(x_matrix).flatten()


class BoundaryConditions:
    """Class to manage boundary conditions.

            Attributes:
                nu0 (float): initial true anomaly.
                nuf (float): final true anomaly.
                half_dim (int): half-dimension of state vector.
                x0 (np.array): initial state vector.
                xf (np.array): final state vector.

    """

    def __init__(self, nu0, nuf, x0, xf):
        """Constructor for the class BoundaryConditions.

                Args:
                    nu0 (float): initial true anomaly.
                    nuf (float): final true anomaly.
                    x0 (np.array): initial state vector.
                    xf (np.array): final state vector.

        """

        if len(x0) != len(xf):
            raise ValueError('BoundaryConditions: miss-match between size of initial and final state vectors')

        self.nu0 = nu0
        self.nuf = nuf
        self.half_dim = int(len(x0) / 2)
        self.x0 = np.array(x0)
        self.xf = np.array(xf)

    def copy(self):
        """Function returning a copy of the object.

                Returns:
                    (BoundaryConditions): copied object.

        """

        return BoundaryConditions(self.nu0, self.nuf, self.x0, self.xf)

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
        for el in self.x0:
            file_object.write(str(el) + " ")
        file_object.write('\n')
        file_object.write("Final state vector \n")
        for el in self.xf:
            file_object.write(str(el) + " ")
        file_object.close()


class ControlLaw:
    """Class to manage control laws.

            Attributes:
                N (int): number of impulses.
                half_dim (int): dimension of control vector.
                nus (np.array): true anomalies where burns occur.
                DVs (np.array): Delta-Vs.
                lamb (np.array): coefficients for primer vector.

    """

    def __init__(self, half_dim, nus, DVs, lamb=None):
        """Constructor of class ControlLaw.

                Args:
                    half_dim (int): dimension of control vector.
                    nus (np.array): true anomalies where burns occur.
                    DVs (np.array): Delta-Vs.
                    lamb (np.array): coefficients for primer vector.

        """

        self.N = len(nus)
        self.half_dim = half_dim
        self.nus = np.array(nus)
        self.DVs = np.zeros((self.N, half_dim))
        if self.half_dim == 1:
            for i, DV in enumerate(DVs):
                self.DVs[i, :] = DV
        else:  # in-plane or complete dynamics
            self.DVs += DVs

        if lamb is not None:
            self.lamb = np.array(lamb)
        else:  # no coefficients of primer vector were provided as inputs
            self.lamb = []

    def copy(self):
        """Function returning a copy of the object.

                Returns:
                    (ControlLaw): copied object.

        """

        return ControlLaw(self.half_dim, self.nus, self.DVs, self.lamb)

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
        for nu in self.nus:
            file_object.write(str(nu) + " ")
        file_object.write('\n')
        for k in range(0, self.N):
            file_object.write("Delta-V #" + str(k+1) + "\n")
            for i in range(0, self.half_dim):
                file_object.write(str(self.DVs[k, i]) + " ")
            file_object.write('\n')
        file_object.close()


class NoControl(ControlLaw):
    """ Class for dummy control law, meaning no actual non-zero impulse.

    """

    def __init__(self, BC):
        """Constructor for class NoControl. Value of independent variable corresponding to a single null impulse is
        arbitrarily set to the initial one.

                Args:
                    BC (BoundaryConditions): constraints for two-point boundary value problem.

        """
        # call to parent constructor
        ControlLaw.__init__(self, BC.half_dim, [BC.nu0], [[0.] * BC.half_dim])


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
        raise ValueError('merge_control: in-plane control vector must have 2 components')
    if CL_oop.half_dim != 1:
        raise ValueError('merge_control: out-of-plane control vector must have 1 component')

    # merge nus and corresponding impulses
    nus_unsorted = np.concatenate((CL_ip.nus, CL_oop.nus), axis=0)
    DV_unsorted = np.zeros((len(nus_unsorted), 3))
    for k in range(0, len(nus_unsorted)):
        if k < len(CL_ip.nus):
            DV_unsorted[k, :2] = CL_ip.DVs[k, :]
        else:  # last loop
            DV_unsorted[k, 2] = CL_oop.DVs[k - len(CL_ip.nus)]

    # sort nus and corresponding impulses
    indices_sorting = np.argsort(nus_unsorted)
    nus_conc = nus_unsorted[indices_sorting]
    DV_conc = DV_unsorted[indices_sorting, :]

    # remove duplicated nus and merge impulses accordingly
    nus = []
    DVs = []
    for k, nu in enumerate(nus_conc):
        if nu not in nus:
            nus.append(nu)
            DVs.append(DV_conc[k, :])
        else:
            DVs[-1] += DV_conc[k, :]

    if len(CL_ip.lamb) != 0 and len(CL_oop.lamb) != 0:
        lamb = stack_state(CL_ip.lamb, CL_oop.lamb)
    else:  # coefficients of sub-primer vectors are not all provided
        lamb = None

    return ControlLaw(3, nus, DVs, lamb)
