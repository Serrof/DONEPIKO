# py: set of tuning parameters for algorithms and other
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import default_conf


class Singleton(type):
    """Class implementing the singleton design pattern


    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config:
    """Class emulating the configuration (tuning parameters and physical constants).

            Attributes:
                params_indirect (dict): tuning parameters related to indirect optimal control.
                params_direct (dict): tuning parameters related to direct optimal control.
                params_plot (dict): tuning parameters related to plots.
                params_other (dict): other tuning parameters.
                const_dist (dict): physical constants related to distance.
                const_mass (dict): physical constants related to mass.
                const_grav (dict): physical constants related to gravity.

    """

    __metaclass__ = Singleton

    def __init__(self):
        """Constructor for class Conf.

        """

        # initialize configuration
        self.params_indirect = default_conf.params_indirect
        self.params_direct = default_conf.params_direct
        self.params_plot = default_conf.params_plot
        self.params_other = default_conf.params_other
        self.const_dist = default_conf.const_dist
        self.const_mass = default_conf.const_mass
        self.const_grav = default_conf.const_grav

        # save initial config
        self._params_indirect_init = self.params_indirect.copy()
        self._params_direct_init = self.params_direct.copy()
        self._params_plot_init = self.params_plot.copy()
        self._params_other_init = self.params_other.copy()
        self._const_dist_init = self.const_dist.copy()
        self._const_mass_init = self.const_mass.copy()
        self._const_grav_init = self.const_grav.copy()

        # generate dependent constants
        self.compute_complement()

    def compute_complement(self):
        """Function to generate values of constants that depend exclusively on others.

        """

        self.const_dist["radius_geo"] = self.const_dist["radius_Earth"] + self.const_dist["alt_geo"]

        self.const_mass["mass_EM"] = self.const_mass["mass_Earth"] + self.const_mass["mass_Moon"]
        self.const_mass["mu_EM"] = self.const_mass["mass_Moon"] / self.const_mass["mass_EM"]
        self.const_mass["mu_SE"] = self.const_mass["mass_EM"] / (self.const_mass["mass_Sun"] +
                                                                 self.const_mass["mass_EM"])
        self.const_grav["Earth_constant"] = self.const_grav["G"] * self.const_mass["mass_Earth"]
        self.const_grav["Moon_constant"] = self.const_grav["G"] * self.const_mass["mass_Moon"]
        self.const_grav["EM_constant"] = self.const_grav["G"] * self.const_mass["mass_EM"]
        self.const_grav["Sun_constant"] = self.const_grav["G"] * self.const_mass["mass_Sun"]

    def reset(self):
        """Function to reset all parameters and constants to initial values.

        """

        self.params_indirect = self._params_indirect_init
        self.params_direct = self._params_direct_init
        self.params_plot = self._params_plot_init
        self.params_other = self._params_other_init
        self.const_dist = self._const_dist_init
        self.const_mass = self._const_mass_init
        self.const_grav = self._const_grav_init

    def set_params_indirect(self, field, value):
        """Setter for parameter in indirect optimal control.

                Args:
                    field (string): name of field to be edited.
                    value (): occurrence to be set in input field.

        """

        self.params_indirect[field] = value

    def set_params_direct(self, field, value):
        """Setter for parameter in direct optimal control.

                Args:
                    field (string): name of field to be edited.
                    value (): occurrence to be set in input field.

        """

        self.params_direct[field] = value

    def set_params_plot(self, field, value):
        """Setter for parameter in plots.

                Args:
                    field (string): name of field to be edited.
                    value (): occurrence to be set in input field.

        """

        self.params_plot[field] = value

    def set_params_other(self, field, value):
        """Setter for miscellaneous parameter.

                Args:
                    field (string): name of field to be edited.
                    value (): occurrence to be set in input field.

        """

        self.params_other[field] = value

    def set_const_dist(self, field, value):
        """Setter for parameter in direct optimal control.

                Args:
                    field (string): name of field to be edited.
                    value (): occurrence to be set in input field.

        """

        self.const_dist[field] = value
        self.compute_complement()

    def set_const_mass(self, field, value):
        """Setter for parameter in plots.

                Args:
                    field (string): name of field to be edited.
                    value (): occurrence to be set in input field.

        """

        self.const_mass[field] = value
        self.compute_complement()

    def set_const_grav(self, field, value):
        """Setter for other parameters.

                Args:
                    field (string): name of field to be edited.
                    value (): occurrence to be set in input field.

        """

        self.const_grav[field] = value
        self.compute_complement()


# create instance of singleton
conf = Config()
