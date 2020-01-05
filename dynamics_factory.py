# dynamics_factory.py: set of classes implementing some dynamics of interest
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import body_prob_dyn
import orbital_mechanics
from config import conf


class RestriTwoBodyProbEarthFromSMA(body_prob_dyn.RestriTwoBodyProb):
    """Class implementing the dynamics of a spacecraft w.r.t. a reference, elliptical Earth orbit defined by
    its semi-major axis.

    """

    def __init__(self, ecc, sma):
        """Constructor for class RestriTwoBodyProbEarthFromSMA.

                  Args:
                      ecc (float): eccentricity.
                      sma (float): semi-major axis

        """

        # call to parent constructor
        body_prob_dyn.RestriTwoBodyProb.__init__(self, ecc,
                                           orbital_mechanics.sma_to_period(sma, conf.const_grav["Earth_constant"]), sma)

    @classmethod
    def circular(cls, sma):
        """Factory for circular version of class.

                  Args:
                      sma (float): semi-major axis

        """

        return cls(0., sma)


class RestriTwoBodyProbEarthFromPeriod(body_prob_dyn.RestriTwoBodyProb):
    """Class implementing the dynamics of a spacecraft w.r.t. a reference, elliptical Earth orbit defined by its period.

    """

    def __init__(self, ecc, period):
        """Constructor for class RestriTwoBodyProbEarthFromPeriod.

                  Args:
                      ecc (float): eccentricity.
                      period (float): orbital period

        """

        # call to parent constructor
        body_prob_dyn.RestriTwoBodyProb.__init__(self, ecc, period,
                                                 orbital_mechanics.period_to_sma(period,
                                                                                  conf.const_grav["Earth_constant"]))

    @classmethod
    def circular(cls, period):
        """Factory for circular version of class.

                  Args:
                      period (float): orbital period

        """

        return cls(0., period)

    @classmethod
    def geostationary(cls):
        """Factory for geostationary version of class.


        """

        return RestriTwoBodyProbEarthFromPeriod.circular(3600. * 24.)


class CircRestriThreeBodyProb(body_prob_dyn.RestriThreeBodyProb):
    """Class implementing the restricted three body problem with a circular reference orbit.

    """

    def __init__(self, mu, period, sma, Li):
        """Constructor for class CircRestriThreeBodyProb.

                  Args:
                        mu (float): ratio of minor mass over total mass.
                        period (float): orbital period.
                        sma (float): semi-major axis (must be consistent with period)
                        Li (int): index of Lagrange Point

        """

        body_prob_dyn.RestriThreeBodyProb.__init__(self, mu, 0., period, sma, Li)

    @classmethod
    def Earth_Moon(cls, Li):
        """Factory for the dynamics of a spacecraft w.r.t. a given Lagrange Point in the Earth-Moon system.

                  Args:
                      Li (int): index of Lagrange Point

        """

        sma = conf.const_dist["dist_Earth_Moon"]
        return cls(conf.const_mass["mu_EM"], orbital_mechanics.sma_to_period(sma, conf.const_grav["EM_constant"]), sma,
                   Li)

    @classmethod
    def Sun_Earth(cls, Li):
        """Factory for the dynamics of a spacecraft w.r.t. a given Lagrange Point in the Sun-Earth system (takes into
        account the Moon's mass).

                  Args:
                      Li (int): index of Lagrange Point

        """

        period = 3600.0 * 24.0 * 365.25
        return cls(conf.const_mass["mu_SE"], period,
                   orbital_mechanics.period_to_sma(period, conf.const_grav["Sun_constant"]), Li)
