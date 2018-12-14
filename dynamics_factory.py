# dynamics_factory.py: set of classes implementing some dynamics of interest
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import body_prob_dyn
import orbital_mechanics
from config import conf


class EllipticalRestricted2BodyProblemEarthFromSMA(body_prob_dyn.BodyProbDyn):
    """Class implementing the dynamics of a spacecraft w.r.t. a reference, elliptical Earth orbit defined by
    its semi-major axis.

    """

    def __init__(self, ecc, sma):
        """Constructor for class EllipticalRestricted2BodyProblemEarthFromSMA.

                  Args:
                      ecc (float): eccentricity.
                      sma (float): semi-major axis

        """

        # call to parent constructor
        body_prob_dyn.BodyProbDyn.__init__(self, 0., ecc,
                                              orbital_mechanics.sma_to_period(sma, conf.const_grav["Earth_constant"]),
                                              sma)


class EllipticalRestricted2BodyProblemEarthFromPeriod(body_prob_dyn.BodyProbDyn):
    """Class implementing the dynamics of a spacecraft w.r.t. a reference, elliptical Earth orbit defined by its period.

    """

    def __init__(self, ecc, period):
        """Constructor for class EllipticalRestricted2BodyProblemEarthFromPeriod.

                  Args:
                      ecc (float): eccentricity.
                      period (float): orbital period

        """

        # call to parent constructor
        body_prob_dyn.BodyProbDyn.__init__(self, 0., ecc, period,
                                              orbital_mechanics.period_to_sma(period,
                                                                                  conf.const_grav["Earth_constant"]))


class CircularRestricted2BodyProblemEarthFromSMA(EllipticalRestricted2BodyProblemEarthFromSMA):
    """Class implementing the dynamics of a spacecraft w.r.t. a reference, circular Earth orbit defined by
    its semi-major axis.

    """

    def __init__(self, sma):
        """Constructor for class CircularRestricted2BodyProblemEarthFromSMA.

                  Args:
                      sma (float): semi-major axis

        """

        # call to parent constructor
        EllipticalRestricted2BodyProblemEarthFromSMA.__init__(self, 0., sma)


class CircularRestricted2BodyProblemEarthFromPeriod(EllipticalRestricted2BodyProblemEarthFromPeriod):
    """Class implementing the dynamics of a spacecraft w.r.t. a reference, circular Earth orbit defined by its period.

    """

    def __init__(self, period):
        """Constructor for class CircularRestricted2BodyProblemEarthFromPeriod.

                  Args:
                      period (float): orbital period

        """

        # call to parent constructor
        EllipticalRestricted2BodyProblemEarthFromPeriod.__init__(self, 0., period)


class GEO(CircularRestricted2BodyProblemEarthFromPeriod):
    """Class implementing the dynamics of a spacecraft w.r.t. a reference Geostationary Earth Orbit.

    """

    def __init__(self):
        """Constructor for class GEO.

        """

        # call to parent constructor
        CircularRestricted2BodyProblemEarthFromPeriod.__init__(self, 3600. * 24.)


class EarthMoonLP(body_prob_dyn.BodyProbDyn):
    """Class implementing the dynamics of a spacecraft w.r.t. a given Lagrange Point in the Earth-Moon system.

    """

    def __init__(self, Li):
        """Constructor for class EarthMoonLP.

                  Args:
                      Li (int): index of Lagrange Point.

        """

        sma = conf.const_dist["dist_Earth_Moon"]
        # call to parent constructor
        body_prob_dyn.BodyProbDyn.__init__(self, conf.const_mass["mu_EM"], 0.,
                                              orbital_mechanics.sma_to_period(sma, conf.const_grav["EM_constant"]),
                                              sma, Li)


class SunEarthLP(body_prob_dyn.BodyProbDyn):
    """Class implementing the dynamics of a spacecraft w.r.t. a given Lagrange Point in the Sun-Earth system.

    """

    def __init__(self, Li):
        """Constructor for class SunEarthLP.

                  Args:
                      Li (int): index of Lagrange Point.

        """

        period = 3600.0 * 24.0 * 365.25
        # call to parent constructor
        body_prob_dyn.BodyProbDyn.__init__(self, conf.const_mass["mu_SE"], 0., period,
                                              orbital_mechanics.period_to_sma(period, conf.const_grav["Sun_constant"]),
                                              Li)
