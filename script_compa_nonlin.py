# script_compa_nonlin.py: script showing how the optimal solution with linear dynamics behaves with the original one
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import utils
import random
import numpy
import math
import master
import plotter
import matplotlib.pyplot as plt
import dynamics_factory

nu0 = 2.0 * math.pi * random.random()  # initial true anomaly in rad
nuf = nu0 + 2.0 * math.pi * random.random()  # final true anomaly in rad

dyn = dynamics_factory.EarthMoonLP(Li=3)  # 3-body restricted dynamics is linearized around Earth-Moon Lagrange Point 3

hd = 3  # half-dimension of state vector (1 for out-of-plane dynamics only, 2 for in-plane only and 3 for complete)
x0 = numpy.zeros(hd * 2)
xf = numpy.zeros(hd * 2)
for i in range(0, len(x0)):
    if i < hd:
        x0[i] = (-1.0 + 2.0 * random.random()) * 1.0e3
        xf[i] = (-1.0 + 2.0 * random.random()) * 1.0e3
    else:
        x0[i] = (-1.0 + 2.0 * random.random()) * 1.0e3 * math.pi / dyn.params.period
        xf[i] = (-1.0 + 2.0 * random.random()) * 1.0e3 * math.pi / dyn.params.period

p = 1  # norm for minimization (1 or 2)

BC = utils.BoundaryConditions(nu0, nuf, x0, xf)
plr = plotter.Plotter(dyn, BC, p, anomaly=False, linearized=False, analytical=True)
master = master.Master(indirect=False, p=p, plr=plr)

master.solve()

master.set_linearity_plot(False)
master.plot()
master.write_states("nonlin_states.txt")
plt.show()
