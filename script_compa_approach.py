# script_compa_approach.py: script comparing the results of the direct and indirect approaches 
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import utils
import random
import numpy as np
import math
import master
import plotter
import dynamics_factory
import time

nu0 = 2.0 * math.pi * random.random()  # initial true anomaly in rad
nuf = nu0 + 2.0 * math.pi * random.random()  # final true anomaly in rad

dyn = dynamics_factory.CircRestriThreeBodyProb.Sun_Earth(Li=4)  # 3-body restricted dynamics is
# linearized around Sun-Earth Lagrange Point 4

hd = 3  # half-dimension of state vector (1 for out-of-plane dynamics only, 2 for in-plane only and 3 for complete)
x0 = np.zeros(hd * 2)
xf = np.zeros(hd * 2)
for i in range(0, len(x0)):
	if i < hd:
		x0[i] = (-1.0 + 2.0 * random.random()) * 1.0e3
		xf[i] = (-1.0 + 2.0 * random.random()) * 1.0e3
	else:
		x0[i] = (-1.0 + 2.0 * random.random()) * 1.0e3 * math.pi / dyn.params.period
		xf[i] = (-1.0 + 2.0 * random.random()) * 1.0e3 * math.pi / dyn.params.period

p = 2  # norm for minimization (1 or 2)

BC = utils.BoundaryConditions(nu0, nuf, x0, xf)
plr = plotter.Plotter(dyn, BC, p, anomaly=True, linearized=True, analytical=True)
mast = master.Master(indirect=True, p=p, plr=plr)

t1 = time.time()
mast.solve()
print("Time elapsed with indirect approach: " + str(time.time() - t1) + " s")
mast.plot()
mast.write_boundary_cond("boundary_conditions_compa_approach.txt")
mast.write_control_law("control_law_indirect.txt")

mast.set_approach(indirect=False)
t2 = time.time()
mast.solve()
print("Time elapsed with direct approach: " + str(time.time() - t2) + " s")
mast.plot()
mast.write_control_law("control_law_direct.txt")

master.Master.show()
