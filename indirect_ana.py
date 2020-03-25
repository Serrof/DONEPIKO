# indirect_ana.py: routines to analytically solve for the out-of-plane optimal trajectory
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

import math
import numpy as np
from numpy import linalg


def solver_ana(u, e, n, nu_0, nu_f):
	"""Function that computes elapsed time given final and initial true anomalies.

				Args:
					u (list): right-hand side of moment equation.
					e (float): eccentricity.
					n (float) :  mean motion.
					nu_0 (float) : initial true anomaly.
					nu_f (float) : final true anomaly.

				Returns:
					_nus (list): optimal true anomalies of burn.
					DVs (list): optimal Delta-Vs.
					lamb (np.array): coefficients of primer vector.

	"""

	# sanity check(s)
	if(len(u) != 2):
		return ValueError('SOLVER_ANA: input vector needs to be two-dimensional')
	if(e >= 1.0) or (e < 0.0):
		return ValueError('SOLVER_ANA: eccentricity must be larger or equal to 0 and strictly less than 1')
	if(n < 0.0):
		return ValueError('SOLVER_ANA: mean motion cannot be smaller than 0')
	if(nu_f <= nu_0):
		return ValueError('SOLVER_ANA: initial true anomaly cannot be larger than final one')

	lamb = np.array([0.0, 0.0])  # vector to store Lagrange coefficients
	DVs = []  # vector to store DVs magnitude
	nus = []  # vector to store impulses location

	# pre-computations
	arcu = math.atan2(u[1], u[0])
	magnu = linalg.norm(u, 2)

	if e == 0.0:  # circular case
		y = nu_0 + ((arcu + math.pi / 2.0 - nu_0) % math.pi) 
		if (y > nu_f):
			eps = 1.0
			if math.cos(nu_0) * u[0] + math.sin(nu_0) * u[1] < 0.0:
				eps = -1.0
			lamb[0] = -eps * (math.cos(nu_f) + math.cos(nu_0)) / math.sin(nu_f - nu_0)
			lamb[1] = -eps * (math.sin(nu_f) + math.sin(nu_0)) / math.sin(nu_f - nu_0)
			nus.extend([nu_0, nu_f])
			DVs.append(math.cos(nu_f - arcu) * magnu / math.sin(nu_f - nu_0))
			DVs.append(-math.cos(nu_0 - arcu) * magnu / math.sin(nu_f - nu_0))
		else:
			eps = 1.0
			if math.cos(y) * u[1] < 0.0:
				eps = -1.0		
			m = 0.0
			while y + m * math.pi <= nu_f:
				nus.append(y + m * math.pi)
				m += 1.0
			aux = eps * magnu / m
			for k in range(0, len(nus)):
				if k % 2 == 0:
					DVs.append(aux)
				else:
					DVs.append(-aux)
			
			inter = -1.0 / magnu
			lamb[0] = inter * u[0]
			lamb[1] = inter * u[1]

	else:  # elliptical case
	
		if nu_f - nu_0 >= 2.0 * math.pi:
			
			if (e * math.fabs(u[0]) - math.sqrt(1.0 - e * e) * math.fabs(u[1]) >= 0.0):
				x = math.acos(-e)
				x1 = nu_0 + ((x - nu_0) % (2.0 * math.pi))
				x2 = nu_0 + ((2.0 * math.pi - x - nu_0) % (2.0*  math.pi))
				if x1 > x2:
					x = x1
					x1 = x2
					x2 = x
				
				if math.sin(x1) >= 0.0:
					y_plus = x1
					y_minus = x2
				else:
					y_plus = x2
					y_minus = x1

				Nb_plus = 1.0
				nus.append(0.0)
				DVs.append(0.0)
				while y_plus + 2.0 * math.pi * Nb_plus <= nu_f:
					Nb_plus += 1.0
					nus.append(0.0)
					DVs.append(0.0)

				Nb_minus = 1.0
				nus.append(0.0)
				DVs.append(0.0)
				while y_minus + 2.0 * math.pi * Nb_minus <= nu_f:
					Nb_minus += 1.0
					nus.append(0.0)
					DVs.append(0.0)
			
				k_minus = 1.0
				k_plus = 1.0
				if math.sin(x1) < 0.0:
					for k in range(0, len(nus)):
						if k % 2 == 0:
							nus[k] = x1 + 2.0 * math.pi * (k_minus - 1.0)
							DVs[k] = (math.sqrt(1.0 - e * e) / (2.0 * e)) * (e * u[0] - math.sqrt(1.0 - e * e) * u[1]) / Nb_minus
							k_minus += 1.0
						else:
							nus[k] = x2 + 2.0 * math.pi * (k_plus - 1.0)
							DVs[k] = -(math.sqrt(1.0 - e * e) / (2.0 * e)) * (e * u[0] + math.sqrt(1.0 - e * e) * u[1]) / Nb_plus
							k_plus += 1.0
				else:
					for k in range(0, len(nus)):
						if k % 2 == 0:
							nus[k] = x1 + 2.0 * math.pi * (k_plus - 1.0)
							DVs[k] = -(math.sqrt(1.0 - e * e) / (2.0 * e))*(e * u[0] + math.sqrt(1.0 - e * e) * u[1]) / Nb_plus
							k_plus += 1.0	
						else:
							nus[k] = x2 + 2.0 * math.pi * (k_minus - 1.0)
							DVs[k] = (math.sqrt(1.0 - e * e) / (2.0 * e)) * (e * u[0] - math.sqrt(1.0 - e * e) * u[1]) / Nb_minus
							k_minus += 1.0								

				lamb[0] = -np.sign(u[0]) * math.sqrt(1.0 - e * e)
			
			else:
				y = math.acos(-math.fabs(u[1]) / magnu)
				if u[0] * u[1] < 0.0:
					y = 2.0 * math.pi - y
			
				y = nu_0 + ((y - nu_0) % (2.0 * math.pi))
			
				Nb = 1.0
				nus.append(0.0)
				DVs.append(0.0)
				while y + 2.0 * math.pi * Nb <= nu_f:
					Nb += 1.0
					nus.append(0.0)
					DVs.append(0.0)
			
				for k in range(0, len(nus)):
					nus[k] = y + 2.0 * math.pi * k
					DVs[k] = -np.sign(u[1]) * (magnu - e * math.fabs(u[1])) / Nb
			
				lamb[0] = -u[0] / magnu
				lamb[1] = np.sign(u[1]) * (e - math.fabs(u[1]) / magnu)

		elif (nu_f-nu_0<math.pi):
		
			if (math.sin(nu_0)>=math.sqrt(1.0-e*e)) and (math.sin(nu_f)<=-math.sqrt(1.0-e*e)) and (e*math.fabs(u[0])-math.sqrt(1.0-e*e)*math.fabs(u[1])>=0.0):
				x=math.acos(-e)
				x1=nu_0+((x-nu_0)%(2.0*math.pi))
				x2=nu_0+((2.0*math.pi-x-nu_0)%(2.0*math.pi))
				if x1>x2:
					x1, x2 = x2, x1
				nus.extend([x1, x2])
				DV_p=-(math.sqrt(1.0-e*e)/(2.0*e))*(e*u[0]+math.sqrt(1.0-e*e)*u[1])
				DV_m=(math.sqrt(1.0-e*e)/(2.0*e))*(e*u[0]-math.sqrt(1.0-e*e)*u[1])
				if math.sin(x1)<0.0:
					DVs.extend([DV_m, DV_p])
				else:
					DVs.extend([DV_p, DV_m])
				lamb[0]=-np.sign(u[0])*math.sqrt(1.0-e*e)

			if (np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])==-np.sign(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])):
				eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
				y=math.acos(-eps*(u[1]/magnu))
				if (u[0]*eps<0.0):
					y=2.0*math.pi-y
				y=nu_0+((y-nu_0)%(2.0*math.pi))
				if (e-eps*(u[1]/magnu)<=0.0) or ((magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_0) + eps*u[0]*math.sin(nu_0)>0.0) and (eps*u[0]*(e+math.cos(nu_0))+math.sin(nu_0)*(-e*magnu+eps*u[1])>0.0) and (magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_f) +eps*u[0]*math.sin(nu_f)>0.0) and (eps*u[0]*(e+math.cos(nu_f))+math.sin(nu_f)*(-e*magnu+eps*u[1])<0.0)):
					nus.append(y)
					DVs.append(-eps*magnu+e*u[1])
					lamb[0]=-u[0]/magnu
					lamb[1]=eps*e-u[1]/magnu

			if (1.0+2.0*e*math.cos(nu_0)+math.cos(nu_f-nu_0)<=0.0) and (math.sin(nu_0)<=math.sqrt(1.0-e*e)):
				y=nu_0+math.acos(-(1.0+2.0*e*math.cos(nu_0)))
				eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
				if (magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_0) + eps*u[0]*math.sin(nu_0)<=0.0):
					DVs.append(((1.0+e*math.cos(nu_0))/math.sin(y-nu_0))*(math.cos(y)*u[0]+u[1]*math.sin(y)))
					DVs.append(((1.0+e*math.cos(y))/math.sin(y-nu_0))*(-math.cos(nu_0)*u[0]-u[1]*math.sin(nu_0)))
					nus.extend([nu_0, y])
					lamb[0]=eps*(math.sin(nu_0)*(1.0+2.0*e*math.cos(nu_0))-2.0*math.cos(nu_0)*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))
					lamb[1]=eps*(e-math.cos(nu_0)*(1.0+2.0*e*math.cos(nu_0))-2.0*math.sin(nu_0)*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))

			if (1.0+2.0*e*math.cos(nu_f)+math.cos(nu_f-nu_0)<=0.0) and (math.sin(nu_f)>=-math.sqrt(1.0-e*e)):
				y=nu_f-math.acos(-(1.0+2.0*e*math.cos(nu_f)))
				eps=np.sign(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])
				if (magnu+(2.0*e*magnu+eps*u[1])*math.cos(nu_f) - eps*u[0]*math.sin(nu_f)<=0.0):
					DVs.append(((1.0+e*math.cos(y))/math.sin(nu_f-y))*(math.cos(nu_f)*u[0]+u[1]*math.sin(nu_f)))
					DVs.append(((1.0+e*math.cos(nu_f))/math.sin(nu_f-y))*(-math.cos(y)*u[0]-u[1]*math.sin(y)))
					nus.extend([y, nu_f])
					lamb[0]=-eps*(math.sin(nu_f)*(1.0+2.0*e*math.cos(nu_f))+2.0*math.cos(nu_f)*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))
					lamb[1]=-eps*(e-math.cos(nu_f)*(1.0+2.0*e*math.cos(nu_f))+2.0*math.sin(nu_f)*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))

			if (np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])==np.sign(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])) and (1.0+2.0*e*math.cos(nu_0)+math.cos(nu_f-nu_0)>=0.0) and (1.0+2.0*e*math.cos(nu_f)+math.cos(nu_f-nu_0)>=0.0):
				nus.extend([nu_0, nu_f])
				DVs.append(((1.0+e*math.cos(nu_0))/math.sin(nu_f-nu_0))*(math.cos(nu_f)*u[0]+u[1]*math.sin(nu_f)))
				DVs.append(((1.0+e*math.cos(nu_f))/math.sin(nu_f-nu_0))*(-math.cos(nu_0)*u[0]-u[1]*math.sin(nu_0)))
				eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
				lamb[0]=-eps*((math.cos(nu_f)+math.cos(nu_0)+2.0*e*math.cos(nu_f)*math.cos(nu_0))/math.sin(nu_f-nu_0))
				lamb[1]=-eps*((math.sin(nu_f)+math.sin(nu_0)+e*math.sin(nu_f+nu_0))/math.sin(nu_f-nu_0))
		
		elif (math.pi<nu_f-nu_0) and (nu_f-nu_0<2*math.pi):
		
			if (e*math.fabs(u[0])-math.sqrt(1.0-e*e)*math.fabs(u[1])>=0.0):
				if (math.sin(nu_0)>=math.sqrt(1.0-e*e)) or ((math.sin(nu_0)<=-math.sqrt(1.0-e*e)) and (math.sin(nu_f)<=-math.sqrt(1.0-e*e))) or (math.fabs(math.sin(nu_0))<math.sqrt(1.0-e*e) and ((e+math.cos(nu_0))*(e+math.cos(nu_f))>0.0)):
					x=math.acos(-e)
					x1=nu_0+((x-nu_0)%(2.0*math.pi))
					x2=nu_0+((2.0*math.pi-x-nu_0)%(2.0*math.pi))
					if (x1>x2):
						x1, x2 = x2, x1
				
					nus.extend([x1, x2])
					DV_p=-(math.sqrt(1.0-e*e)/(2.0*e))*(e*u[0]+math.sqrt(1.0-e*e)*u[1])
					DV_m=(math.sqrt(1.0-e*e)/(2.0*e))*(e*u[0]-math.sqrt(1.0-e*e)*u[1])
					if (math.sin(x1)<0.0):
						DVs.extend([DV_m, DV_p])
					else:
						DVs.extend([DV_p, DV_m])
					lamb[0]=-np.sign(u[0])*math.sqrt(1.0-e*e)

			if (math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])*(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])<0.0:
				eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
				y=math.acos(-eps*(u[1]/magnu))
				if (u[0]*eps)<0.0:
					y=2.0*math.pi-y
				y=nu_0+((y-nu_0)%(2.0*math.pi))
				if (e-eps*(u[1]/magnu)<=0.0) or ((magnu+(2*e*magnu-eps*u[1])*math.cos(nu_0) + eps*u[0]*math.sin(nu_0)>0.0) and (eps*u[0]*(e+math.cos(nu_0))+math.sin(nu_0)*(-e*magnu+eps*u[1])>0.0) and (magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_f) +eps*u[0]*math.sin(nu_f)>0.0) and (eps*u[0]*(e+math.cos(nu_f))+math.sin(nu_f)*(-e*magnu+eps*u[1])<0.0)):
					nus.append(y)
					DVs.append(-eps*magnu+e*u[1])
					lamb[0]=-u[0]/magnu
					lamb[1]=eps*e-u[1]/magnu
			
			else:
				eps=np.sign(u[1])
				y=math.acos(-math.fabs(u[1])/magnu)
				if (u[0]*u[1]<0):
					y=2.0*math.pi-y
				y=nu_0+((y-nu_0)%(2.0*math.pi))        
				if (e-math.fabs(u[1])/magnu<=0.0) or ((magnu+(2.0*e*magnu-math.fabs(u[1]))*math.cos(nu_0) + eps*u[0]*math.sin(nu_0)>0.0) and (eps*u[0]*(e+math.cos(nu_0))+math.sin(nu_0)*(-e*magnu+math.fabs(u[1]))>0.0) and (magnu+(2.0*e*magnu-math.fabs(u[1]))*math.cos(nu_f) +eps*u[0]*math.sin(nu_f)>0.0) and (eps*u[0]*(e+math.cos(nu_f))+math.sin(nu_f)*(-e*magnu+math.fabs(u[1]))<0.0)):
					nus.append(y)
					DVs.append(-eps*magnu+e*u[1])
					lamb[0]=-u[0]/magnu
					lamb[1]=eps*e-u[1]/magnu

			if (math.cos(nu_0)<0.0):
				eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
				if (math.sin(nu_0)<=math.sqrt(1.0-e*e)) and ((1.0+e*math.cos(nu_0))*(math.sin(nu_f-nu_0)-e*math.sin(nu_0))+(math.cos(nu_f-nu_0)+e*math.cos(nu_0))*(2.0*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0)))-e*math.sin(nu_0))<=0.0) and (1.0-math.cos(nu_f-nu_0)-2.0*math.sin(nu_f-nu_0)*(e*math.sin(nu_0)-math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))>=0.0) and (magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_0) + eps*u[0]*math.sin(nu_0)<=0.0):
					y=nu_0+math.acos(-(1.0+2.0*e*math.cos(nu_0)))
					DVs.append(((1.0+e*math.cos(nu_0))/math.sin(y-nu_0))*(math.cos(y)*u[0]+u[1]*math.sin(y)))
					DVs.append(((1.0+e*math.cos(y))/math.sin(y-nu_0))*(-math.cos(nu_0)*u[0]-u[1]*math.sin(nu_0)))
					nus.extend([nu_0, y])
					lamb[0]=eps*(math.sin(nu_0)*(1.0+2.0*e*math.cos(nu_0))-2.0*math.cos(nu_0)*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))
					lamb[1]=eps*(e-math.cos(nu_0)*(1.0+2.0*e*math.cos(nu_0))-2.0*math.sin(nu_0)*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))
				elif (1.0+2.0*e*math.cos(nu_0)+math.cos(nu_f-nu_0)>=0.0) and (math.sin(nu_0)<=-math.sqrt(1.0-e*e)) and ((1.0+e*math.cos(nu_0))*(math.sin(nu_f-nu_0)-e*math.sin(nu_0))-(math.cos(nu_f-nu_0)+e*math.cos(nu_0))*(2.0*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0)))+e*math.sin(nu_0))<=0.0) and (1.0-math.cos(nu_f-nu_0)-2.0*math.sin(nu_f-nu_0)*(e*math.sin(nu_0)+math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))>=0.0) and (magnu+(2.0*e*magnu+eps*u[1])*math.cos(nu_0) - eps*u[0]*math.sin(nu_0)<=0.0):
					y=nu_0+2.0*math.pi-math.acos(-(1.0+2.0*e*math.cos(nu_0)))
					DVs.append(((1.0+e*math.cos(nu_0))/math.sin(y-nu_0))*(math.cos(y)*u[0]+u[1]*math.sin(y)))
					DVs.append(((1.0+e*math.cos(y))/math.sin(y-nu_0))*(-math.cos(nu_0)*u[0]-u[1]*math.sin(nu_0)))
					nus.extend([nu_0, y])
					lamb[0]=-eps*(math.sin(nu_0)*(1.0+2.0*e*math.cos(nu_0))+2.0*math.cos(nu_0)*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))
					lamb[1]=-eps*(e-math.cos(nu_0)*(1.0+2.0*e*math.cos(nu_0))+2.0*math.sin(nu_0)*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))

			if (math.cos(nu_f)<0.0):
				eps=np.sign(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])
				if (math.sin(nu_f)>=-math.sqrt(1.0-e*e)) and ((1+e*math.cos(nu_f))*(math.sin(nu_f-nu_0)+e*math.sin(nu_f))+(math.cos(nu_f-nu_0)+e*math.cos(nu_f))*(2.0*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f)))+e*math.sin(nu_f))<=0.0) and (1.0-math.cos(nu_f-nu_0)+2.0*math.sin(nu_f-nu_0)*(e*math.sin(nu_f)+math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))>=0.0) and (magnu+(2.0*e*magnu+eps*u[1])*math.cos(nu_f) - eps*u[0]*math.sin(nu_f)<=0.0):
					y=nu_f-math.acos(-(1.0+2.0*e*math.cos(nu_f)))
					DVs.append(((1.0+e*math.cos(y))/math.sin(nu_f-y))*(math.cos(nu_f)*u[0]+u[1]*math.sin(nu_f)))
					DVs.append(((1.0+e*math.cos(nu_f))/math.sin(nu_f-y))*(-math.cos(y)*u[0]-u[1]*math.sin(y)))
					nus.extend([y, nu_f])
					lamb[0]=-eps*(math.sin(nu_f)*(1.0+2.0*e*math.cos(nu_f))+2.0*math.cos(nu_f)*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))
					lamb[1]=-eps*(e-math.cos(nu_f)*(1.0+2.0*e*math.cos(nu_f))+2.0*math.sin(nu_f)*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))
				elif (1.0+2.0*e*math.cos(nu_f)+math.cos(nu_f-nu_0)>=0.0) and (math.sin(nu_f)>=math.sqrt(1.0-e*e)) and ((1.0+e*math.cos(nu_f))*(math.sin(nu_f-nu_0)+e*math.sin(nu_f))-(math.cos(nu_f-nu_0)+e*math.cos(nu_f))*(2*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f)))-e*math.sin(nu_f))<=0.0) and (1.0-math.cos(nu_f-nu_0)+2.0*math.sin(nu_f-nu_0)*(e*math.sin(nu_f)-math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))>=0.0) and (magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_f) + eps*u[0]*math.sin(nu_f)<=0.0):
					y=nu_f-2.0*math.pi+math.acos(-(1.0+2.0*e*math.cos(nu_f)))
					DVs.append(((1.0+e*math.cos(y))/math.sin(nu_f-y))*(math.cos(nu_f)*u[0]+u[1]*math.sin(nu_f)))
					DVs.append(((1.0+e*math.cos(nu_f))/math.sin(nu_f-y))*(-math.cos(y)*u[0]-u[1]*math.sin(y)))
					nus.extend([y, nu_f])
					lamb[0]=eps*(math.sin(nu_f)*(1.0+2.0*e*math.cos(nu_f))-2.0*math.cos(nu_f)*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))
					lamb[1]=eps*(e-math.cos(nu_f)*(1.0+2.0*e*math.cos(nu_f))-2.0*math.sin(nu_f)*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))

			if (((math.sin(nu_0)<=-math.sqrt(1.0-e*e)) and (math.sin(nu_f)>=-math.sqrt(1.0-e*e))) or (math.fabs(math.sin(nu_0))<math.sqrt(1.0-e*e) and ((e+math.cos(nu_0))*(e+math.cos(nu_f))<=0.0))) and (np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])==np.sign(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])) and (1.0+2.0*e*math.cos(nu_0)+math.cos(nu_f-nu_0)<=0.0) and (1.0+2.0*e*math.cos(nu_f)+math.cos(nu_f-nu_0)<=0.0):
					nus.extend([nu_0, nu_f])
					DVs.append(((1.0+e*math.cos(nu_0))/math.sin(nu_f-nu_0))*(math.cos(nu_f)*u[0]+u[1]*math.sin(nu_f)))
					DVs.append(((1.0+e*math.cos(nu_f))/math.sin(nu_f-nu_0))*(-math.cos(nu_0)*u[0]-u[1]*math.sin(nu_0)))
					eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
					lamb[0]=eps*((math.cos(nu_f)+math.cos(nu_0)+2.0*e*math.cos(nu_f)*math.cos(nu_0))/math.sin(nu_f-nu_0))
					lamb[1]=eps*((math.sin(nu_f)+math.sin(nu_0)+e*math.sin(nu_f+nu_0))/math.sin(nu_f-nu_0))

			if (np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])==-np.sign(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])):
				a1=((math.cos(nu_f)-math.cos(nu_0))/math.sin(nu_f-nu_0))
				a2=((math.sin(nu_f)-math.sin(nu_0))/math.sin(nu_f-nu_0)+e)
				q=a1/a2
				if (math.fabs(a2)*(1.0+q*q)-e-math.sqrt(1.0+(q*q)*(1.0-e*e))<0.0):
					nus.extend([nu_0, nu_f])
					DVs.append(((1.0+e*math.cos(nu_0))/math.sin(nu_f-nu_0))*(math.cos(nu_f)*u[0]+u[1]*math.sin(nu_f)))
					DVs.append(((1.0+e*math.cos(nu_f))/math.sin(nu_f-nu_0))*(-math.cos(nu_0)*u[0]-u[1]*math.sin(nu_0)))
					eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
					lamb[0]=-eps*((math.cos(nu_f)-math.cos(nu_0))/math.sin(nu_f-nu_0))
					lamb[1]=-eps*((math.sin(nu_f)-math.sin(nu_0))/math.sin(nu_f-nu_0)+e)

		else: # nu_f-nu_0==math.pi
		
			if (math.sin(nu_0)>=math.sqrt(1.0-e*e)) and (math.sin(nu_f)<=-math.sqrt(1.0-e*e)) and (e*math.fabs(u[0])-math.sqrt(1.0-e*e)*math.fabs(u[1])>=0.0):
				x=math.acos(-e)
				x1=nu_0+((x-nu_0)%(2.0*math.pi))
				x2=nu_0+((2.0*math.pi-x-nu_0)%(2.0*math.pi))
				if x1>x2:
					x1, x2 = x2, x1
				nus.extend([x1, x2])
				DV_p=-(math.sqrt(1.0-e*e)/(2.0*e))*(e*u[0]+math.sqrt(1.0-e*e)*u[1])
				DV_m=(math.sqrt(1.0-e*e)/(2.0*e))*(e*u[0]-math.sqrt(1.0-e*e)*u[1])
				if math.sin(x1)<0.0:
					DVs.extend([DV_m, DV_p])
				else:
					DVs.extend([DV_p, DV_m])
				lamb[0]=-np.sign(u[0])*math.sqrt(1.0-e*e)

			if (np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])==-np.sign(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])):
				eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
				y=math.acos(-eps*(u[1]/magnu))
				if (u[0]*eps<0.0):
					y=2.0*math.pi-y
				y=nu_0+((y-nu_0)%(2.0*math.pi))
				if (e-eps*(u[1]/magnu)<=0.0) or ((magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_0) + eps*u[0]*math.sin(nu_0)>0.0) and (eps*u[0]*(e+math.cos(nu_0))+math.sin(nu_0)*(-e*magnu+eps*u[1])>0.0) and (magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_f) +eps*u[0]*math.sin(nu_f)>0.0) and (eps*u[0]*(e+math.cos(nu_f))+math.sin(nu_f)*(-e*magnu+eps*u[1])<0.0)):
					nus.append(y)
					DVs.append(-eps*magnu+e*u[1])
					lamb[0]=-u[0]/magnu
					lamb[1]=eps*e-u[1]/magnu

			if (1.0+2.0*e*math.cos(nu_0)+math.cos(nu_f-nu_0)<=0.0) and (math.sin(nu_0)<=math.sqrt(1.0-e*e)):
				y=nu_0+math.acos(-(1.0+2.0*e*math.cos(nu_0)))
				eps=np.sign(math.cos(nu_0)*u[0]+math.sin(nu_0)*u[1])
				if (magnu+(2.0*e*magnu-eps*u[1])*math.cos(nu_0) + eps*u[0]*math.sin(nu_0)<=0.0):
					DVs.append(((1.0+e*math.cos(nu_0))/math.sin(y-nu_0))*(math.cos(y)*u[0]+u[1]*math.sin(y)))
					DVs.append(((1.0+e*math.cos(y))/math.sin(y-nu_0))*(-math.cos(nu_0)*u[0]-u[1]*math.sin(nu_0)))
					nus.extend([nu_0, y])
					lamb[0]=eps*(math.sin(nu_0)*(1.0+2.0*e*math.cos(nu_0))-2.0*math.cos(nu_0)*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))
					lamb[1]=eps*(e-math.cos(nu_0)*(1.0+2.0*e*math.cos(nu_0))-2.0*math.sin(nu_0)*math.sqrt(-e*math.cos(nu_0)*(1.0+e*math.cos(nu_0))))

			if (1.0+2.0*e*math.cos(nu_f)+math.cos(nu_f-nu_0)<=0.0) and (math.sin(nu_f)>=-math.sqrt(1.0-e*e)):
				y=nu_f-math.acos(-(1.0+2.0*e*math.cos(nu_f)))
				eps=np.sign(math.cos(nu_f)*u[0]+math.sin(nu_f)*u[1])
				if (magnu+(2.0*e*magnu+eps*u[1])*math.cos(nu_f) - eps*u[0]*math.sin(nu_f)<=0.0):
					DVs.append(((1.0+e*math.cos(y))/math.sin(nu_f-y))*(math.cos(nu_f)*u[0]+u[1]*math.sin(nu_f)))
					DVs.append(((1.0+e*math.cos(nu_f))/math.sin(nu_f-y))*(-math.cos(y)*u[0]-u[1]*math.sin(y)))
					nus.extend([y, nu_f])
					lamb[0]=-eps*(math.sin(nu_f)*(1.0+2.0*e*math.cos(nu_f))+2.0*math.cos(nu_f)*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))
					lamb[1]=-eps*(e-math.cos(nu_f)*(1.0+2.0*e*math.cos(nu_f))+2.0*math.sin(nu_f)*math.sqrt(-e*math.cos(nu_f)*(1.0+e*math.cos(nu_f))))

	return nus, DVs, -lamb  # switch from Carter's sign convention to Lawden's
