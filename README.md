# DONEPIKO 1.0.5
DONEPIKO stands for Delta-v Optimization Near Equilibrium Points In Keplerian Orbits.

It is a Python toolbox under the GPL 3 license designed to minimize the fuel consumption of trajectories in the Restricted 2- or 3- Body Problem,
when the dynamics has been linearized around equilibrium. In the R2BP, it can be any reference orbit, while in the R3BP, it is a Lagrange Point of a given binary system. 
The motion is always formulated in the frame rotating with the Keplerian orbit, with the equilibrium point as the origin. The non-linear motion is available only for visualization, not yet for control design.

Mathematically speaking, it solves two-point value boundary problems while optimizing the control law, that is minimizing fuel expenditure. 
The assumption on propulsion is that the exhaust velocity is constant, implying that the cost function can be written as the integral of the norm of the control vector. Moreover, the optimal solution can always be expressed as impulsive i.e. instantaneous jumps in velocity. 
The thrusters themselves can either be body-mounted in the 3 directions of the reference orbital frame or gimballed. This translates into the type of norm (1 or 2) whose integral is minimized by the control vector.

DONEPIKO offers two approaches to tackle this optimal control problem. 
The direct method is a relaxation, achieved by discretizing the time horizon. Formulations as linear or semi-definite programs (depending on the norm) is performed via the introduction of slack variables.
The indirect builds on Neustadt's approach to the problem, later revisited by Carter, that can be seen as the linear version of the primer vector theory introduced by Lawden. 
In short, it solves first a finite-dimension problem that is the primal to the original one to then derive the optimal thrust strategy. If technically possible, the solver will prefer the analytical solution.
More details can be found about the methods in the documents listed in the bibliography.

INSTALLATION:
You can run DONEPIKO on Linux-like OS. It requires Python 2 with Numpy, Matplotlib and Scipy. You also need CVXOPT.
The following command lines are enough to install the dependencies on Ubuntu 16 with admin rights:
>> sudo apt-get install software-properties-common
>> sudo apt-add-repository universe && sudo apt-get update
>> sudo apt-get install python-pip python-numpy python-matplotlib python-scipy
>> sudo CVOPT_BUILD_GLPK=1 pip install cvxopt

TUTORIAL:
DONEPIKO is a toolbox and as such a script should be written to make calls to it.
4 examples are provided with this version, designed to demonstrate its features and illustrate both theory and practise.
Physical constants used by default can be added or modified in default_conf.xml. For expert use, changes to tuning parameters can also be made in this file. 
Note that any value can also be overloaded using the appropriate setters of the conf object.

BIBLIOGRAPHY:
- SERRA, Romain, ARZELIER, Denis, BRÉHARD, Florent, et al. Fuel-optimal impulsive fixed-time trajectories in the linearized circular restricted 3-body-problem. In : IAF Astrodynamics Symposium in 69TH international astronautical congress. 2018. https://hal.archives-ouvertes.fr/hal-01830253
- ARZELIER, Denis, BRÉHARD, Florent, DEAK, Norbert, et al. Linearized impulsive fixed-time fuel-optimal space rendezvous: A new numerical approach. IFAC-PapersOnLine, 2016, vol. 49, no 17, p. 373-378. https://hal.archives-ouvertes.fr/hal-01275427/
- SERRA, Romain, ARZELIER, Denis, and RONDEPIERRE, Aude. Analytical solutions for impulsive elliptic out-of-plane rendezvous problem via primer vector theory. IEEE Transactions on Control Systems Technology, 2018, vol. 26, no 1, p. 207-221
- LOUEMBET, Christophe, ARZELIER, Denis, and DEACONU, Georgia. Robust rendezvous planning under maneuver execution errors. Journal of Guidance, Control, and Dynamics, 2014, vol. 38, no 1, p. 76-93.
- CARTER, T. E. and BRIENT, J. Linearized impulsive rendezvous problem. Journal of Optimization Theory and Applications, 1995, vol. 86, no 3, p. 553-584.
- NEUSTADT, Lucien W. Optimization, a moment problem, and nonlinear programming. Journal of the Society for Industrial and Applied Mathematics, Series A: Control, 1964, vol. 2, no 1, p. 33-53.
- LAWDEN, Derek F. Optimal trajectories for space navigation. Butterworths, 1963.
