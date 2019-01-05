# integrators.py: range of classes implementing integrators
# Copyright(C) 2019 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

from abc import ABCMeta, abstractmethod


class Integrator:
    """Abstract class for the implementation of numerical integrators with a fixed step-size.

        Attributes:
            _func (method): function to be integrated.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func):
        """Constructor for class Integrator.

                Args:
                     func (function): function to be integrated.

        """

        self._func = func

    @abstractmethod
    def integration_step(self, t, x, h):
        """Abstract method to be overwritten in classes inheriting from abstract class.

                Args:
                    t (float): current value of independent variable.
                    x (list): state vector at t.
                    h (float): step-size.

        """

        pass

    def integrate(self, t0, tf, x0, n_step):
        """Function that performs integration between two values of independent variable.

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (): state vector at t0.
                    n_step (int): number of integration steps to be performed.

                Returns:
                    Xs (list): history of state vectors at integration steps.
                    Ts (list): values taken by the independent variable at successive integration steps.

        """

        h = (tf - t0) / float(n_step)
        Ts = []
        Xs = []
        Ts.append(t0)
        x0_copy = []
        for i in range(0, len(x0)):
            x0_copy.append(x0[i])
        Xs.append(x0_copy)
        for k in range(0, n_step):
            Xs.append(self.integration_step(Ts[-1], Xs[-1], h))
            Ts.append(Ts[-1] + h)

        return Xs, Ts


class Euler(Integrator):
    """Class implementing the classic Euler integration scheme.

    """

    def integration_step(self, t, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t+h where h is the step-size.

                Args:
                    t (float): current value of independent variable.
                    x (list): state vector at t.
                    h (float): step-size.

                Returns:
                    xf (list): state vector at t+h.

        """

        f = self._func(t, x)  # function call
        xf = []
        for k in range(0, len(x)):
            xf.append(x[k] + h * f[k])

        return xf


class Heun(Integrator):
    """Class implementing the Heun integration scheme.

    """

    def integration_step(self, t, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t+h where h is the step-size.

                Args:
                    t (float): current value of independent variable.
                    x (list): state vector at t.
                    h (float): step-size.

                Returns:
                    xf (list): state vector at t+h.

        """

        f1 = self._func(t, x)  # function call
        x1 = []
        for k in range(0, len(x)):
            x1.append(x[k] + h * f1[k])
        f2 = self._func(t + h, x1)  # function call
        xf = []
        for k in range(0, len(x)):
            xf.append(x[k] + h * (f1[k] + f2[k]) / 2.)

        return xf


class RK4(Integrator):
    """Class implementing the classic Runge-Kutta 4 integration scheme.

    """

    def integration_step(self, t, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t+h where h is the step-size.

                Args:
                    t (float): current value of independent variable.
                    x (list): state vector at t.
                    h (float): step-size.

                Returns:
                    xf (list): state vector at t+h.

        """

        f1 = self._func(t, x)  # function call
        x1 = []
        for k in range(0, len(x)):
            x1.append(x[k] + h / 2. * f1[k])
        f2 = self._func(t + h / 2., x1)  # function call
        x2 = []
        for k in range(0, len(x)):
            x2.append(x[k] + h / 2. * f2[k])
        f3 = self._func(t + h / 2., x2)  # function call
        x3 = []
        for k in range(0, len(x)):
            x3.append(x[k] + h * f3[k])
        f4 = self._func(t + h, x3)  # function call
        xf = []
        for k in range(0, len(x)):
            xf.append(x[k] + h * (f1[k] + 2. * f2[k] + 2. * f3[k] + f4[k]) / 6.)

        return xf


class BS(Integrator):
    """Class implementing the Bulirsch-Stoer integration scheme.

            Attributes:
                 _extrapol (int): extrapolation order.
                 sequence (list): Burlirsch sequence of integers to be used in scheme.

    """

    def __init__(self, func, extrapol):
        """Constructor for class BS.

                Args:
                     func (function): function to be integrated.
                     extrapol (integer): extrapolation order.

        """
        Integrator.__init__(self, func)
        self._extrapol = extrapol

        sequence = []
        for i in range(0, self._extrapol):
            sequence.append(2)
            if self._extrapol > 1:
                sequence.append(4)
            if self._extrapol > 2:
                sequence.append(6)
            if self._extrapol > 3:
                for k in range(3, self._extrapol):
                    sequence.append(2 * sequence[-2])
        self.sequence = sequence

    def integration_step(self, t, x, H):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t+H where H is the step-size.

                Args:
                    t (float): current value of independent variable.
                    x (list): state vector at t.
                    H (float): step-size.

                Returns:
                     (list): state vector at t+H.

        """

        M = self._extrapolation(self._extrapol, H, x, t)

        return M[-1]

    def _midpoint(self, n, H, y, t):
        """Function applying the mid-point rule of the Bulirsch-Stoer method.

                Args:
                    n (int): order.
                    H (float): step-size.
                    y (numpy.array): current state vector.
                    t (float): current value of independent variable.

                Returns:
                    (numpy.array): output of mid-point rule.

        """

        h = H / float(n)
        h2 = 2. * h

        u0 = []
        for el in y:
            u0.append(el)

        u1 = []
        f1 = self._func(t, y)  # function call
        for i in range(0, len(y)):
            u1.append(y[i] + h * f1[i])

        u2 = []
        f2 = self._func(t + h, u1)  # function call
        for i in range(0, len(y)):
            u2.append(y[i] + h2 * f2[i])

        v = []
        w = []
        for el in y:
            v.append(el)
            w.append(el)

        for j in range(2, n+1):
            for i in range(0, len(y)):
                v[i] = u2[i]
                u2[i] = u1[i]
            f = self._func(t + j * h, v)  # function call
            for i in range(0, len(y)):
                u2[i] += h2 * f[i]
                w[i] = u1[i]
                u1[i] = v[i]
                u0[i] = w[i]

        eta = []
        for i in range(0, len(y)):
            eta.append(u0[i] / 4. + u1[i] / 2. + u2[i] / 4.)

        return eta

    def _extrapolation(self, i, H, y, t):
        """Function performing the extrapolation according to the Bulirsch-Stoer algorithm.

                Args:
                    i (int): extrapolation order.
                    H (float): step-size.
                    y (numpy.array): current state vector.
                    t (float): current value of independent variable.
                Returns:
                    M (numpy.array): concatenated extrapolated vectors.

        """

        eta = self._midpoint(self.sequence[i - 1], H, y, t)
        M = [eta]

        if i > 1:
            Mp = self._extrapolation(i - 1, H, y, t)

            for j in range(1, i):
                eta = M[j-1]
                M.append(eta)
                aux1 = float(self.sequence[i-1]) / float(self.sequence[i-1-j])
                aux2 = aux1 * aux1 - 1.
                for k in range(0, len(y)):
                    M[j][k] += (eta[k] - Mp[j-1][k]) / aux2

        return M


class MultistepFixedsize(Integrator):
    """Abstract class for the implementation of multi-step integrators with fixed step-size.

            Attributes:
                 saved_steps (list): values of state derivative at previous steps.
                 _order (int): order of integration scheme.
                 _stepsize (float): step-size.
                 _beta (list): vector of numbers used in integration scheme.
                 _initializer (Integrator): integrator used to initialize the multi-step method.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order):
        """Constructor for class MultistepFixedsize.

                Args:
                     func (function): function to be integrated.
                     order (integer): order of integrator.

        """
        Integrator.__init__(self, func)

        self._order = order
        self._stepsize = 0.
        self.saved_steps = []
        self._beta = None
        self._initializer = None

    def update_saved_steps(self, t, x):
        """Function updating the saved values of self._func at the past self._order steps.

                Args:
                    t (float): current value of independent variable.
                    x (list): state vector at t.

        """

        copy_steps = []
        for step in self.saved_steps:
            copy_steps.append(step)

        self.saved_steps = []
        for j in range(0, len(copy_steps)-1):
            self.saved_steps.append(copy_steps[j+1])  # shift

        f = self._func(t, x)  # function call
        self.saved_steps.append(f)

    def update_state(self, x):
        """Function propagating the state vector over one integration step.

                Args:
                    x (list): current state vector.

                Returns:
                    xf (list): state vector at next integration step.

        """

        xf = []
        for i in range(0, len(x)):
            xf.append(x[i])
            for j in range(0, self._order):
                xf[i] += self._stepsize * self._beta[j] * self.saved_steps[j][i]

        return xf

    def integration_step(self, t, x, h=None):
        """Function performing a single integration step.

                Args:
                    t (float): current value of independent variable.
                    x (list): state vector at t.
                    h (float): step-size (dummy variable in multi-step integrator here to match parent signature)

                Returns:
                    xf (list): state vector at t+self._stepsize.

        """

        xf = self.update_state(x)

        self.update_saved_steps(t + self._stepsize, xf)

        return xf

    def initialize(self, t0, x0, h):
        """Function initializing the integrator with a single-step scheme.

                Args:
                    t0 (float): initial value of independent variable.
                    x0 (list): state vector at t0.
                    h (float): step-size

                Returns:
                    states (list): history of state vector after initialization with single-step integrator.
                    indVar (list): values of independent variable corresponding to history of state vector.

        """

        self._stepsize = h
        n_steps = self._order - 1
        (states, indVar) = self._initializer.integrate(t0, t0 + float(n_steps) * h, x0, n_steps)

        self.saved_steps = []
        for k, state in enumerate(states):
            self.saved_steps.append(self._func(indVar[k], state))

        return states, indVar

    def integrate(self, t0, tf, x0, n_step, saved_steps=None):
        """Function that performs integration between two values of independent variable.

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (list): state vector at t0.
                    n_step (int): number of integration steps to be performed.
                    saved_steps (list): past values of self._func.

                Returns:
                    Xs (list): history of state vectors at integration steps.
                    Ts (list): values taken by the independent variable at successive integration steps.

        """
        self.saved_steps = []
        if saved_steps is not None and len(saved_steps) == self._order:
            # input saved steps are recyclable
            for el in saved_steps:
                self.saved_steps.append(el)

        h = (tf - t0) / float(n_step)

        if self._stepsize != h or self.saved_steps == []:
            (Xs, Ts) = self.initialize(t0, x0, h)
            n_start = self._order - 1
        else:  # step-size has not changed and there are available saved steps
            Ts = []
            Xs = []
            Ts.append(t0)
            x0_copy = []
            for i in range(0, len(x0)):
                x0_copy.append(x0[i])
            Xs.append(x0_copy)
            n_start = 0

        for k in range(n_start, n_step):
            Xs.append(self.integration_step(Ts[-1], Xs[-1]))
            Ts.append(Ts[-1] + self._stepsize)

        return Xs, Ts


class AB8(MultistepFixedsize):
    """Class implementing the Adam-Bashforth integration scheme of order 8.

    """

    def __init__(self, func):
        """Constructor for class AB8.

                Args:
                     func (function): function to be integrated.

        """

        MultistepFixedsize.__init__(self, func, 8)

        self._beta = [-36799.0 / 120960.0, 295767.0 / 120960.0, -1041723.0 / 120960.0, 2102243.0 / 120960.0,
                      -2664477.0 / 120960.0, 2183877.0 / 120960.0, -1152169.0 / 120960.0, 434241.0 / 120960.0]
        self._initializer = BS(self._func, int((self._order + 1) / 2))


class ABM8(MultistepFixedsize):
    """Class implementing the Adam-Bashforth-Moulton integration scheme of order 8.

    """
    def __init__(self, func):
        """Constructor for class ABM8.

                Args:
                     func (function): function to be integrated.

        """

        MultistepFixedsize.__init__(self, func, 8)

        self._beta = [1375.0/120960.0, -11351.0/120960.0, 41499.0/120960.0, -88547.0/120960.0,
                      123133.0/120960.0, -121797.0/120960.0, 139849.0/120960.0, 36799.0/120960.0]
        self._predictor = AB8(self._func)
        self._initializer = self._predictor

    def integration_step(self, t, x, h=None):
        """Function performing a single integration step.

                Args:
                    t (float): current value of independent variable.
                    x (list): state vector at t.
                    h (float): step-size (dummy variable in multi-step integrator here to match parent signature)

                Returns:
                    xf (list): state vector at t+h.

        """

        self._predictor.integration_step(t, x)

        self.saved_steps = []
        for step in self._predictor.saved_steps:
            self.saved_steps.append(step)

        xf = self.update_state(x)

        f = self._func(t + self._stepsize, xf)  # function call
        del self._predictor.saved_steps[-1]
        self._predictor.saved_steps.append(f)

        return xf
