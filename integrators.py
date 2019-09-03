# integrators.py: range of classes implementing integrators
# Copyright(C) 2019 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

from abc import ABCMeta, abstractmethod
import numpy
import math


class Integrator:
    """Abstract class for the implementation of numerical integrators.

        Attributes:
            _func (method): function to be integrated.
            _order (int): order of integration scheme.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order):
        """Constructor for class Integrator.

                Args:
                     func (function): function to be integrated.
                     order (int): order of integration scheme.

        """

        self._func = func
        self._order = order

    @abstractmethod
    def integrate(self, t0, tf, x0, n_step):
        """Abstract method to implement. Performs the numerical integration between initial to final values of
        independent variable, with provided initial conditions.

                Args:
                     t0 (float): initial time.
                     tf (int): final time.
                     x0 (iterable): initial conditions.
                     n_step (int): number of integrations steps

        """
        return NotImplementedError


class FixedstepIntegrator(Integrator):
    """Abstract class for the implementation of numerical integrators with a fixed step-size.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order):
        """Constructor for class FixedstepIntegrator.

                Args:
                     func (function): function to be integrated.
                     order (int): order of integration scheme.

        """

        Integrator.__init__(self, func, order)

    @abstractmethod
    def integration_step(self, t, x, h):
        """Abstract method to be overwritten in classes inheriting from abstract class. Performs a single integration
        step.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): step-size.

        """
        pass

    def integrate(self, t0, tf, x0, n_step):
        """Function that performs integration between two values of independent variable.

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (iterable): state vector at t0.
                    n_step (int): number of integration steps to be performed.

                Returns:
                    Xs (iterable): history of state vectors at integration steps.
                    Ts (iterable): values taken by the independent variable at successive integration steps.

        """

        h = (tf - t0) / float(n_step)
        Ts = [t0]
        Xs = [x0]
        for k in range(0, n_step):
            Xs.append(self.integration_step(Ts[-1], Xs[-1], h))
            Ts.append(Ts[-1] + h)

        return Xs, Ts


class Euler(FixedstepIntegrator):
    """Class implementing the classic Euler integration scheme.

    """

    def __init__(self, func):
        """Constructor for Euler class.

                Args:
                     func (function): function to be integrated.

        """
        FixedstepIntegrator.__init__(self, func, 1)

    def integration_step(self, t, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t + h where h is the step-size.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): step-size.

                Returns:
                    xf (iterable): state vector at t + h.

        """

        return x + h * self._func(t, x)


class Heun(FixedstepIntegrator):
    """Class implementing the Heun integration scheme.

    """

    def __init__(self, func):
        """Constructor for Heun class.

                Args:
                     func (function): function to be integrated.

        """
        FixedstepIntegrator.__init__(self, func, 2)

    def integration_step(self, t, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t + h where h is the step-size.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): step-size.

                Returns:
                    xf (iterable): state vector at t + h.

        """

        f1 = self._func(t, x)  # function call

        x1 = x + h * f1
        f2 = self._func(t + h, x1)  # function call

        return x + 0.5 * h * (f1 + f2)


class RK4(FixedstepIntegrator):
    """Class implementing the classic Runge-Kutta 4 integration scheme.

    """

    def __init__(self, func):
        """Constructor for RK4 class.

                Args:
                     func (function): function to be integrated.

        """
        FixedstepIntegrator.__init__(self, func, 4)

    def integration_step(self, t, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t + h where h is the step-size.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): step-size.

                Returns:
                    xf (iterable): state vector at t + h.

        """

        half_step = h / 2.
        middle_time = t + half_step

        f1 = self._func(t, x)  # function call
        x1 = x + half_step * f1
        f2 = self._func(middle_time, x1)  # function call

        x2 = x + half_step * f2
        f3 = self._func(middle_time, x2)  # function call

        x3 = x + h * f3
        f4 = self._func(t + h, x3)  # function call

        return x + h / 6. * (f1 + 2. * (f2 + f3) + f4)


class BS(FixedstepIntegrator):
    """Class implementing the Bulirsch-Stoer integration scheme.

            Attributes:
                 sequence (list): Burlirsch sequence of integers to be used in scheme.

    """

    def __init__(self, func, order):
        """Constructor for BS class.

                Args:
                     func (function): function to be integrated.
                     order (int): order of integrator.

        """
        FixedstepIntegrator.__init__(self, func, order)

        sequence = [2]
        if self._order > 1:
            sequence.append(4)
            if self._order > 2:
                sequence.append(6)
                if self._order > 3:
                    for k in range(3, self._order):
                        sequence.append(2 * sequence[-2])
        self.sequence = sequence

    def integration_step(self, t, x, H):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t + H where H is the step-size.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    H (float): step-size.

                Returns:
                     (iterable): state vector at t + H.

        """

        M = self._extrapolation(self._order, H, x, t)

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

        u0 = None

        f1 = self._func(t, y)  # function call
        u1 = y + h * f1

        f2 = self._func(t + h, u1)  # function call
        u2 = y + h2 * f2

        for j in range(2, n + 1):
            f = self._func(t + j * h, u2)  # function call
            u2, u1, u0 = u1 + h2 * f, u2, u1

        return (u0 + u1 * 2. + u2) / 4.

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
                M[j] += (eta - Mp[j-1]) / aux2

        return M


class MultistepIntegrator(FixedstepIntegrator):
    """Abstract class for the implementation of multi-step integrators with fixed step-size.

            Attributes:
                 saved_steps (list): values of state derivative at previous steps.
                 _stepsize (float): step-size.
                 _beta (array): vector of numbers used in integration scheme.
                 _initializer (FixedstepIntegrator): integrator used to initialize the multi-step method.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order):
        """Constructor for class MultistepIntegrator.

                Args:
                     func (function): function to be integrated.
                     order (int): order of integrator.

        """
        FixedstepIntegrator.__init__(self, func, order)

        self._stepsize = 0.
        self.saved_steps = []
        self._beta = None
        self._initializer = None

    def update_saved_steps(self, t, x):
        """Function updating the saved values of self._func at the past self._order steps.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.

        """

        copy_steps = []
        for step in self.saved_steps:
            copy_steps.append(step)

        self.saved_steps = []
        for j in range(0, len(copy_steps) - 1):
            self.saved_steps.append(copy_steps[j + 1])  # shift

        f = self._func(t, x)  # function call
        self.saved_steps.append(f)

    def update_state(self, x):
        """Function propagating the state vector over one integration step.

                Args:
                    x (iterable): current state vector.

                Returns:
                    xf (iterable): state vector at next integration step.

        """

        dx = self.saved_steps[0] * self._beta[0]
        for j in range(1, self._order):
            dx += self.saved_steps[j] * self._beta[j]

        return x + self._stepsize * dx

    def integration_step(self, t, x, h=None):
        """Function performing a single integration step.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): step-size (dummy variable in multi-step integrator here to match parent signature)

                Returns:
                    xf (iterable): state vector at t + self._stepsize.

        """

        xf = self.update_state(x)

        self.update_saved_steps(t + self._stepsize, xf)

        return xf

    def initialize(self, t0, x0, h):
        """Function initializing the integrator with a single-step scheme.

                Args:
                    t0 (float): initial value of independent variable.
                    x0 (iterable): state vector at t0.
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
                    x0 (iterable): state vector at t0.
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
            Ts = [t0]
            Xs = [x0]
            n_start = 0

        for k in range(n_start, n_step):
            Xs.append(self.integration_step(Ts[-1], Xs[-1]))
            Ts.append(Ts[-1] + self._stepsize)

        return Xs, Ts


class AB8(MultistepIntegrator):
    """Class implementing the Adam-Bashforth integration scheme of order 8.

    """

    def __init__(self, func):
        """Constructor for class AB8.

                Args:
                     func (function): function to be integrated.

        """

        MultistepIntegrator.__init__(self, func, 8)

        self._beta = numpy.array([-36799., 295767., -1041723., 2102243., -2664477., 2183877., -1152169., 434241.]) / 120960.
        self._initializer = BS(self._func, int((self._order + 1) / 2))


class ABM8(MultistepIntegrator):
    """Class implementing the Adam-Bashforth-Moulton integration scheme of order 8.

    """
    def __init__(self, func):
        """Constructor for class ABM8.

                Args:
                     func (function): function to be integrated.

        """

        MultistepIntegrator.__init__(self, func, 8)

        self._beta = numpy.array([1375., -11351., 41499., -88547., 123133., -121797., 139849., 36799.]) / 120960.
        self._predictor = AB8(self._func)
        self._initializer = self._predictor

    def integration_step(self, t, x, h=None):
        """Function performing a single integration step.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): step-size (dummy variable in multi-step integrator here to match parent signature)

                Returns:
                    xf (iterable): state vector at t + self._stepsize.

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


class VariableStepIntegrator(Integrator):
    """Abstract class for the implementation of integrators with step-size control.

            Attributes:
                _dim_state (int): dimension of state vector.
                _last_step_ok (bool): false if last step didn't satisfy the constraint on the absolute error, true
                otherwise.
                _abs_tol (iterable): tolerance vector on estimated absolute error. Should have same number of
                components than there are state variables. Default is 1.e-8 for each.
                _max_stepsize (float): maximum step-size allowed. Default is + infinity.
                _step_multiplier (float): multiplicative factor to increase step-size when an integration step has
                been successful.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order, dim_state, abs_error_tol=None, max_stepsize=None, step_multiplier=None):
        """Constructor for class VariableStepIntegrator.

                Args:
                     func (function): function to be integrated.
                     order (integer): order of integrator.
                     dim_state (int): dimension of state factor.
                     abs_error_tol (iterable): tolerance vector on estimated absolute error. Should have same number of
                     components than there are state variables. Default is 1.e-8 for each.
                     max_stepsize (float): maximum step-size allowed. Default is + infinity.
                     step_multiplier (float): multiplicative factor to increase step-size when an integration step has
                     been successful.

        """

        Integrator.__init__(self, func, order)

        self._dim_state = dim_state

        self._last_step_ok = True

        default_step_multiplier = 2.
        if step_multiplier is None:
            self._step_multiplier = default_step_multiplier
        else:
            if 1. <= step_multiplier <= 5.:
                self._step_multiplier = float(step_multiplier)
            else:
                print("input step multiplier is not in [1, 5], switching to default value of"
                      + str(default_step_multiplier))
                self._step_multiplier = default_step_multiplier

        if max_stepsize is None:
            self._max_stepsize = numpy.inf
        else:
            self._max_stepsize = max_stepsize

        default_abs_tol = 1.e-8
        self._abs_tol = numpy.ones(self._dim_state) * default_abs_tol
        if abs_error_tol is not None:
            if len(abs_error_tol) != self._dim_state:
                raise ValueError("wrong input in VariableStepIntegrator: tolerance on absolute error must have same "
                      "dimension than state vector")
            for i in range(0, len(abs_error_tol)):
                if abs_error_tol[i] <= 0.:
                    print("input tolerance on absolute error is negative, switching to default value of"
                          + str(default_abs_tol) + "with state variable" + str(i))
                else:
                    self._abs_tol[i] = abs_error_tol[i]

    @abstractmethod
    def integration_step(self, t, x, h):
        """Abstract method to be overwritten in classes inheriting from abstract class. Performs a single integration
        step.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): current step-size.

        """
        return NotImplementedError

    def integrate(self, t0, tf, x0, n_step):
        """Function that performs integration between two values of independent variable.

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (): state vector at t0.
                    n_step (int): initial guess for number of integration steps.

                Returns:
                    Xs (iterable): history of state vectors at integration steps.
                    Ts (iterable): values taken by the independent variable at successive integration steps.

        """

        if len(x0) != self._dim_state:
            return ValueError("wrong input in integrate: state vector has different dimension than the one given when "
                  "the integrator was instantiated")

        # initial guess for step-size
        h = (tf - t0) / float(n_step)

        Ts = [t0]
        Xs = [x0]

        t = t0
        while math.fabs(t - t0) < math.fabs(tf - t0):
            # check and possibly decrease step-size
            if math.fabs(h) > self._max_stepsize:
                if tf >= t0:
                    h = self._max_stepsize
                else:
                    h = -self._max_stepsize
            if (t + h > tf >= t0) or (t + h < tf < t0):
                h = tf - t

            x, err = self.integration_step(t, Xs[-1], h)

            # check viability of integration step
            self._last_step_ok = True
            max_err_ratio = 0.
            for i in range(0, len(err)):
                inter = math.fabs(err[i]) / self._abs_tol[i]
                if inter > max_err_ratio:
                    max_err_ratio = inter
                    if max_err_ratio > 1.:
                        self._last_step_ok = False

            if self._last_step_ok:
                t += h
                Ts.append(t)
                Xs.append(x)
                factor = self._step_multiplier
            else:  # step was not successful
                factor = 0.9 * (1. / max_err_ratio) ** (1. / float(self._order))

            # step-size update
            h *= factor

        return Xs, Ts


class RKF45(VariableStepIntegrator):
    """Class implementing the Runge-Kutta-Fehlberg 4(5) integration scheme.

    """

    def __init__(self, func, dim_state, abs_error_tol=None, max_stepsize=None, step_multiplier=None):
        VariableStepIntegrator.__init__(self, func, 4, dim_state, abs_error_tol, max_stepsize, step_multiplier)

    def integration_step(self, t, x, h):
        """Method performing a single integration step (satisfying the error tolerance or not).

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): current step-size.

                Returns:
                    xf (iterable): tentative state vector at t + h.
                    err (iterable): estimated error vector.

        """
        # values of independent variable where the model will be evaluated
        t1 = t
        t2 = t + h / 4.
        t3 = t + h * 3. / 8.
        t4 = t + h * 12. / 13.
        t5 = t + h
        t6 = t + h / 2.

        f1 = self._func(t1, x)  # function call

        x1 = x + (h / 4.) * f1
        f2 = self._func(t2, x1)  # function call

        x2 = x + h * (3. / 32.) * (f1 + f2 * 3.)
        f3 = self._func(t3, x2)  # function call

        x3 = x + (h / 2197.) * (f1 * 1932. + f2 * (-7200.) + f3 * 7296.)
        f4 = self._func(t4, x3)  # function call

        x4 = x + h * (f1 * (439. / 216.) + f2 * (-8.) + f3 * (3680. / 513.) + f4 * (-845. / 4104.))
        f5 = self._func(t5, x4)  # function call

        x5 = x + h * (f1 * (-8. / 27.) + f2 * 2. + f3 * (-3544. / 2565.) + f4 * (1859. / 4104.) + f5 * (-11. / 40.))
        f6 = self._func(t6, x5)  # function call

        inter1 = f1 * (25. / 216.) + f3 * (1408. / 2565.) + f4 * (2197. / 4104.) + f5 * (-1. / 5.)
        xf = h * inter1
        inter2 = f1 * (16. / 135.) + f3 * (6656. / 12825.) + f4 * (28561. / 56430.) + f5 * (-9. / 50.) + f6 * (2. / 55.)
        x_hat = h * inter2
        err = xf - x_hat
        xf += x
        return xf, err
