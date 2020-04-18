# integrators.py: range of classes implementing integrators
# Copyright(C) 2018-2020 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

from abc import ABCMeta, abstractmethod
import numpy as np


class Integrator:
    """Abstract class for the implementation of numerical integrators.

        Attributes:
            _func (Callable): function of the independent variable and the state vector defining the derivative of the 
            latter w.r.t. the former.
            _order (int): order of integration scheme.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order):
        """Constructor for class Integrator.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative 
                     of the latter w.r.t. the former.
                     order (int): order of integration scheme.

        """

        self._func = func
        self._order = order

    @abstractmethod
    def integrate(self, t0, tf, x0, n_step, keep_history):
        """Abstract method to implement. Performs the numerical integration between initial to final values of
        independent variable, with provided initial conditions.

                Args:
                     t0 (float): initial time.
                     tf (int): final time.
                     x0 (iterable): initial conditions.
                     n_step (int): number of integrations steps.
                     keep_history (bool): set to True to return the whole history of successful steps, False to return
                     only the initial and final states.

        """
        raise NotImplementedError


class FixedstepIntegrator(Integrator):
    """Abstract class for the implementation of numerical integrators with a fixed step-size.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order):
        """Constructor for class FixedstepIntegrator.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.
                     order (int): order of integration scheme.

        """

        Integrator.__init__(self, func, order)

    @staticmethod
    def step_size(t0, tf, n_step):
        """Static method computing the constant step-size corresponding to initial and final values of independent
        variable as well as number of integration steps.

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    n_step (int): number of steps.

                Returns:
                    (float): step-size.

        """
        return (tf - t0) / n_step

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

    def integrate(self, t0, tf, x0, n_step, keep_history):
        """Function that performs integration between two values of independent variable. It is vectorized w.r.t. x0 if
        self._func is: in other words, several initial states can be propagated in one call (with the same value for the
        initial independent variable and the same number of steps).

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (iterable): state vector at t0.
                    n_step (int): number of integration steps to be performed.
                    keep_history (bool): set to True to return the whole history of successful steps, False to return
                     only the initial and final states.

                Returns:
                    Xs (List): state vectors at integration steps of interest.
                    Ts (List): values taken by the independent variable at integration steps.

        """

        h = self.step_size(t0, tf, n_step)
        Ts, Xs = [t0], [x0]
        if keep_history:
            for k in range(0, n_step):
                Xs.append(self.integration_step(Ts[k], Xs[k], h))
                Ts.append(Ts[k] + h)
        else:
            # first step
            Xs.append(self.integration_step(t0, x0, h))
            Ts.append(t0 + h)
            # rest of integration
            for __ in range(1, n_step):
                Xs[1] = self.integration_step(Ts[1], Xs[1], h)
                Ts[1] += h

        return Xs, Ts


class Euler(FixedstepIntegrator):
    """Class implementing the classic Euler integration scheme.

    """

    def __init__(self, func):
        """Constructor for Euler class.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.

        """
        FixedstepIntegrator.__init__(self, func, order=1)

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

        return x + h * self._func(t, x)  # function call


class Heun(FixedstepIntegrator):
    """Class implementing the Heun integration scheme.

        Attributes:
            _half_step (float): stored half step-size.

    """

    def __init__(self, func):
        """Constructor for Heun class.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.

        """
        FixedstepIntegrator.__init__(self, func, order=2)
        self._half_step = None

    def integrate(self, t0, tf, x0, n_step, keep_history):
        """Overload parent implementation in order to pre-compute the half step-size.

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (iterable): state vector at t0.
                    n_step (int): number of integration steps to be performed.
                    keep_history (bool): set to True to return the whole history of successful steps, False to return
                     only the initial and final states.

                Returns:
                    Xs (List): state vectors at integration steps of interest.
                    Ts (List): values taken by the independent variable at integration steps.

        """

        self._half_step = 0.5 * self.step_size(t0, tf, n_step)
        return FixedstepIntegrator.integrate(self, t0, tf, x0, n_step, keep_history)

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

        return x + self._half_step * (f1 + f2)


class RK4(FixedstepIntegrator):
    """Class implementing the classic Runge-Kutta 4 integration scheme.

        Attributes:
            _half_step (float): stored half step-size.
            _one_third_step (float): stored one third of step-size.
            _one_sixth_step (float): stored one sixth of step-size.

    """

    def __init__(self, func):
        """Constructor for RK4 class.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.

        """
        FixedstepIntegrator.__init__(self, func, order=4)
        self._half_step = None
        self._one_third_step = None
        self._one_sixth_step = None

    def integrate(self, t0, tf, x0, n_step, keep_history):
        """Overload parent implementation in order to pre-compute quantities such as the half step-size.

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (iterable): state vector at t0.
                    n_step (int): number of integration steps to be performed.
                    keep_history (bool): set to True to return the whole history of successful steps, False to return
                     only the initial and final states.

                Returns:
                    Xs (List): state vectors at integration steps of interest.
                    Ts (List): values taken by the independent variable at integration steps.

        """

        h = self.step_size(t0, tf, n_step)
        self._half_step = h / 2.
        self._one_third_step = h / 3.
        self._one_sixth_step = h / 6.
        return FixedstepIntegrator.integrate(self, t0, tf, x0, n_step, keep_history)

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

        middle_time = t + self._half_step

        f1 = self._func(t, x)  # function call

        x1 = x + self._half_step * f1
        f2 = self._func(middle_time, x1)  # function call

        x2 = x + self._half_step * f2
        f3 = self._func(middle_time, x2)  # function call

        x3 = x + h * f3
        f4 = self._func(t + h, x3)  # function call

        return x + (self._one_sixth_step * (f1 + f4) + self._one_third_step * (f2 + f3))


class BS(FixedstepIntegrator):
    """Class implementing the Bulirsch-Stoer integration scheme.

            Attributes:
                 _sequence (array_like): Burlirsch sequence of integers to be used in scheme.

    """

    def __init__(self, func, order):
        """Constructor for BS class.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.
                     order (int): order of integrator.

        """
        FixedstepIntegrator.__init__(self, func, order)

        self._sequence = np.zeros(self._order, dtype=int)
        self._sequence[0] = 2
        if self._order > 1:
            self._sequence[1] = 4
            if self._order > 2:
                self._sequence[2] = 6
                for k in range(3, self._order):
                    self._sequence[k] = 2 * self._sequence[k - 2]

        # pre-compute intermediate quantities for extrapolation
        self._aux_extrap = np.zeros((self._order + 1, self._order + 1))
        inter = 1. / np.flip(self._sequence)
        for i, el in enumerate(self._sequence[1:], 1):
            self._aux_extrap[i + 1, : i] = el * inter[-i:]
        self._aux_extrap = 1. / (self._aux_extrap ** 2 - 1.)

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
                    y (np.array): current state vector.
                    t (float): current value of independent variable.

                Returns:
                    (np.array): output of mid-point rule.

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

        return 0.25 * (u0 + u2) + 0.5 * u1

    def _extrapolation(self, i, H, y, t):
        """Function performing the extrapolation according to the Bulirsch-Stoer algorithm.

                Args:
                    i (int): extrapolation order.
                    H (float): step-size.
                    y (np.array): current state vector.
                    t (float): current value of independent variable.
                Returns:
                    M (np.array): concatenated extrapolated vectors.

        """

        M = [self._midpoint(self._sequence[i - 1], H, y, t)]

        if i > 1:
            Mp = self._extrapolation(i - 1, H, y, t)  # recursive call
            for j, el in enumerate(Mp):
                eta = M[j]
                M.append(eta + (eta - el) * self._aux_extrap[i, j])

        return M


class MultistepIntegrator(FixedstepIntegrator):
    """Abstract class for the implementation of multi-step integrators with fixed step-size.

            Attributes:
                 saved_steps (List): values of state derivative at previous steps.
                 _stepsize (float): step-size.
                 _beta (array_like): vector of numbers used in integration scheme.
                 _initializer (FixedstepIntegrator): integrator used to initialize the multi-step method.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order):
        """Constructor for class MultistepIntegrator.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.
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

        self.saved_steps = self.saved_steps[1:]  # shift
        f = self._func(t, x)  # function call
        self.saved_steps.append(f)

    def update_state(self, x):
        """Function propagating the state vector over one integration step.

                Args:
                    x (iterable): current state vector.

                Returns:
                    xf (iterable): state vector at next integration step.

        """

        dx = sum(step * beta for step, beta in zip(self.saved_steps, self._beta))

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
                    states (List): history of state vector after initialization with single-step integrator.
                    ind_vars (List): values of independent variable corresponding to history of state vector.

        """

        self._stepsize = h
        n_steps = self._order - 1
        states, ind_vars = self._initializer.integrate(t0, t0 + float(n_steps) * h, x0, n_steps, keep_history=True)

        self.saved_steps = [self._func(ind_var, state) for ind_var, state in zip(ind_vars, states)]

        return states, ind_vars

    def integrate(self, t0, tf, x0, n_step, keep_history, saved_steps=None):
        """Function that performs integration between two values of independent variable. It is vectorized w.r.t. x0 if
        self._func is: in other words, several initial states can be propagated in one call (with the same value for the
        initial independent variable and the same number of steps).

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (iterable): state vector at t0.
                    n_step (int): number of integration steps to be performed.
                    keep_history (bool): set to True to return the whole history of successful steps, False to return
                     only the initial and final states.
                    saved_steps (List): past values of self._func.

                Returns:
                    Xs (List): state vectors at integration steps of interest.
                    Ts (List): values taken by the independent variable at integration steps.

        """
        self.saved_steps = []
        if saved_steps is not None and len(saved_steps) == self._order:
            # input saved steps are recyclable
            self.saved_steps = list(saved_steps)

        h = self.step_size(t0, tf, n_step)

        # initialize steps
        if self._stepsize != h or self.saved_steps == []:
            Xs, Ts = self.initialize(t0, x0, h)
            n_start = len(Ts) - 1  # number of steps already performed
            if n_step <= n_start:
                # enough steps have already been performed, integration is over
                if keep_history:
                    return Xs[:n_step + 1], Ts[:n_step + 1]
                else:
                    return [x0, Xs[n_step + 1]], [t0, Ts[n_step + 1]]
            elif not keep_history:
                Ts, Xs = [t0, Ts[-1]], [x0, Xs[-1]]
        else:  # step-size has not changed and there are available saved steps
            Ts, Xs = [t0, t0 + self._stepsize], [x0, self.integration_step(t0, x0)]
            n_start = 1  # number of steps already performed

        # perform the rest of the integration
        if keep_history:
            for k in range(n_start, n_step):
                Xs.append(self.integration_step(Ts[k], Xs[k]))
                Ts.append(Ts[k] + self._stepsize)
        else:
            for __ in range(n_start, n_step):
                Xs[1] = self.integration_step(Ts[1], Xs[1], self._stepsize)
                Ts[1] += self._stepsize

        return Xs, Ts


class AB8(MultistepIntegrator):
    """Class implementing the Adam-Bashforth integration scheme of order 8.

    """

    def __init__(self, func):
        """Constructor for class AB8.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.

        """

        MultistepIntegrator.__init__(self, func, order=8)

        self._beta = np.array([-36799., 295767., -1041723., 2102243., -2664477., 2183877., -1152169., 434241.]) / 120960.
        self._initializer = BS(self._func, (self._order + 1) // 2)


class ABM8(MultistepIntegrator):
    """Class implementing the Adam-Bashforth-Moulton integration scheme of order 8.

    """
    def __init__(self, func):
        """Constructor for class ABM8.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.

        """

        MultistepIntegrator.__init__(self, func, order=8)

        self._beta = np.array([1375., -11351., 41499., -88547., 123133., -121797., 139849., 36799.]) / 120960.
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

        self._predictor.integration_step(t, x)  # (hides a function call)

        self.saved_steps = list(self._predictor.saved_steps)

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
                _abs_tol (array_like): tolerance vector on estimated absolute error. Should have same number of
                components than there are state variables. Default is 1.e-8 for each.
                _rel_tol (array_like): tolerance vector on estimated relative error. Should have same number of
                components than there are state variables. Default is 1.e-4 for each.
                _max_stepsize (float): maximum step-size allowed. Default is + infinity.
                _step_multiplier (float): multiplicative factor to increase step-size when an integration step has
                been successful.

    """

    __metaclass__ = ABCMeta

    def __init__(self, func, order, dim_state, abs_error_tol=None, rel_error_tol=None, max_stepsize=None,
                 step_multiplier=None):
        """Constructor for class VariableStepIntegrator.

                Args:
                     func (Callable): function of the independent variable and the state vector defining the derivative
                     of the latter w.r.t. the former.
                     order (int): order of integrator.
                     dim_state (int): dimension of state factor.
                     abs_error_tol (array_like): tolerance vector on estimated absolute error. Should have same number
                     of components than there are state variables. Default is 1.e-8 for each.
                     rel_error_tol (array_like): tolerance vector on estimated relative error. Should have same number
                     of components than there are state variables. Default is 1.e-4 for each.
                     max_stepsize (float): maximum step-size allowed. Default is + infinity.
                     step_multiplier (float): multiplicative factor to increase step-size when an integration step has
                     been successful.

        """

        Integrator.__init__(self, func, order)

        self._dim_state = dim_state

        self._last_step_ok = True
        self._error_exponent = None

        default_step_multiplier = 2.
        if step_multiplier is None:
            self._step_multiplier = default_step_multiplier
        else:
            if 1. <= step_multiplier <= 5.:
                self._step_multiplier = float(step_multiplier)
            else:
                print("input step multiplier is not in [1, 5], switching to default value of "
                      + str(default_step_multiplier))
                self._step_multiplier = default_step_multiplier

        self._max_stepsize = np.inf if max_stepsize is None else max_stepsize

        default_abs_tol = 1.e-8
        self._abs_tol = np.ones(self._dim_state) * default_abs_tol
        if abs_error_tol is not None:
            if len(abs_error_tol) != self._dim_state:
                raise ValueError("wrong input in VariableStepIntegrator: tolerance on absolute error must have same "
                                 "dimension than state vector")
            for i, tol in enumerate(abs_error_tol):
                if tol <= 0.:
                    print("input tolerance on absolute error is negative, switching to default value of "
                          + str(default_abs_tol) + " with state variable " + str(i))
                else:
                    self._abs_tol[i] = tol

        default_rel_tol = 1.e-4
        self._rel_tol = np.ones(self._dim_state) * default_rel_tol
        if rel_error_tol is not None:
            if len(rel_error_tol) != self._dim_state:
                raise ValueError("wrong input in VariableStepIntegrator: tolerance on relative error must have same "
                                 "dimension than state vector")
            for i, tol in enumerate(rel_error_tol):
                if tol <= 0.:
                    print("input tolerance on relative error is negative, switching to default value of"
                          + str(default_rel_tol) + "with state variable" + str(i))
                else:
                    self._rel_tol[i] = tol

    @abstractmethod
    def integration_step(self, t, x, h):
        """Abstract method to be overwritten in classes inheriting from abstract class. Performs a single integration
        step.

                Args:
                    t (float): current value of independent variable.
                    x (iterable): state vector at t.
                    h (float): current step-size.

        """
        raise NotImplementedError

    def integrate(self, t0, tf, x0, n_step, keep_history):
        """Function that performs integration between two values of independent variable.

                Args:
                    t0 (float): initial value of independent variable.
                    tf (float): final value of independent variable.
                    x0 (iterable): state vector at t0.
                    n_step (int): initial guess for number of integration steps.
                    keep_history (bool): set to True to return the whole history of successful steps, False to return
                     only the initial and final states.

                Returns:
                    Xs (List): state vectors at integration steps of interest.
                    Ts (List): values taken by the independent variable at integration steps.

        """

        if len(x0) != self._dim_state:
            raise ValueError("wrong input in integrate: state vector has different dimension than the one given when "
                              "the integrator was instantiated")

        # initial guess for step-size
        h = FixedstepIntegrator.step_size(t0, tf, n_step)

        # save direction of integration
        forward = tf > t0

        if keep_history:
            Ts, Xs = [t0], [x0]
        else:
            Ts, Xs = [t0, t0], [x0, x0]

        t = t0
        abs_dt = abs(tf - t0)
        while abs(t - t0) < abs_dt:
            # check and possibly decrease step-size
            if abs(h) > self._max_stepsize:
                h = self._max_stepsize if forward else -self._max_stepsize
            if (t + h > tf and forward) or (t + h < tf and not forward):
                h = tf - t

            # compute candidate new state and associated integration error
            x, err = self.integration_step(t, Xs[-1], h)

            # check viability of integration step
            err_ratios = np.fabs(err) / np.max(self._abs_tol + self._rel_tol * np.fabs(x))
            max_err_ratio = np.max(err_ratios)
            self._last_step_ok = max_err_ratio < 1.

            if self._last_step_ok:
                factor = self._step_multiplier
                t += h
                if keep_history:
                    Ts.append(t)
                    Xs.append(x)
                else:
                    Ts[1], Xs[1] = t, x
            else:  # step was not successful
                factor = 0.9 * (1. / float(max_err_ratio)) ** self._error_exponent

            # step-size update
            h *= factor

        return Xs, Ts


class RKF45(VariableStepIntegrator):
    """Class implementing the Runge-Kutta-Fehlberg 4(5) integration scheme.

            Attributes:
                _factor_t3 (float): pre-computed factor involved in calculation of t3
                _factor_t4 (float): pre-computed factor involved in calculation of t4
                _factor_x2 (float): pre-computed factor involved in calculation of x2
                _factor_x3 (float): pre-computed factor involved in calculation of x3
                _factor_x4_f1 (float): pre-computed factor multiplied by f1 to obtain x4
                _factor_x4_f3 (float): pre-computed factor multiplied by f3 to obtain x4
                _factor_x4_f4 (float): pre-computed factor multiplied by f4 to obtain x4
                _factor_x5_f1 (float): pre-computed factor multiplied by f1 to obtain x5
                _factor_x5_f3 (float): pre-computed factor multiplied by f3 to obtain x5
                _factor_x5_f4 (float): pre-computed factor multiplied by f4 to obtain x5
                _factor_x5_f5 (float): pre-computed factor multiplied by f5 to obtain x5
                _factor_xf_f1 (float): pre-computed factor multiplied by f1 to obtain xf
                _factor_xf_f3 (float): pre-computed factor multiplied by f3 to obtain xf
                _factor_xf_f4 (float): pre-computed factor multiplied by f4 to obtain xf
                _factor_xf_f5 (float): pre-computed factor multiplied by f5 to obtain xf
                _factor_err_f1 (float): pre-computed factor multiplied by f1 to obtain err
                _factor_err_f3 (float): pre-computed factor multiplied by f3 to obtain err
                _factor_err_f4 (float): pre-computed factor multiplied by f4 to obtain err
                _factor_err_f5 (float): pre-computed factor multiplied by f5 to obtain err
                _factor_err_f6 (float): pre-computed factor multiplied by f6 to obtain err

    """

    def __init__(self, func, dim_state, abs_error_tol=None, rel_error_tol=None, max_stepsize=None,
                 step_multiplier=None):
        VariableStepIntegrator.__init__(self, func, order=4, dim_state=dim_state, abs_error_tol=abs_error_tol,
                                        rel_error_tol=rel_error_tol, max_stepsize=max_stepsize,
                                        step_multiplier=step_multiplier)
        self._error_exponent = 1. / (self._order + 1.)
        self._factor_t3 = 3. / 8.
        self._factor_t4 = 12. / 13.
        self._factor_x2 = 3. / 32.
        self._factor_x3 = 1. / 2197.
        self._factor_x4_f1 = 439. / 216.
        self._factor_x4_f3 = 3680. / 513.
        self._factor_x4_f4 = -845. / 4104.
        self._factor_x5_f1 = -8. / 27.
        self._factor_x5_f3 = -3544. / 2565.
        self._factor_x5_f4 = 1859. / 4104.
        self._factor_x5_f5 = -11. / 40.
        self._factor_xf_f1 = 25. / 216.
        self._factor_xf_f3 = 1408. / 2565.
        self._factor_xf_f4 = 2197. / 4104.
        self._factor_xf_f5 = -1. / 5.
        self._factor_err_f1 = 16. / 135.
        self._factor_err_f3 = 6656. / 12825.
        self._factor_err_f4 = 28561. / 56430.
        self._factor_err_f5 = -9. / 50.
        self._factor_err_f6 = 2. / 55.

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
        dt2 = 0.25 * h
        t2 = t + dt2
        t3 = t + h * self._factor_t3
        t4 = t + h * self._factor_t4
        t5 = t + h
        t6 = t + 0.5 * h

        f1 = self._func(t1, x)  # function call

        x1 = x + dt2 * f1
        f2 = self._func(t2, x1)  # function call

        x2 = x + h * self._factor_x2 * (f1 + f2 * 3.)
        f3 = self._func(t3, x2)  # function call

        x3 = x + h * self._factor_x3 * (f1 * 1932. + f2 * (-7200.) + f3 * 7296.)
        f4 = self._func(t4, x3)  # function call

        x4 = x + h * (f1 * self._factor_x4_f1 + f2 * (-8.) + f3 * self._factor_x4_f3 + f4 * self._factor_x4_f4)
        f5 = self._func(t5, x4)  # function call

        x5 = x + h * (f1 * self._factor_x5_f1 + f2 * 2. + f3 * self._factor_x5_f3 + f4 * self._factor_x5_f4
                      + f5 * self._factor_x5_f5)
        f6 = self._func(t6, x5)  # function call

        inter1 = f1 * self._factor_xf_f1 + f3 * self._factor_xf_f3 + f4 * self._factor_xf_f4 + f5 * self._factor_xf_f5
        xf = h * inter1
        inter2 = f1 * self._factor_err_f1 + f3 * self._factor_err_f3 + f4 * self._factor_err_f4 \
                 + f5 * self._factor_err_f5 + f6 * self._factor_err_f6
        x_hat = h * inter2
        err = xf - x_hat
        xf += x
        return xf, err
