from numba import jit
import math


@jit(nopython=True, cache=True)
def newton(e, m, x0, tol=1.48e-8, maxiter=50):

    # Multiply by 1.0 to convert to floating point.  We don't use float(x0)
    # so it still works if x0 is complex.
    p0 = 1.0 * x0

    # Secant method
    if x0 >= 0:
        p1 = x0*(1 + 1e-4) + 1e-4
    else:
        p1 = x0*(1 + 1e-4) - 1e-4
    q0 = p0 - e * math.sin(p0) - m
    q1 = p1 - e * math.sin(p1) - m
    while True:
        if q1 == q0:
            if p1 != p0:
                return (p1 + p0)/2.0
        else:
            p = p1 - q1*(p1 - p0)/(q1 - q0)
            if abs(p - p1) < tol:
                return p
            p0 = p1
            q0 = q1
            p1 = p
            q1 = p1 - e * math.sin(p1) - m

    raise ArithmeticError
