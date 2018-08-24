import math
import numpy as np
import podaci
from numpy import array as u_niz
from numpy.polynomial.chebyshev import chebval  # chebfit
from scipy import optimize


def chebyshevval(coef, n):
    return chebval(np.linspace(-1, 1, n), coef)


def polozaj_planeta(index, t, e_prev):  # kao sadasnjost se racuna godina 2000.
    (a0, a1, e0, e1, l0, l1, omegabar0) = podaci.info[index]
    # omegabar1 = 0.44441088
    t_ = t/36525
    a = a0 + a1*t_
    e = e0 + e1*t_
    l_ = l0 + l1*t_
    e_ = 180/math.pi * e
    m = ((l_ - omegabar0) % 360)-180
    e_ = optimize.newton(lambda unk: unk - e_*math.sin(np.deg2rad(unk)) - m, e_prev, maxiter=40, tol=1e-6)
    x = a * (math.cos(np.deg2rad(e_)) - e)
    y = a * math.sqrt(1-e**2) * math.sin(np.deg2rad(e_))
    eprev = e_
    return [x, y, eprev]


def simulacija(x, y, vx, vy, gm, n):  # simulira kretanje tela samo pocetnom brzinom u gravitacionom polju sunca
    _x = np.empty(n)
    _y = np.empty(n)
    _t = np.empty(n)
    data = (x, y, vx, vy)
    r = math.sqrt(x*x+y*y)
    v = math.sqrt(vx*vx+vy*vy)
    time = 0.0
    limit = 0.007
    eprev = 0
    for i in range(n):
        a = gm/(r ** 2)
        # print(a)
        if a == 0:
            step = 3600*12
        else:
            step = math.ceil((v/a)*limit)
        eprev = polozaj_planeta(1, (time+step)/86400, eprev)[2]
        data = runge_kuta4(data, gm, step)
        _x[i] = data[0]
        _y[i] = data[1]
        _t[i] = time+step
        time = _t[i]
        r = math.sqrt(_x[i]*_x[i]+_y[i]*_y[i])
        v = math.sqrt(data[2]*data[2]+data[3]*data[3])
    return np.array((_x, _y))
    # print(time/(3600*24))
    # print(100 * ((-gm / math.sqrt(data[0] ** 2 + data[1] ** 2) + 0.5 * (data[2] ** 2 + data[3] ** 2)) / (
    #        -gm / math.sqrt(x ** 2 + y ** 2)
    #        + 0.5 * (vx * vx + vy * vy)) - 1), '%')
    # print(n * step)

    # vrednosti = splev(T,f)
    # plt.figure();
    # plt.plot(Teta,'r--',ms=2,linewidth=1);
    # plt.plot(vrednosti,'k.',linewidth=0.1,ms=2);
    # plt.show()
    # print(Teta)
    # plt.legend()


def izracunaj(gm, r, v0, k, step):  # k je niz dva koef.- jedan za izvod polozaja drugi za izvod brzine
    r = r + k[1] * step  # r je niz sa dve elem - x i y koord
    r_ = np.sqrt(r[0] * r[0] + r[1] * r[1])
    a = -1 * (r / r_) * gm / (r_ * r_)
    v = v0 + k[0] * step
    # print(r,R,a,v,'End\n')
    # print('a ', a,'shape', np.shape (a),'b ',v,'shape ', np.shape(v));
    return u_niz([a, v])


def runge_kuta4(data, gm, step):  # runge kuta 4. reda. mislim da i njegovo objasnjeneje imas na driveu.
    x, y, vx, vy = data
    r = u_niz([x, y])
    v = u_niz([vx, vy])
    k1 = izracunaj(gm, r, v, u_niz([[0, 0], [0, 0]]), 0)
    k2 = izracunaj(gm, r, v, k1, step / 2)
    k3 = izracunaj(gm, r, v, k2, step / 2)
    k4 = izracunaj(gm, r, v, k3, step)
    k = k1 + 2 * k2 + 2 * k3 + k4
    # print(k[0][0],k[0][0]);
    # print('rk: ',[r+(step/6)*k[1],v+(step/6)*k[0]]);
    return np.concatenate((r + (step / 6) * k[1], v + (step / 6) * k[0]), axis=0)
