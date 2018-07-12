import numpy as np
# from numba import jit, float64, int32
import math
import podaci
from podaci import indeksi, au, kernel
import motor
from newton import newton
import datetime
import time as tm
import julian
# import time as tm
# from scipy.optimize import newton
# import jplephem


def modulo(vector):
    return math.sqrt(np.sum(vector ** 2))


def jed_vec(ugao):
    return np.array((math.cos(ugao), math.sin(ugao)))


# @jit(float64[:, :](float64[:], float64[:], float64[:], float64, float64,
#                   float64, float64[:], float64[:, :], float64), nopython=True, cache=True)
def izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, k=np.zeros((3, 2)), step=0.0):
    r_, v_, brod = np.array((r_, v_, brod)) + k
    r = math.sqrt(np.sum(r_ ** 2))
    a_ = a_gravitacija(t+step, r_)[0]
    delta_mass = 0
    if motor_uklj and snaga == 1:
        a_ = a_ + motor.thrust(r, t + step) / np.sum(brod) * jed_vec(ugao)
        delta_mass = -motor.flow_rate(r, t + step)
    return np.array((v_, a_, np.array((0., delta_mass))), dtype=np.float64)


# @jit(float64[:, :](float64[:], float64[:], float64, float64[:],
#                   float64, float64, float64, float64[:]), cache=True)
def runge_kuta4(r_, v_, t, brod, ugao, step, motor_uklj, snaga):
    k1 = step * izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga)
    k2 = step * izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, k1/2, step/2)
    k3 = step * izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, k2/2, step/2)
    k4 = step * izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, k3, step)
    k = k1 + 2 * k2 + 2 * k3 + k4
    return np.array((r_, v_, brod), dtype=np.float64) + k/6


# simulira kretanje tela samo pocetnom brzinom u gravitacionom polju sunca
# @jit(float64[:, :](float64[:], float64[:], float64[:],
#                   float64[:], float64[:], float64), cache=True)
def simulacija(r_, v_, brod, uglovi, snaga, y_max):
    # m = m_ukupna
    motor_uklj = True
    n = len(uglovi)
    t_max = y_max * 365.25 * 24 * 3600
    # print(t_max)
    len_nizova = int(t_max / (6*36000))
    _r = np.zeros((len_nizova, 2))
    _v = np.zeros((len_nizova, 2))
    _time = np.zeros(len_nizova)
    _step = np.zeros(len_nizova)
    r = modulo(r_)
    v = modulo(v_)
    time = 0.0
    limit = 0.007
    i = 0
    # prev_sol = 0
    while time < t_max:
        print(time, i, len_nizova)
        # start = tm.process_time()
        ind = math.floor(time / t_max * n)
        dry_mass, fuel_mass = brod
        if fuel_mass <= 0:
            motor_uklj = False
        a_, e_previ = a_gravitacija(time, r_)
        if motor_uklj and snaga[ind] == 1:
            a_ = a_ + motor.thrust(r, time) / (dry_mass + fuel_mass) * jed_vec(uglovi[ind])
        a = modulo(a_)
        if a == 0:
            step = 3600*12
        else:
            step = math.ceil((v/a)*limit)

        if motor.flow_rate(r, time) * step > fuel_mass and motor_uklj:
            step = math.ceil(fuel_mass / motor.flow_rate(r, time))
        
        if math.floor((time + step) / t_max * n) != ind:
            step = math.ceil((ind+1) * t_max/n - time)

        (r_, v_, brod) = runge_kuta4(r_, v_, time, brod, uglovi[ind], step, motor_uklj, snaga[ind])

        r = modulo(r_)
        v = modulo(v_)
        if i == len_nizova:
            _r = np.append(_r, [r_], axis=0)
            _v = np.append(_v, [v_], axis=0)
            _time = np.append(_time, [time + step], axis=0)
            _step = np.append(_step, [step], axis=0)
            len_nizova = len_nizova + 1
        else:
            _r[i] = r_
            _v[i] = v_
            _time[i] = time + step
            _step[i] = step
        time = _time[i]
        # print(time, i)
        i = i + 1
        # print(tm.process_time() - start)
    return _r[:i-1], _v[:i-1], _step[:i-1]


# @jit(cache=True, nopython=True)
def a_gravitacija(t, r_):  # vraca niz ubrzanja od sva
    n = len(indeksi)
    _pol_g = np.empty((n, 2))
    e_previ_new = np.empty((n,))
    for i in range(n):
        start = tm.process_time()
        rez_tren = polozaj_planeta(indeksi[i], t)
        podaci.trajanje = podaci.trajanje + tm.process_time() - start
        _pol_g[i] = np.array(rez_tren[:2])
        # e_previ_new[i] = rez_tren[2]
    _rel_pol = np.subtract(_pol_g, r_)
    # print(r_)
    # print("_pol_g:", _pol_g)
    # print("_pol_r:", _rel_pol)
    if n == 1:
        moduo = np.sqrt(np.sum(_rel_pol ** 2))
    else:
        moduo = np.sqrt(np.sum(_rel_pol ** 2, axis=1))
    _crash = moduo - np.take(podaci.planet_radii, indeksi)
    if np.product(_crash) < 0:
        raise ValueError
    # print("moduo:", moduo)
    # print(moduo.shape())
    temp = np.take(podaci.grav_par, indeksi)
    a_ = _rel_pol.T/(moduo ** 3)
    a_ = (a_ * temp).T
    # print("grav: ", np.sqrt(np.sum(a_ ** 2, axis=1)))
    return np.array((np.sum(a_, axis=0), e_previ_new))


# @jit(float64[3](int32, float64), nopython=True, cache=True)
def polozaj_planeta(index, t):  # kao sadasnjost se racuna godina 2000.
    if index == 0:
        return np.array([0.0, 0.0, 0.0])

    (a0, a1, e0, e1, l0, l1, omegabar0) = podaci.info[index]
    # omegabar1 = 0.44441088
    t = t/(24*3600)
    t_ = t/36525
    a = a0 + a1*t_
    e = e0 + e1*t_
    l_ = l0 + l1*t_
    m = np.rad2deg(((l_ - omegabar0) % 360) - 180)
    e_ = newton(e, m, m, tol=1e-10, maxiter=10)
    x = a * (math.cos(e_) - e) * au
    y = a * math.sqrt(1-e**2) * math.sin(e_) * au
    return np.array([x, y, e_], dtype=np.float64)


def ephemeris(index, t):
    time = julian.julian.to_jd(podaci.beg_of_time + datetime.timedelta(seconds=t))
    if index == 0:
        index = 10
    return kernel[0, index].compute(time)
