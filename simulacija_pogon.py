import numpy as np
import math
import podaci
from podaci import indeksi
import motor
import scipy.optimize as optimize
# import time as tm


def modulo(vector):
    return math.sqrt(np.sum(vector ** 2))


def jed_vec(ugao):
    return np.array((math.cos(ugao), math.sin(ugao)))


def izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, e_previ, k=np.zeros((3, 2)), step=0.0):
    r_, v_, brod = np.array((r_, v_, brod)) + k
    r = modulo(r_)
    a_ = a_gravitacija(t+step, e_previ, r_)
    delta_mass = 0
    if motor_uklj and snaga == 1:
        a_ = a_ + motor.thrust(r, t + step) / np.sum(brod) * jed_vec(ugao)
        delta_mass = -motor.flow_rate(r, t + step)
    return np.array((v_, a_, np.array((0, delta_mass))))


def runge_kuta4(r_, v_, t, brod, ugao, step, motor_uklj, snaga, e_previ):
    k1 = step * izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, e_previ)
    k2 = step * izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, e_previ, k1/2, step/2)
    k3 = step * izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, e_previ, k2/2, step/2)
    k4 = step * izracunaj(r_, v_, t, brod, ugao, motor_uklj, snaga, e_previ, k3, step)
    k = k1 + 2 * k2 + 2 * k3 + k4
    return np.array((r_, v_, brod)) + k/6


# simulira kretanje tela samo pocetnom brzinom u gravitacionom polju sunca
def simulacija(r_, v_, brod, uglovi, snaga, y_max):
    # m = m_ukupna
    motor_uklj = True
    n = len(uglovi)
    t_max = y_max * 365.25 * 24 * 3600
    # print(t_max)
    len_nizova = int(t_max / (6*3600))
    _r = np.zeros((len_nizova, 2))
    _v = np.zeros((len_nizova, 2))
    _time = np.zeros(len_nizova)
    _step = np.zeros(len_nizova)
    r = modulo(r_)
    v = modulo(v_)
    time = 0.0
    limit = 0.007
    i = 0
    e_previ = np.zeros(len(indeksi))
    # prev_sol = 0
    while time < t_max:
        # start = tm.process_time()
        ind = math.floor(time / t_max * n)
        dry_mass, fuel_mass = brod
        if fuel_mass <= 0:
            motor_uklj = False
        a_, e_previ = a_gravitacija(time, e_previ, r_)
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
        
        (r_, v_, brod) = runge_kuta4(r_, v_, time, brod, uglovi[ind], step, motor_uklj, snaga[ind], e_previ)

        r = modulo(r_)
        v = modulo(v_)
        if i == len_nizova:
            _r = np.append(_r, np.empty(100))
            _v = np.append(_v, np.empty(100))
            _time = np.append(_time, np.empty(100))
            _step = np.append(_step, np.empty(100))
        _r[i] = r_
        _v[i] = v_
        _time[i] = time+step
        _step[i] = step
        time = _time[i]
        # print(time, i)
        i = i + 1
        # print(tm.process_time() - start)
    return _r[:i-1], _v[:i-1], _step[:i-1]


def a_gravitacija(t, e_previ, r_):  # vraca niz ubrzanja od sva

    n = len(indeksi)
    _pol_g = np.empty((n, 2))
    e_previ_new = np.empty((n,))
    for i in range(n):
        rez_tren = polozaj_planeta(indeksi[i], t, e_previ[i])
        _pol_g[i] = np.array(rez_tren[:2])
        e_previ_new[i] = rez_tren[2]
    _rel_pol = np.subtract(_pol_g, r_)
    print(r_)
    print(_pol_g)
    moduo = np.sqrt(np.sum(_rel_pol ** 2, axis=1))
    # print(moduo.shape())
    temp = np.take(podaci.grav_par, indeksi)
    a_ = -(_rel_pol.T/(moduo ** 3)) * temp
    return np.sum(a_, axis=1), e_previ_new


def polozaj_planeta(index, t, e_prev):  # kao sadasnjost se racuna godina 2000.
    if index == 0:
        return [0.0, 0.0, 0.0]

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
