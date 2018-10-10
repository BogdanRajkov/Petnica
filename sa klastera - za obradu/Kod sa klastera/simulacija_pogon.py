import datetime as dt
import numpy as np
import time as tm
import math

from podaci import indeksi, au  # , kernel
from newton import newton
import interpolate
import podaci
import motor

# from numba import jit, float64, int32
# from scipy.optimize import newton
# import time as tm
# import jplephem
# import julian


def modulo(vector):
    return np.sqrt(np.sum(vector ** 2))


def jed_vec(ugao):
    return np.array((math.cos(ugao), math.sin(ugao)))


def izbacivanje(days, masa_broda):
    date = dt.datetime(2019, 1, 1, 0, 0, 0) + dt.timedelta(days=int(days))
    x, x_prime, y, y_prime = polozaj_planeta(3, (date - podaci.beg_of_time).total_seconds(),
                                             r_speed=True)
    r_zemlja_ = np.array([x, y])
    r_zemlja = modulo(r_zemlja_)
    v_zemlja_ = np.array([x_prime, y_prime])
    # print(modulo(v_zemlja_))
    v_zemlja = modulo(v_zemlja_)
    v_inf = interpolate.numpy_interpol(masa_broda)
    v_brod = math.sqrt(v_inf**2 + 2*podaci.grav_par[3]/podaci.r_parking)
    r_brod_ = r_zemlja_/r_zemlja * (r_zemlja + podaci.r_parking)
    # print(modulo(r_brod_))
    v_brod_ = v_zemlja_ * (1 + v_brod/v_zemlja)
    return r_brod_, v_brod_


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


# @jit(float64[3](int32, float64), nopython=True, cache=True)
def polozaj_planeta(index, t, r_speed=False, matrica=False):  # kao sadasnjost se racuna godina 2000.
    if index == 0:
        return np.array([0.0, 0.0, 0.0])

    if matrica and not r_speed:
        if not podaci.polozaji_ucitani:
            podaci.polozaji_ucitani = True
            podaci.beg_date = np.load('datum.npy')[0]
            podaci.years_saved = np.load('years.npy')
            podaci.step_matrica = np.load('step.npy')[0]
            podaci.polozaji_matrica = np.load('polozaji_planeta.npy')
        offset = dt.timedelta(seconds=t)
        offset = podaci.beg_of_time + offset - podaci.beg_date
        index_vreme = int(offset/podaci.step_matrica)
        (x1, y1) = podaci.polozaji_matrica[index_vreme][index]
        (x2, y2) = podaci.polozaji_matrica[index_vreme + 1][index]
        # ugao1 = math.atan2(y1, x1)
        # ugao2 = math.atan2(y2, x2)
        # ugao = ugao1 + (offset-index_vreme*podaci.step_matrica)/podaci.step_matrica * (ugao2-ugao1)
        # r = (modulo(np.array(x1, y1)) + modulo(np.array(x2, y2))) / 2
        # return np.array((r*math.cos(ugao), r*math.sin(ugao)))
        return np.array((x1 + (offset-index_vreme*podaci.step_matrica)/podaci.step_matrica * (x2-x1),
                         y1 + (offset-index_vreme*podaci.step_matrica)/podaci.step_matrica * (y2-y1)))
    else:
        a0, a1, e0, e1, l0, l1, omegabar0, omegabar1 = podaci.info[index]
        # omegabar1 = 0.44441088
        t = t / (24*3600)
        t_ = t / 36525
        a0 = a0 + a1*t_
        e0 = e0 + e1*t_
        l0 = l0 + l1*t_
        omegabar0 = omegabar0 + omegabar1*t_
        m = np.deg2rad(((l0 - omegabar0) % 360) - 180)
        e_ = newton(e0, m, m, tol=1e-10, maxiter=10)
        if not r_speed:
            e_ = e_ * 1
        x = a0 * (math.cos(e_) - e0) * au
        y = a0 * math.sqrt(1-e0**2) * math.sin(e_) * au
    
        if r_speed:
            m_prime = np.deg2rad(l1 - omegabar1)  # /cty
            e_prime = (m_prime + e1*math.sin(e_))/(1 - e0*math.cos(e_))  # /cty
            x_prime = a1*(math.cos(e_)-e0) - a0*math.sin(e_)*e_prime - a0*e1  # au/cty
            x_prime = x_prime * au / (36525*24*3600)  # iz au/cty u m/s
            y_prime = (a1*math.sqrt(1-e0**2)*math.sin(e_)
                       - a0*e0*e1*math.sin(e_)/math.sqrt(1-e0**2)
                       + a0*math.sqrt(1-e0**2)*math.cos(e_)*e_prime)
            y_prime *= au / (36525*24*3600)
            return np.array([x, x_prime, y, y_prime], dtype=np.float64)
        return np.array([x, y, e_], dtype=np.float64)


# def ephemeris(index, t):
#    time = julian.julian.to_jd(podaci.beg_of_time + dt.timedelta(seconds=t))
#    if index == 0:
#        index = 10
#    return kernel[0, index].compute(time)


# @jit(cache=True, nopython=True)
def a_gravitacija(t, r_):  # vraca niz ubrzanja od sva
    n = len(indeksi)
    _pol_g = np.empty((n, 2))
    e_previ_new = np.empty((n,))
    for i in range(n):
        start = tm.process_time()
        rez_tren = polozaj_planeta(indeksi[i], t, matrica=False)
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
    dist_dest = np.take(moduo, podaci.destination)
    min_index = np.argmin(moduo)
    min_dist = moduo[min_index]
    if np.product(_crash) < 0:
        podaci.crashed = True
    # print("moduo:", moduo)
    # print(moduo.shape())
    temp = np.take(podaci.grav_par, indeksi)
    a_ = _rel_pol.T/(moduo ** 3)
    a_ = (a_ * temp).T
    # print("grav: ", np.sqrt(np.sum(a_ ** 2, axis=1)))
    return np.array((np.sum(a_, axis=0), e_previ_new, (min_dist, podaci.indeksi[min_index]), dist_dest))


# simulira kretanje tela samo pocetnom brzinom u gravitacionom polju sunca
# @jit(float64[:, :](float64[:], float64[:], float64[:],
#                   float64[:], float64[:], float64), cache=True)
def simulacija(days, fuel_mass, uglovi, snaga, y_max):
    fuel_mass = podaci.min_fuel_mass + fuel_mass/256 * (podaci.max_fuel_mass - podaci.min_fuel_mass)
    brod = np.array([podaci.dry_mass, fuel_mass])
    r_, v_ = izbacivanje(days, np.sum(brod))
    time_offset = (dt.datetime(2019, 1, 1, 0, 0, 0) + dt.timedelta(days=int(days)) - podaci.beg_of_time).total_seconds()
    podaci.crashed = False
    min_dist_dest = 1e150 * np.ones((len(podaci.destination), 2))
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
    # prev_sol = 0
    while time < t_max:
        # print(' ', time/t_max)
        # start = tm.process_time()
        ind = math.floor(time / t_max * n)
        dry_mass, fuel_mass = brod
        if fuel_mass <= 0:
            motor_uklj = False
        a_, e_previ, curr_dist, curr_dist_dest = a_gravitacija(time + time_offset, r_)
        if math.isnan(modulo(a_)):
            print(time + time_offset)
            print(r_)
            raise ValueError('Ubrzanje je nan')
        for _ in range(len(podaci.destination)):
            if curr_dist_dest[_] < min_dist_dest[_][0]:
                min_dist_dest[_][0] = curr_dist_dest[_]
                min_dist_dest[_][1] = time
        if motor_uklj and snaga[ind] == 1:
            if math.isnan(motor.thrust(r, time)):
                print(r)
                print(time)
                raise ValueError('Motor ne radi')
            a_ = a_ + motor.thrust(r, time) / (dry_mass + fuel_mass) * jed_vec(uglovi[ind])
        a = modulo(a_)
        if a == 0:
            step = 3600*12
        else:
            if math.isnan(a) or math.isnan(v):
                print('brzina: ', v)
                print('ubrzanje: ', a)
                print('motor: ', motor.thrust(r,time))
                raise ValueError
            else:
              step = math.ceil((v/a)*limit)

        if motor.flow_rate(r, time) * step > fuel_mass and motor_uklj:
            step = math.ceil(fuel_mass / motor.flow_rate(r, time))

        if math.floor((time + step) / t_max * n) != ind:
            step = math.ceil((ind+1) * t_max/n - time)

        if step > 0.15 * curr_dist[0] / v:
            step = 0.15 * curr_dist[0] / v
            # print('blizu neke planete ', modulo(r_)/podaci.au, curr_dist[0], curr_dist[1])

        if podaci.crashed:
            print('crash')
            print(min_dist_dest)
            return _r[:i - 1], _v[:i - 1], _step[:i - 1], min_dist_dest

        (r_, v_, brod) = runge_kuta4(r_, v_, time + time_offset, brod, uglovi[ind], step, motor_uklj, snaga[ind])

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
    return _r[:i-1], _v[:i-1], _step[:i-1], min_dist_dest
