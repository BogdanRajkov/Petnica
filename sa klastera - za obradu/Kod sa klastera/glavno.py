from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import time as tm

from podaci import broj_segmenata, broj_jedinki, broj_gen, y_max, max_time_span  # , au
from genetski_algoritam import fitnes
import genetski_algoritam
import simulacija_pogon
import podaci

# import datetime as dt
# import datetime
# import timeit
# import math


def x_osa(a):
    return np.linspace(0, 1, a)


def crtanje_planeta(plot=True):
    broj_tacaka = int(y_max*365*2)
    broj_planeta = 5
    t = np.linspace(0, y_max*365*24*3600, broj_tacaka)
    x = [[] for _ in range(broj_planeta)]
    y = [[] for _ in range(broj_planeta)]
    indeksi_planeta = [1, 2, 3, 4, 5, 6, 7, 8]  # indeksi planeta koje crtamo
    for i in range(broj_tacaka):
        for j in range(broj_planeta):  # koliko planeta crtamo
            rez = simulacija_pogon.polozaj_planeta(indeksi_planeta[j], t[i], matrica=False)
            x[j].append(rez[0])
            y[j].append(rez[1])

    if plot:
        plt.plot(x[0], y[0], 'g', x[1], y[1], 'y',
                 x[2], y[2], 'b', x[3], y[3], linewidth=0.8)

        # plt.setp(putanje,markersize=2.5)
        plt.plot(0.0, 0.0, 'k*', markersize=7)
        plt.axis('scaled')
        plt.title('Putanje Merkura, Venere, Zemlje i Marsa oko Sunca')
        plt.xlabel('x koord [astronomska jedinica]')
        plt.ylabel('y koord [astronomska jedinica]')
        plt.show()
    else:
        return x, y


def pop_init():
    rand_days = np.random.random_integers(max_time_span, size=broj_jedinki).astype(np.uint16)
    fuel_mass = np.random.random_integers(0, 255, broj_jedinki).astype(np.uint8)
    uglovi_matrica = np.multiply(np.random.random_sample((broj_jedinki, broj_segmenata)), 2 * pi)
    snaga = np.random.random_integers(0, 1, (broj_jedinki, broj_segmenata)).astype(np.uint8)
    return rand_days, fuel_mass, uglovi_matrica, snaga


def float_to_bit(days, fuel_mass, koeficijenti, snaga, fit):
    days_bit = np.unpackbits(days.view(np.uint8), axis=1)
    fuel_bit = np.unpackbits(np.array([fuel_mass]).T, axis=1)
    koef_bit = np.unpackbits(koeficijenti.view(np.uint8), axis=1)
    return np.concatenate((days_bit, fuel_bit, koef_bit, snaga, np.array([fit]).T), axis=1)


def bit_to_float(matrica_bit):
    matrica_bit, fitness = np.split(matrica_bit, [-1], axis=1)
    days_bit, fuel_bit, koeficijenti_bit, snaga = np.split(matrica_bit,
                                                           np.array((16, 24, (1+podaci.chebdeg)*64 + 24)),
                                                           axis=1)
    days = np.packbits(days_bit.astype(bool), axis=1).view(np.uint16)
    brod = np.packbits(fuel_bit.astype(bool), axis=1).view(np.uint8)
    koeficijenti = np.packbits(koeficijenti_bit.astype(bool), axis=1).view(np.float64)
    uglovi = chebval(x_osa(broj_segmenata), koeficijenti)
    return days, brod, uglovi, snaga, fitness


# start = time.process_time()
def main(plot=False):
    days, fuel_mass, uglovi, snaga = pop_init()
    pop_fit = np.empty(broj_jedinki)
    koeficijenti = np.empty((broj_jedinki, podaci.chebdeg + 1))
    for i in range(broj_gen):
        start = tm.process_time()
        for j in range(broj_jedinki):
            # print(j)
            _r, _, _, min_dist_dest = simulacija_pogon.simulacija(days[j], fuel_mass[j], uglovi[j], snaga[j], y_max)
            # if pop_fit[j] == -1:
            pop_fit[j] = fitnes(min_dist_dest)
            koeficijenti[j] = chebfit(x_osa(broj_segmenata), uglovi[j], podaci.chebdeg)
            if plot:
                x, y = crtanje_planeta(plot=False)
                plt.plot(x[0], y[0], 'g',
                         x[1], y[1], 'y',
                         x[2], y[2], 'b--',
                         x[3], y[3], 'r',
                         _r[:, 0], _r[:, 1],
                         linewidth=0.8)
                plt.plot(0.0, 0.0, 'k*', markersize=7)
                plt.axis('scaled')
                
                plt.show()
        print(tm.process_time() - start)
        print(podaci.trajanje)
        pop_bit = float_to_bit(days[:, np.newaxis], fuel_mass, koeficijenti, snaga, pop_fit)
        # print(pop_bit)
        pop_bit_new = genetski_algoritam.genetski_algoritam(pop_bit, podaci.p_elit, podaci.p_mut)
        date_time, fuel_mass, uglovi, snaga, pop_fit = bit_to_float(pop_bit_new)
        print(np.min(pop_fit))


if __name__ == "__main__":
    main()

# uglovi = np.ones(broj_segmenata) * pi
# snaga = np.random.random_integers(0, 1, (broj_jedinki, broj_segmenata))

# date_time, brod, uglovi_matrica, snaga = pop_init()
# print(date_time[0])
# print(brod[0])
# _r = simulacija_pogon.simulacija(date_time[0], (500, 0), uglovi_matrica[0], snaga[0], y_max)[0]
# _r, _v, _step, min_dist_dest = simulacija_pogon.simulacija(podaci.r0_, podaci.v0_, (300, 0), uglovi, snaga, y_max)
# print("ukupno: ", (time.process_time() - start)/10)
# print("polozaj: ", podaci.trajanje)
