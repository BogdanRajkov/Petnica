from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import warnings
import math

from podaci import broj_segmenata, broj_jedinki, y_max, max_time_span  # , au
import simulacija_pogon
import podaci
# import datetime
# import timeit
# import time
# import math


def x_osa(a):
    return np.arange(a)


def crtanje_planeta(plot=True):
    broj_tacaka = int(y_max*365*2)
    broj_planeta = 5
    t = np.linspace(0, y_max*365*24*3600, broj_tacaka)
    x = [[] for _ in range(broj_planeta)]
    y = [[] for _ in range(broj_planeta)]
    indeksi_planeta = [1, 2, 3, 4, 5, 6, 7, 8]  # indeksi planeta koje crtamo
    for i in range(broj_tacaka):
        for j in range(broj_planeta):  # koliko planeta crtamo
            rez = simulacija_pogon.polozaj_planeta(indeksi_planeta[j], t[i], matrica=True)
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
    len_days = len(days)
    days_bit = np.unpackbits(days.view(np.uint8).reshape(len_days, 2), axis=1)
    fuel_bit = np.unpackbits(fuel_mass).reshape(len(fuel_mass), 8)
    koef_bit = np.unpackbits(koeficijenti.view(np.uint8), axis=1)
    return np.concatenate((days_bit, fuel_bit, koef_bit, snaga, fit[:, np.newaxis]), axis=1)


def bit_to_float(matrica_bit):
    warnings.filterwarnings("error", category=RuntimeWarning)
    days_bit, fuel_bit, koeficijenti_bit, snaga = np.split(matrica_bit,
                                                           np.array((16, 24, (1+podaci.chebdeg)*64 + 24)),
                                                           axis=1)
    days = np.packbits(days_bit.astype(bool), axis=1).view(np.uint16)
    brod = np.packbits(fuel_bit.astype(bool), axis=1).view(np.uint8)
    koeficijenti = np.packbits(koeficijenti_bit.astype(bool), axis=1).view(np.float64)
    try:
        uglovi = chebval(x_osa(broj_segmenata), koeficijenti.T)
        uglovi = np.mod(uglovi, 2*pi)
    except RuntimeWarning:
        uglovi = np.random.random_sample(size=(broj_jedinki, broj_segmenata)) * 2*pi
    print(days.shape, brod.shape, uglovi.shape, snaga.shape)
    return days[:, 0], brod[:, 0], uglovi, snaga