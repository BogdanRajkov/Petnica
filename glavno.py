from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import time
from podaci import au, r0_, v0_, n
import simulacija_pogon
import prvi
# import genetski_algoritam
import podaci
# import drugi
# import math
# import timeit


def crtanje_planeta():
    broj_tacaka = 365 * 5
    broj_planeta = 4
    t = np.linspace(0, 365 * 5, broj_tacaka)
    x = [[] for _ in range(broj_planeta)]
    y = [[] for _ in range(broj_planeta)]
    e_prev = [2.0 for _ in range(broj_planeta)]
    indeksi_planeta = [1, 2, 3, 4]  # indeksi planeta koje crtamo
    for i in range(broj_tacaka):
        for j in range(broj_planeta):  # koliko planeta crtamo
            rez = prvi.polozaj_planeta(indeksi_planeta[j], t[i], e_prev[j])
            x[j].append(rez[0])
            y[j].append(rez[1])
            e_prev[j] = rez[2]

    plt.plot(x[0], y[0], 'g', linewidth=0.8)
    plt.plot(x[1], y[1], 'y', linewidth=0.8)
    plt.plot(x[2], y[2], 'b', linewidth=0.8)
    plt.plot(x[3], y[3], 'r', linewidth=0.8)

    # plt.setp(putanje,markersize=2.5)
    plt.plot(0.0, 0.0, 'k*', markersize=7)
    plt.axis('scaled')
    plt.title('Putanje Merkura, Venere, Zemlje i Marsa oko Sunca')
    plt.xlabel('x koord [astronomska jedinica]')
    plt.ylabel('y koord [astronomska jedinica]')
    plt.show()


def x_osa(a):
    return np.arange(a)


def inicijalizacija(br_jed, br_segm, chebdeg):
    uglovi_matrica = np.multiply(np.random.random_sample((br_jed, br_segm)), 2 * pi)
    print(uglovi_matrica)
    koeficijenti = np.array([chebfit(x_osa(br_segm), _, chebdeg) for _ in uglovi_matrica])
    temp = np.array([chebval(x_osa(br_segm), cheb) for cheb in koeficijenti])
    print(np.divide(temp, uglovi_matrica))
    snaga = np.random.random_integers(0, 1, (br_jed, br_segm))
    return koeficijenti, snaga


def otpakivanje(koeficijenti, snaga):
    koef_bitovi = np.unpackbits(koeficijenti.view(np.uint8), axis=1)
    matrica = np.concatenate((koef_bitovi, snaga), axis=1)
    return matrica


def pakovanje(matrica, chebdeg):
    koef_bitovi = matrica[:, :(chebdeg + 1) * 64]
    snaga = matrica[:, (chebdeg + 1) * 64:]
    koeficijenti = np.packbits(koef_bitovi, axis=1).view(np.float_)
    return koeficijenti, snaga

# def kodiranje(koeficijenti, snaga, n, br_segm):


# uglovi = np.zeros(1000)
uglovi = np.ones(n) * pi
rand_snaga = np.random.random_integers(0, 1, n)
y_max = 1.0

start = time.process_time()
_r, _v, _step = simulacija_pogon.simulacija(r0_, v0_, (3e2, 0), uglovi, rand_snaga, y_max)
# _b = simulacija_pogon.simulacija(r0_, v0_, (4e2, 20), np.zeros(1000), 1000)
# _c = prvi.simulacija(150e9, 0, 0, 29780, podaci.grav_par[0], 1000)
print("ukupno: ", (time.process_time() - start)/10)
print("polozaj: ", podaci.trajanje)
print(_r.shape)
print(_r[730])
# print("polozaji: ", _r)
plt.plot(_r[:, 0]/au, _r[:, 1]/au)
# plt.plot(_b[:, 0], _b[:, 1])
# plt.plot(_c[0], _c[1])
plt.axis('equal')
plt.show()
# crtanje_planeta()


