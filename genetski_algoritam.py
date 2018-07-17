from numpy import random
import numpy as np
import time

import podaci


def fitnes(min_dist):
    _dist, _t = np.split(min_dist, 2, axis=1)
    length = len(_dist)
    _gmaks = _dist/podaci.fitness_max_dist - 1
    _gmin = 1-_dist/podaci.fitness_min_dist
    nule = np.zeros(length)
    maks = np.maximum(nule, _gmaks)
    mini = np.maximum(nule, _gmin)
    return np.amax(_t) + np.multiply(podaci.r_maks, maks) + np.multiply(podaci.r_min, mini)


def elitizam(populacija, p_elit):  # populacija je lista koja cuva podatke za sve jedinke u populaciji.
    #  svaka jedinka ima binarni kod (predstavljen nizom 0 i 1) i
    # vrednost fitnes funkcije
    br_jed = len(populacija)  # elitizam bira procenat najboljih u populaciji i njima menja isti procenat najgorih
    m = int(round(p_elit * br_jed))
    populacija = np.array(sorted(populacija, key=lambda x: x[1], reverse=True))

    populacija = populacija[:br_jed - m]
    populacija = np.concatenate((populacija, populacija[:m]))
    return populacija


# turnir je vrsenje prirodne selekcije. u svakom paru koji se takmici pobedjuje
def tournament(populacija, fitness, lista_parova, br_jed):
    #  najbolji. imas objasnjen geneticki algoritam u predlogu projekta.
    pobednici = []
    for i in range(br_jed):
        if fitness[lista_parova[i][0]] > fitness[lista_parova[i][1]]:
            pobednici.append(populacija[lista_parova[i][0]])
        else:
            pobednici.append(populacija[lista_parova[i][1]])

    return np.array(pobednici)


# vrsi selekciju. promesa listu nasumicno, daje da se na osnovu nje biraju parovi i onda
#  radi turnir da odredi najbolje
def selekcija(populacija, fitness):
    br_jed = len(populacija)
    indeksi = np.arange(br_jed)

    random.shuffle(indeksi)
    parovi = np.split(indeksi, np.arange(br_jed//2)[1:]*2)

    tournament1 = tournament(populacija, fitness, parovi, br_jed // 2)
    # print(tournament1)

    random.shuffle(indeksi)
    parovi = np.split(indeksi, np.arange(br_jed//2)[1:]*2)

    tournament2 = tournament(populacija, fitness, parovi, br_jed // 2)
    # turnir se radi dvaput i onda se rezultati turnira spajaju

    return np.concatenate((tournament1, tournament2))


# def genetski_algoritam (populacija,podaci.pop_size,max_gen,gen):


def kros_over(kod1, kod2, br_jed):  # svaki bit novog koda generise izborom bita jednog od roditelja.
    novi_kod = []
    for i in range(br_jed):
        if random.random() < 0.5:
            novi_kod.append(kod1[i])
        else:
            novi_kod.append(kod2[i])
    return novi_kod


def ukrstanje_parova(populacija, parovi, br_jed):  # dvaput ukrsta svaki par da bi populacija dece
    #  imala isti broj clanova kao populacija roditelja
    potomstvo = []
    m = len(populacija[0])

    for i in range(br_jed):
        potomstvo.append(kros_over(populacija[parovi[i][0]], populacija[parovi[i][1]], m))
        potomstvo.append(kros_over(populacija[parovi[i][0]], populacija[parovi[i][1]], m))
    return np.array(potomstvo)


def ukrstanje(populacija):  # ukrsta parove jedinki - svojevrsno razmnozavanje
    br_jed = len(populacija)
    indeksi = np.arange(podaci.broj_jedinki)
    random.shuffle(indeksi)
    parovi = np.split(indeksi, 2*np.arange(br_jed//2)[1:])
    return ukrstanje_parova(populacija, parovi, br_jed // 2)


def izvrni_bit(string, indeks):  # mutira bitove
    if string[indeks] == 0:
        string[indeks] = 1
    else:
        string[indeks] = 0
    return string


def mutacija(populacija, p_mut):  # mutira odredjen procenat bitova u populaciji
    br_jed = len(populacija)
    m = len(populacija[0])
    for i in range(br_jed):
        br_prom = random.binomial(m, p_mut)
        if br_prom != 0:
            indeksi = random.choice(m, br_prom, replace=False)
            for j in indeksi:
                populacija[i] = izvrni_bit(populacija[i], j)
    return populacija


def genetski_algoritam(populacija, p_elit, p_mut):
    populacija, fitness = np.split(populacija, [-1], axis=1)
    populacija = elitizam(populacija, p_elit)
    populacija = selekcija(populacija, fitness)
    populacija = ukrstanje(populacija)
    populacija = mutacija(populacija, p_mut)
    # print(populacija.shape)
    return np.concatenate((populacija, fitness), axis=1)


if __name__ == "__main__":
    pop = [[random.random_integers(0, 1, podaci.chromosome_len),
            random.random_sample()] for i in range(podaci.pop_size)]
    t = time.process_time()
    genetski_algoritam(pop, 0.05, 0.03)
    print(time.process_time() - t)