
import numpy as np
from numpy import array as uNiz
import matplotlib.pyplot as plt
from numpy import random
import scipy
import time

n = 150
chromsome_len = 270
populacija = [[random.random_integetrs(0,1,chromsome_len),random.random_integetrs()] for i in range(n)]

def to_binary(niz_num):
    duz = len(niz_num)
    kod = []
    for j in range(duz):
            bitan_deo = [int(digit) for digit in bin(niz_num[j])[2:]]
            bitan_deo = [0 for _ in range(16 - len(bitan_deo))] + bitan_deo
            kod = kod + bitan_deo
    return kod

# from bitstring import BitArray
# >>> bitlist=[1,0,0,0,0,0,0,0]
# >>> b = BitArray(bitlist)
# >>> b.uint


def to_uint16(nizbin):  # ova funkcija prevodi niz 1 i 0 u nizniz dekadnih brojeva. inverzna funkcija prosloj
    duz = len(nizbin)//16
    # print(duz)
    niz_num = []
    for j in range(duz):
        niz = nizbin[16*j:16*j+16]
        niz.reverse()
        suma = 0
        stepen = 1
        for i in range(16):
            suma = suma + stepen * niz[i]
            stepen = stepen << 1
        niz_num.append(suma)
    return niz_num

# print(toBinary(0.15625))
# gornje dve funkcije su potrebne jer se 'genetski kod' putanja zapisuje u obliku dugackog binarnog broja,
# koje je u stvari spojeni niz zapisa uglova pod kojima radi motor


def kljuc(jedinka):
    return jedinka[1]  # kljuc za poredjenje pri sortiranju jedinki


def elitizam(populacija, p_elit):   # populacija je lista koja cuva podatke za sve jedinke u populaciji.
                                    #  svaka jedinka ima binarni kod (predstavljen nizom 0 i 1) i
                                    # vrednost fitnes funkcije
    n = len(populacija)  # elitizam bira procenat najboljih u populaciji i njima menja isti procenat najgorih
    m = int(round(p_elit * n))
    populacija = sorted(populacija, key=kljuc, reverse=True)
    populacija = populacija[:n-m]
    populacija.extend(populacija[:m])
    return populacija


def tournament(populacija,lista_parova, n):    # turnir je vrsenje prirodne selekcije. u svakom paru koji se takmici pobedjuje
                                    #  najbolji. imas objasnjen geneticki algoritam u predlogu projekta.
    pobednici = []
    for i in range(n):
        if populacija[lista_parova[i][0][1] > populacija[lista_parova[i][1]][1]:
            pobednici.append(populacija[lista_parova[i][0]][0])
        else:
            pobednici.append(populacija[lista_parova[i][1]][0])
    
    return pobednici


def uparivanje(lista, n):  # deli elemente liste na parove uzastopnih elem
    return [lista[i:i+2] for i in range(0, n-1, 2)]


def selekcija(populacija):  # vrsi selekciju. promesa listu nasumicno, daje da se na osnovu nje biraju parovi i onda
                            #  radi turnir da odredi najbolje
    n = len(populacija)
    indeksi = [i for i in range(n)]
    random.shuffle(indeksi)
    
    parovi = uparivanje(indeksi, n)
    
    tournament1 = tournament(populacija, parovi, n//2)
    # print(tournament1)
    random.shuffle(indeksi)
    
    parovi = uparivanje(indeksi, n)
    
    tournament2 = tournament(populacija,parovi, n//2)  # turnir se radi dvaput i onda se rezultati turnira spajaju
    # print(tournament2)
    return tournament1 + tournament2


# def genetski_algoritam (populacija,n,max_gen,gen):


def kros_over(kod1, kod2, n):  # svaki bit novog koda generise izborom bita jednog od roditelja.
    novi_kod = []
    for i in range(n):
        if random.random() < 0.5:
            novi_kod.append(kod1[i])
        else:
            novi_kod.append(kod2[i])
    return novi_kod


def ukrstanje_parova(populacija,parovi, n):    # dvaput ukrsta svaki par da bi populacija dece
                                    #  imala isti broj clanova kao populacija roditelja
    potomstvo = []
    m = len(populacija[parovi[0][0]][0])
    for i in range(n):
        potomstvo.append([kros_over(populacija[parovi[i][0]][0], populacija[parovi[i][1]][0], m)])
        potomstvo.append([kros_over(populacija[parovi[i][0]][0], populacija[parovi[i][1]][0], m)])
    return potomstvo


def ukrstanje(populacija):  # ukrsta parove jedinki - svojevrsno razmnozavanje
    n = len(populacija)
    indeksi = [i for i in range(n)]
    random.shuffle(indeksi)
    parovi = uparivanje(indeksi, n)
    return ukrstanje_parova(populacija, parovi, n//2)


def izvrni_bit(string, indeks):  # mutira bitove
    if string[indeks] == 0:
        string[indeks] = 1
    else:
        string[indeks] = 0
    return string


def mutacija(populacija, p_mut):  # mutira odredjen procenat bitova u populaciji
    n = len(populacija)
    m = len(populacija[0][0])
    for i in range(n):
        br_prom = random.binomial(m, p_mut)
        if br_prom != 0:
            indeksi = random.sample(list(range(m)), br_prom)
            for j in indeksi:
                izvrni_bit(populacija[i][0], j)


def genetski_algoritam(populacija, n, p_elit, p_mut):
    populacija = elitizam(populacija, p_elit)
    populacija = selekcija(populacija)
    populacija = ukrstanje(populacija)
    mutacija(populacija, p_mut)
    return populacija
