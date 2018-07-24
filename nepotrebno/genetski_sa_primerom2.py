<<<<<<< HEAD
from numpy import random
import podaci
import time

pop = [[random.random_integers(0, 1, podaci.chromosome_len), random.random_sample()] for i in range(podaci.pop_size)]
# print(len(pop[0][0]))


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
    br_jed = len(populacija)  # elitizam bira procenat najboljih u populaciji i njima menja isti procenat najgorih
    m = int(round(p_elit * br_jed))
    populacija = sorted(populacija, key=kljuc, reverse=True)

    populacija = populacija[:br_jed-m]
    populacija.extend(populacija[:m])
    return populacija


# turnir je vrsenje prirodne selekcije. u svakom paru koji se takmici pobedjuje
def tournament(populacija, lista_parova, br_jed):
                                    #  najbolji. imas objasnjen geneticki algoritam u predlogu projekta.
    pobednici = []
    for i in range(br_jed):
        if populacija[lista_parova[i][0]][1] > populacija[lista_parova[i][1]][1]:
            pobednici.append(populacija[lista_parova[i][0]])
        else:
            pobednici.append(populacija[lista_parova[i][1]])
    
    return pobednici


def uparivanje(lista, br_jed):  # deli elemente liste na parove uzastopnih elem
    return [lista[i:i+2] for i in range(0, br_jed-1, 2)]


def selekcija(populacija):  # vrsi selekciju. promesa listu nasumicno, daje da se na osnovu nje biraju parovi i onda
                            #  radi turnir da odredi najbolje
    br_jed = len(populacija)
    indeksi = [i for i in range(br_jed)]
    random.shuffle(indeksi)
    
    parovi = uparivanje(indeksi, br_jed)
    
    tournament1 = tournament(populacija, parovi, br_jed//2)
    # print(tournament1)
    random.shuffle(indeksi)
    
    parovi = uparivanje(indeksi, br_jed)
    
    tournament2 = tournament(populacija, parovi, br_jed//2)  # turnir se radi dvaput i onda se rezultati turnira spajaju
    # print(tournament2)
    return tournament1 + tournament2


# def genetski_algoritam (populacija,podaci.pop_size,max_gen,gen):


def kros_over(kod1, kod2, br_jed):  # svaki bit novog koda generise izborom bita jednog od roditelja.
    novi_kod = []
    for i in range(br_jed):
        if random.random() < 0.5:
            novi_kod.append(kod1[i])
        else:
            novi_kod.append(kod2[i])
    return novi_kod


def ukrstanje_parova(populacija, parovi, br_jed):    # dvaput ukrsta svaki par da bi populacija dece
                                    #  imala isti broj clanova kao populacija roditelja
    potomstvo = []
    m = podaci.chromosome_len

    for i in range(br_jed):
        potomstvo.append([kros_over(populacija[parovi[i][0]][0], populacija[parovi[i][1]][0], m)])
        potomstvo.append([kros_over(populacija[parovi[i][0]][0], populacija[parovi[i][1]][0], m)])
    return potomstvo


def ukrstanje(populacija):  # ukrsta parove jedinki - svojevrsno razmnozavanje
    br_jed = len(populacija)
    indeksi = [i for i in range(br_jed)]
    random.shuffle(indeksi)
    parovi = uparivanje(indeksi, br_jed)
    return ukrstanje_parova(populacija, parovi, br_jed//2)


def izvrni_bit(string, indeks):  # mutira bitove
    if string[indeks] == 0:
        string[indeks] = 1
    else:
        string[indeks] = 0
    return string


def mutacija(populacija, p_mut):  # mutira odredjen procenat bitova u populaciji
    br_jed = len(populacija)
    m = len(populacija[0][0])
    for i in range(br_jed):
        br_prom = random.binomial(m, p_mut)
        if br_prom != 0:
            indeksi = random.choice(m, br_prom, replace=False)
            for j in indeksi:
                izvrni_bit(populacija[i][0], j)


def genetski_algoritam(populacija, p_elit, p_mut):
    populacija = elitizam(populacija, p_elit)
    populacija = selekcija(populacija)
    # print(len(populacija[0][0]))
    populacija = ukrstanje(populacija)
    mutacija(populacija, p_mut)
    return populacija


t = time.process_time()
genetski_algoritam(pop, 0.05, 0.03)
print(time.process_time() - t)
=======
from numpy import random
import podaci
import time

pop = [[random.random_integers(0, 1, podaci.chromosome_len), random.random_sample()] for i in range(podaci.pop_size)]
# print(len(pop[0][0]))


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
    br_jed = len(populacija)  # elitizam bira procenat najboljih u populaciji i njima menja isti procenat najgorih
    m = int(round(p_elit * br_jed))
    populacija = sorted(populacija, key=kljuc, reverse=True)

    populacija = populacija[:br_jed-m]
    populacija.extend(populacija[:m])
    return populacija


# turnir je vrsenje prirodne selekcije. u svakom paru koji se takmici pobedjuje
def tournament(populacija, lista_parova, br_jed):
                                    #  najbolji. imas objasnjen geneticki algoritam u predlogu projekta.
    pobednici = []
    for i in range(br_jed):
        if populacija[lista_parova[i][0]][1] > populacija[lista_parova[i][1]][1]:
            pobednici.append(populacija[lista_parova[i][0]])
        else:
            pobednici.append(populacija[lista_parova[i][1]])
    
    return pobednici


def uparivanje(lista, br_jed):  # deli elemente liste na parove uzastopnih elem
    return [lista[i:i+2] for i in range(0, br_jed-1, 2)]


def selekcija(populacija):  # vrsi selekciju. promesa listu nasumicno, daje da se na osnovu nje biraju parovi i onda
                            #  radi turnir da odredi najbolje
    br_jed = len(populacija)
    indeksi = [i for i in range(br_jed)]
    random.shuffle(indeksi)
    
    parovi = uparivanje(indeksi, br_jed)
    
    tournament1 = tournament(populacija, parovi, br_jed//2)
    # print(tournament1)
    random.shuffle(indeksi)
    
    parovi = uparivanje(indeksi, br_jed)
    
    tournament2 = tournament(populacija, parovi, br_jed//2)  # turnir se radi dvaput i onda se rezultati turnira spajaju
    # print(tournament2)
    return tournament1 + tournament2


# def genetski_algoritam (populacija,podaci.pop_size,max_gen,gen):


def kros_over(kod1, kod2, br_jed):  # svaki bit novog koda generise izborom bita jednog od roditelja.
    novi_kod = []
    for i in range(br_jed):
        if random.random() < 0.5:
            novi_kod.append(kod1[i])
        else:
            novi_kod.append(kod2[i])
    return novi_kod


def ukrstanje_parova(populacija, parovi, br_jed):    # dvaput ukrsta svaki par da bi populacija dece
                                    #  imala isti broj clanova kao populacija roditelja
    potomstvo = []
    m = podaci.chromosome_len

    for i in range(br_jed):
        potomstvo.append([kros_over(populacija[parovi[i][0]][0], populacija[parovi[i][1]][0], m)])
        potomstvo.append([kros_over(populacija[parovi[i][0]][0], populacija[parovi[i][1]][0], m)])
    return potomstvo


def ukrstanje(populacija):  # ukrsta parove jedinki - svojevrsno razmnozavanje
    br_jed = len(populacija)
    indeksi = [i for i in range(br_jed)]
    random.shuffle(indeksi)
    parovi = uparivanje(indeksi, br_jed)
    return ukrstanje_parova(populacija, parovi, br_jed//2)


def izvrni_bit(string, indeks):  # mutira bitove
    if string[indeks] == 0:
        string[indeks] = 1
    else:
        string[indeks] = 0
    return string


def mutacija(populacija, p_mut):  # mutira odredjen procenat bitova u populaciji
    br_jed = len(populacija)
    m = len(populacija[0][0])
    for i in range(br_jed):
        br_prom = random.binomial(m, p_mut)
        if br_prom != 0:
            indeksi = random.choice(m, br_prom, replace=False)
            for j in indeksi:
                izvrni_bit(populacija[i][0], j)


def genetski_algoritam(populacija, p_elit, p_mut):
    populacija = elitizam(populacija, p_elit)
    populacija = selekcija(populacija)
    # print(len(populacija[0][0]))
    populacija = ukrstanje(populacija)
    mutacija(populacija, p_mut)
    return populacija


t = time.process_time()
genetski_algoritam(pop, 0.05, 0.03)
print(time.process_time() - t)
>>>>>>> 1f3444ba8118807f6cea2344c8856061a87c88b4
