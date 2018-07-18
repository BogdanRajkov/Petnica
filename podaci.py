import numpy as np
import datetime
import math

# from jplephem.spk import SPK


# PROMENLJIVE KOJE SE MOGU VARIRATI

fuel_type = "solar"
dry_mass = 2400
max_fuel_mass = 600
max_time_span = 1000
broj_segmenata = 20
broj_jedinki = 20
broj_gen = 1
y_max = 2.5
chebdeg = 4
destination = np.array([4])


# POMOCNE PROMENLJIVE - NE MENJATI AKO NE ZNAS STA JE (ODNOSI SE NA SVE ISPOD)

# kernel = SPK.open('D:/Downloads/de432s.bsp')
beg_of_time = datetime.date(2000, 1, 1)  # pocetak vremena
planet_radii = np.array([6.96342e8, 2.4397e6, 6.0518e6, 6.371e6, 3.3895e6, 6.9911e7, 5.8232e7, 2.5362e7, 2.4622e7])
r_parking = 5e5 + planet_radii[3]  # poluprecnik "parking" orbite - u njoj se nalazi brod pre lansiranja
indeksi = np.array([0, 2, 3, 4, 5, 6, 7])  # planete cija se gravitacija uracunava
au = 149597870700  # astronomska jedinica u metrima
grav_par = [1.327124400189e20, 2.20329e13, 3.248599e14, 3.9860044188e14, 4.2828372e13,  # gravitaconi parametri
            1.266865349e17, 3.79311879e16, 5.7939399e15, 6.8365299e15, 8.719e11]
trajanje = 0.0
crashed = False

polozaji_ucitani = False
beg_date, years_saved = 0, 0
polozaji_matrica = np.empty((math.floor(10*365.25), 8, 2))

# podaci potrebni za racunanje pozicije planete po formulama

# redosled podataka je:
# masa - u jedinicama
# a - velika poluosa i njena promena po veku
# e - ekscentricitet i promena po veku
# L - srednja longituda i promena po veku
# malo teta - longtituda perihela i promena po veku
# Teta - longtitude of ascending node i promena po veku
info = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0.38709927,   0.00000037, 0.20563593,  0.00001906,
                  252.25032350, 149472.67411175,  77.45779628,  0.16047689],
                 [0.72333566,   0.00000390, 0.00677672, -0.00004107,
                  181.97909950,  58517.81538729, 131.60246718,  0.00268329],
                 [1.00000261,   0.00000562, 0.01671123, -0.00004392,
                  100.46457166,  35999.37244981, 102.93768193,  0.32327364],
                 [1.52371034,   0.00001847, 0.09339410,  0.00007882,
                  -4.55343205,   19140.30268499, -23.94362959,  0.44441088],
                 [5.20288700,  -0.00011607, 0.04838624, -0.00013253,
                  34.39644051,    3034.74612775,  14.72847983,  0.21252668],
                 [9.53667594,  -0.00125060, 0.05386179, -0.00050991,
                  49.95424423,    1222.49362201,  92.59887831, -0.41897216],
                 [19.18916464, -0.00196176, 0.04725744, -0.00004397,
                  313.23810451,    428.48202785, 170.95427630,  0.40805281],
                 [30.06992276,  0.00026291, 0.00859048,  0.00005105,
                  -55.12002969,    218.45945325,  44.96476227, -0.32241464],
                 [39.48211675, -0.00031596, 0.24882730,  0.00005170,
                  238.92903833,    145.20780515, 224.06891629, -0.04062942]], dtype=np.float64)


# PODACI O SOLARNOM POGONU

alphaP = 1.0*au
P0_solar = 1000
r_max = 5 * au
r_tilt = 0.7 * au
cm = np.array([475.56e-9, 0.90209e-9, .0, .0, .0])
ct = np.array([-1.9137e-3, 0.036242e-3, .0, .0, .0])
sa = np.array([1.1063, 149.5e-3 * au, -299e-3 * au ** 2, -43.2e-3 / au, 0.0])
beta = np.array([1.0, 0.0, 0.0, 0.0])

psc = 150  # primer vrednosti
Pmin = 649
Pmax = 2600


# RTG

eta = 0.068
halflife = 2765707200
P0_rtg = 4400


# GENETSKI ALGORITAM - 99% ce nam trebati samo p_elit i p_mut
pass
# ga_min_dist = 6e6  # za sad nasumicne vrednosti
# ga_max_dist = 100e6  # za sad nasumicne vrednosti
# rmax = 20  # za sad nasumicne vrednosti
# rmin = 30  # za sad nasumicne vrednosti
# pop_size = 150
# chromosome_len = 600
p_elit = 0.07
p_mut = 0.01


# FITNES FUNKCIJA

fitness_min_dist = np.array([6e6])
fitness_max_dist = np.array([1e8])
r_maks = np.array([10.0])
r_min = np.array([20])
