import numpy as np

# promenljive

fuel_type = "solar"
r0_ = np.array((150e9, 0))
v0_ = np.array((0, 29780))
n = 20
indeksi = np.array([0, 1, 2, 3, 4, 5, 6])
# astronomska jedinica -> metri
au = 149597870700

# gravitacioni parametri
grav_par = [1.327124400189e20, 2.20329e13, 3.248599e14, 3.9860044188e14, 4.90486959e12, 4.2828372e13, 6.26325e10,
            1.266865349e17, 3.79311879e16, 5.7939399e15, 6.8365299e15, 8.719e11]

# podaci potrebni za racunanje pozicije planete po formulama. imas formule u jednom pdf - u na drajvu.
# redosled podataka je:
# masa - u jedinicama
# a - velika poluosa i njena promena po veku
# e - ekscentricitet i promena po veku
# L - srednja longituda i promena po veku
# malo teta - longtituda perihela i promena po veku
# Teta - longtitude of ascending node i promena po veku
info = [(),
        (0.38709927,   0.00000037, 0.20563593,  0.00001906, 252.25032350, 149472.67411175,  77.45779628),
        (0.72333566,   0.00000390, 0.00677672, -0.00004107, 181.97909950,  58517.81538729, 131.60246718),
        (1.00000261,   0.00000562, 0.01671123, -0.00004392, 100.46457166,  35999.37244981, 102.93768193),
        (1.52371034,   0.00001847, 0.09339410,  0.00007882,  -4.55343205,  19140.30268499, -23.94362959),
        (5.20288700,  -0.00011607, 0.04838624, -0.00013253,  34.39644051,   3034.74612775,  14.72847983),
        (9.53667594,  -0.00125060, 0.05386179, -0.00050991,  49.95424423,   1222.49362201,  92.59887831),
        (19.18916464, -0.00196176, 0.04725744, -0.00004397, 313.23810451,    428.48202785, 170.95427630),
        (30.06992276,  0.00026291, 0.00859048,  0.00005105, -55.12002969,    218.45945325,  44.96476227),
        (39.48211675, -0.00031596, 0.24882730,  0.00005170, 238.92903833,    145.20780515, 224.06891629)]

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

# # # # # # # # # # # # #

# RTG

eta = 0.068
halflife = 2765707200
P0_rtg = 4400

# # # # # # # # # # # # #

# GENETSKI ALGORITAM

min_dist = 6e6  # za sad nasumicne vrednosti
max_dist = 100e6  # za sad nasumicne vrednosti
rmax = 20  # za sad nasumicne vrednosti
rmin = 30  # za sad nasumicne vrednosti

# # # # # # # # # # # # #
