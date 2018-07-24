from simulacija_pogon import polozaj_planeta
from podaci import beg_of_time
import datetime as dt
import numpy as np
import math

f = open('polozaji_planeta.txt', 'w+b')
date = dt.datetime(2019, 1, 1, 0, 0, 0)
np.save('datum', [date])
years = 4
np.save('years', years)
step = dt.timedelta(hours=1)
np.save('step', [step])
planets_position = np.empty((math.floor(years*365.25*24), 8, 2))

for i in range(planets_position.shape[0]):
    for j in range(8):
        planets_position[i][j] = polozaj_planeta(j, (date - beg_of_time).total_seconds(), matrica=False)[:2]
    date = date + dt.timedelta(hours=1)

np.save('polozaji_planeta', planets_position)
