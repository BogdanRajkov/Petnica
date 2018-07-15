from simulacija_pogon import polozaj_planeta
from podaci import beg_of_time
import datetime as dt
import numpy as np
import pickle
import math

planets_position = np.empty((math.floor(10*365.25), 8, 2))
date = dt.date(2015, 1, 1)

for i in range(planets_position.shape[0]):
    for j in range(8):
        planets_position[i][j] = polozaj_planeta(j, (date - beg_of_time).total_seconds(), matrica=False)[:2]
    date = date + dt.timedelta(days=1)

f = open('polozaji_planeta.txt', 'w+b')
pickle.dump(planets_position, f)
