import numpy as np
from scipy import interpolate
import matplotlib.pyplot as  plt

mass  = np.array([4943.816242314127, 4202.225127957172,3184.7727524808392, 2002.254765493346,816.4407179654136])
v_inf = np.array([1,1154665925308125e3, 2.426358385073982e3,3.740628764711334e3,5.257614369999692e3,7.176651839495877e3])

f  = interpolate.inter1d(mass,v_inf)
xnew = np.arange(0, 5000, 50)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()

