import numpy as np
#nasumicne vrednosti
max_dist = np.array([1e8])
min_dist = np.array([6e6])
r_maks = np.array([10.0])
r_min = np.array([20])
def fitnes (_dist,_t):
    l = len(_dist)
    _gmaks = _dist/max_dist - 1
    #print('gmaks: ', _gmaks)
    _gmin = 1-_dist/min_dist
    #print('gmin: ', _gmin)
    nule = np.zeros(l)
    maks = np.maximum(nule,_gmaks)
    mini = np.maximum(nule,_gmin)
    #print((maks,mini))
    return np.amax(_t) + np.multiply(r_maks,maks) + np.multiply(r_min,mini)


