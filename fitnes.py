import numpy as np
import podaci


def fitnes(_dist, _t):
    length = len(_dist)
    _gmaks = _dist/podaci.fitness_max_dist - 1
    # print('gmaks: ', _gmaks)
    _gmin = 1-_dist/podaci.fitness_min_dist
    # print('gmin: ', _gmin)
    nule = np.zeros(length)
    maks = np.maximum(nule, _gmaks)
    mini = np.maximum(nule, _gmin)
    # print((maks,mini))
    return np.amax(_t) + np.multiply(podaci.r_maks, maks) + np.multiply(podaci.r_min, mini)
