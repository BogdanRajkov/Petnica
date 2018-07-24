import math
import numpy as np
from numpy import array as u_niz
import matplotlib.pyplot as plt


# napredniji simulator - brod mase m
# ima motor snage f i potrosnju goriva
def simulacija(x, y, vx, vy, gm, time_len, step, f, uglovi, kor_goriva, m):
    segmenti = 10
    duz_segm = 3600 * 24 * 50
    n = int(math.ceil(time_len / step))
    _x = np.empty(n)
    _y = np.empty(n)
    _t = np.empty(n)
    uglovi = u_niz(uglovi) * math.pi / 180
    for i in range(n):
        k = int(i * step) / duz_segm
        if k >= segmenti:
            f = 0.0
            ang = []
        else:
            ang = uglovi[k]
        rez = runge_kuta4(x, y, vx, vy, gm, step, f, m, ang, kor_goriva)
        m = m - kor_goriva * step
        # print(rez[0][0])
        x = rez[0][0]
        y = rez[0][1]
        vx = rez[1][0]
        vy = rez[1][1]
        _x[i] = x
        _y[i] = y
        _t[i] = i * step

    # plt.plot (u_niz(T)/(3600*24),u_niz(X)/1e9,label='x');
    # plt.plot(u_niz(T)/(3600*24),u_niz(Y)/1e9,label='y');
    plt.plot(u_niz(_x) / 1e9, u_niz(_y) / 1e9, 'k', linewidth=0.5)
    # plt.legend()


def izracunaj(gm, r_, v0, k, step, f, ang, gor):
    r_ = r_ + k[1] * step
    r = np.sqrt(r_[0] * r_[0] + r_[1] * r_[1])
    a = -1 * (r_ / r) * gm / (r * r) + (f / (gor[0] - step * gor[1])) * u_niz([math.cos(ang), math.sin(ang)])
    v = v0 + k[0] * step
    # print(r,R,a,v,'End\n')
    # print('a ', a,'shape', np.shape (a),'b ',v,'shape ', np.shape(v));
    return u_niz([a, v])


def runge_kuta4(x, y, vx, vy, gm, step, f, m, ang, kor_goriva):
    r = u_niz([x, y])
    v = u_niz([vx, vy])
    gor = u_niz([m, kor_goriva])
    k1 = izracunaj(gm, r, v, u_niz([[0, 0], [0, 0]]), 0, f, ang, gor)
    k2 = izracunaj(gm, r, v, k1, step / 2, f, ang, gor)
    k3 = izracunaj(gm, r, v, k2, step / 2, f, ang, gor)
    k4 = izracunaj(gm, r, v, k3, step, f, ang, gor)
    k = k1 + 2 * k2 + 2 * k3 + k4
    # print(k[0][0],k[0][0]);
    # print('rk: ',[r+(step/6)*k[1],v+(step/6)*k[0]]);
    return [r + (step / 6) * k[1], v + (step / 6) * k[0]]
