import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d

v_inf = np.array([0.0024449877750618576e3, 0.3023634881825612e3, 0.8109209453952739e3, 1.410757946210269e3,
                  2.2192339038304825e3, 3.125509372453138e3, 3.4319478402607992e3, 3.9405052974735137e3,
                  4.546862265688672e3, 5.2510187449062755e3, 5.753056234718828e3, 6.261613691931547e3,
                  6.561532192339039e3, 7.167889160554198e3, 7.572127139364296e3, 8.080684596577017e3,
                  8.582722086389568e3, 9.091279543602283e3, 9.391198044009784e3])
mass = np.array([5158.301158301159, 5140.926640926642, 5042.471042471044, 4810.810810810813, 4341.6988416988415,
                 3669.8841698841707, 3432.432432432434, 3021.2355212355224, 2540.5405405405413, 2001.930501930503,
                 1648.6486486486492, 1324.324324324325, 1144.7876447876451, 820.463320463321, 635.1351351351404,
                 420.8494208494212, 241.31274131274313, 84.94208494208488, 3.861003861004974]) - 1200
atlas_v_inf = np.sqrt(np.array([4.842484248424846e6, 9.86738673867388e6, 14.891539153915396e6,
                                19.915541554155418e6, 24.938793879387944e6, 29.96159615961598e6, 34.91779177917793e6,
                                40.00570057005703e6, 44.96099609960997e6, 49.981698169816994e6, 60.02175217521753e6]))
atlas_mass = np.array([5563.636363636364, 5054.545454545455, 4590.909090909091, 4136.363636363636,
                       3727.272727272727, 3345.454545454544, 2999.999999999999, 2672.727272727273, 2381.818181818181,
                       2127.272727272727, 1700.0])
# f = interp1d(mass, v_inf, kind='cubic', fill_value="extrapolate")
# xnew = np.arange(0, 5800, 100)
# ynew = f(xnew)  # use interpolation function returned by `interp1d`
# x = np.arange(0, 6500, 10)
# poly = np.poly1d(np.polyfit(v_inf, mass, 4))
# y = poly(x)
# plt.plot(x, y, '-')
# plt.axis.ymin = 0.0
# plt.show()
#a = np.arange(0,4000,100)
#poly = np.poly1d(np.polyfit(atlas_mass, atlas_v_inf, 4))
#poly2 = np.poly1d(np.polyfit(mass, v_inf, 4))
#b = poly(a)
#c = poly2(a)
#plt.plot(b,a,'r-.',c,a,'b:')
#plt.xlabel('brzina')
#plt.ylabel('masa')
#plt.show()
def numpy_atlas_interpol (masa_broda):
    poly = np.poly1d(np.polyfit(atlas_mass, atlas_v_inf, 4))
    brzina = poly(masa_broda)
    if brzina > 0:
        return brzina
    else:
        raise ValueError('v_inf je negativna!')
def numpy_interpol(masa_broda):
    # plt.plot(mass, v_inf)
    # plt.show()
    poly = np.poly1d(np.polyfit(mass, v_inf, 4))
    brzina = poly(masa_broda)
    if brzina > 0:
        return brzina
    else:
        raise ValueError('v_inf je negativna!')


if __name__ == '__main__':
    plt.plot(mass, v_inf)
    plt.title('ARIANE 5')
    plt.show()

