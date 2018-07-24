import podaci
import math


def izracunaj_ka(r):
    return (podaci.sa[0] + podaci.sa[1] / r + (podaci.sa[2] / r) / r) / (1 + podaci.sa[3] * r + podaci.sa[4] * r ** 2)


def izracunaj_ni(r):
    if r < podaci.r_max:
        return izracunaj_ka(r)
    else:
        return izracunaj_ka(podaci.r_max)
    
    
def prel(r):
    if r <= podaci.r_tilt:
        return podaci.alphaP * izracunaj_ni(podaci.r_tilt)
    else:
        return podaci.alphaP * izracunaj_ni(r)


def epsilon(t):
    return podaci.beta[0] + podaci.beta[1] * math.exp(podaci.beta[2] * t) + podaci.beta[3] * t


def p_gen(r, t):
    if podaci.fuel_type == "solar":
        return podaci.P0_solar * epsilon(t) * prel(r)
    elif podaci.fuel_type == "rtg":
        return podaci.P0_rtg*podaci.eta*(0.5**(podaci.halflife/t))
    else:
        raise ValueError


def p_avb(r, t):
    if p_gen(r, t) - podaci.psc <= 0:
        return 0.0
    elif p_gen(r, t) - podaci.psc > podaci.Pmax:
        return podaci.Pmax
    else:
        return p_gen(r, t) - podaci.psc


def thrust(r, t):
    if p_avb(r, t) < podaci.Pmin:
        return 0.0
    else:
        return podaci.ct[0]+podaci.ct[1]*p_avb(r, t)


def flow_rate(r, t):
    if p_avb(r, t) < podaci.Pmin:
        return 0.0
    else:
        return podaci.cm[0] + podaci.cm[1] * p_avb(r, t)
