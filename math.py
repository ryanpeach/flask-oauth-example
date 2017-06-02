import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import choice, randint
from random import random as rand
from datetime import datetime

N = 6
n_donations = 1
n_users = 10
rand_withdraws=False

# Donations
income = np.zeros(N)
if n_donations == 1:
    i_idx = [0]
else:
    i_idx = choice(np.arange(N),n_donations)
for i in i_idx:
    income[i] = rand()
    
# Derivative Balance
dBi = income
Bi  = np.cumsum(dBi)

def main1(g=lambda x: x/2):
    # Define max offset
    B   = Bi.copy()
    dBo = np.zeros(N)
    dBoM = np.zeros(N)

    for n in range(N):
        if rand_withdraws:
            n_withdraws = randint(0,n_users)
        else:
            n_withdraws = n_users
        withdraw_max = g(B[n])/n_users
        bo = withdraw_max*n_withdraws
        dBoM[n] = withdraw_max
        dBo[n] = bo
        B[n:] -= bo
        
    Bo = np.cumsum(dBo)
    
    B = Bi - Bo
    
    return B, Bi, dBi, Bo, dBo, dBoM
    
def main2():
    """ Signals based method."""
    pass

def plot(B, Bi, dBi, Bo, dBo, dBoM, save=True):
    plt.subplot(4,2,1)
    plt.title("B")
    plt.plot(B)
    plt.subplot(4,2,3)
    plt.title("Bi")
    plt.plot(Bi)
    plt.subplot(4,2,4)
    plt.title("Bo")
    plt.plot(Bo)
    plt.subplot(4,2,5)
    plt.title("dBi")
    plt.plot(dBi)
    plt.subplot(4,2,6)
    plt.title("dBo")
    plt.plot(dBo)
    plt.subplot(4,2,7)
    plt.title("dBoM")
    plt.plot(dBoM)
    if save:
        plt.savefig("./output/math/{}.png".format(datetime.now()))
    
plot(*main1(), save=False)
plot(*main2())