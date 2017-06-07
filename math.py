import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
from numpy.random import choice, randint
from random import random as rand
from datetime import datetime

def simulate_withdraw(n_users, withdraw_max, withdraw_p, rand_amounts = True):
    # Calculate number of withdraws
    n_withdraws = sp.stats.binom.rvs(n_users, withdraw_p, loc=0, size=1, random_state=None)

    # Calculate amounts
    if rand_amounts:
        dBo = np.sum(np.random.random(n_withdraws))
    else:
        dBo = withdraw_max*n_withdraws
        
    return dBo, withdraw_max*n_users - dBo
    
def simulate_donation(mu, a, b):
    # Calculate number of donations
    n_withdraws = sp.stats.poisson.rvs(n_users, mu, loc=0, size=1, random_state=None)

    # Calculate intensity of withdraws
    return np.sum(sp.stats.gamma.rvs(a, b, loc=0, size=n_withdraws, random_state=None))

def main1(M, mu, a, b, withdraw_p, n_users):
    # Define max offset
    dBi      = []
    dBo      = []
    leftover = []
    dBoM     = []
    B = 0
    for T in range(1, M+1):
        # Simulate donation
        dBi0 = simulate_donation(mu, a, b)
        dBi.append(dBi0)
        B += dBi0
        
        # Get withdraw Max
        withdraw_max = B/2./n_users
        dBoM.append(withdraw_max)
        
        # Simulate Withdraw
        dBo0, leftover0 = simulate_withdraw(n_users, withdraw_max, withdraw_p)
        dBo.append(dBo0)
        B -= dBo0
        leftover.append(leftover0)
        
    # Return necessary components
    Bo = np.cumsum(dBo)
    Bi = np.cumsum(dBi)
    B = Bi - Bo
    return B, Bi, dBi, Bo, dBo, dBoM
    
def main2(M, mu, a, b, withdraw_p, n_users):
    """ Signals based method."""
    # Type checking
    assert mu > 0, "mu must be greater than 0. (mu={})".format(mu)
    from scipy.stats import poisson
    
    dBi      = []
    dBo      = []
    leftover = []
    dBoM     = []
    
    for T in range(1,M+1):
        # Donate
        dBi.append(simulate_donation(mu, a, b))
        
        # Withdraws
        ## Get max withdraw
        ### Set up indexing
        k = np.arange(-T, T+1)
        zidx = T               # The index of the 0th value
        assert k[zidx] == 0, "Test Error: 0 index not 0."
        
        ### Set up poisson distribution
        g = poisson.pmf(k, mu)
        
        ### Set up transaction data
        ft = np.zeros(len(k))
        ft[zidx:] = (np.asarray(dBi)+np.asarray(leftover+[0]))[:T+1]
        f = ft[T-k]
        
        ### Get the max withdraw total
        withdraw_max = np.sum(f[-k]*g[k]) / n_users
        dBoM.append(withdraw_max)
        
        # Simulate Withdraw
        dBo0, leftover0 = simulate_withdraw(n_users, withdraw_max, withdraw_p)
        dBo.append(dBo0)
        leftover.append(leftover0)
    
    # Return necessary components
    Bo = np.cumsum(dBo)
    Bi = np.cumsum(dBi)
    B = Bi - Bo
    return B, Bi, dBi, Bo, dBo, dBoM

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
        
if __name__=="__main__":
    M = 6
    withdraw_p = .1
    mu, a, b = 1., 1., 1.
    n_users = 10
    
    plot(*main1(M, mu, a, b, withdraw_p, n_users), save=False)
    plot(*main2(M, mu, a, b, withdraw_p, n_users))