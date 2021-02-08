%matplotlib inline

from pyabc import (ABCSMC,
                   RV, Distribution,
                   MedianEpsilon,
                   LocalTransition)
from pyabc.visualization import plot_kde_2d, plot_data_callback
import matplotlib.pyplot as plt
import os
import tempfile
import numpy as np
import scipy as sp

N = 66650000

IA0, IS0, R0, D0= 46000, 2400, 0, 0

S0 = N - IA0 - IS0

beta1, beta2 = 0.06, 0.003
gamma1, gamma2 = 0.06, 0.03
delta = 0.058
row = 0.0007

t = np.linspace(0, 61, 61)

# The SIR model differential equations.
def deriv(y, t, N, beta1, beta2, gamma1, gamma2, delta, row):
    S, IA, IS, R, D = y
    dSdt = -(beta1*S*IA/N)-(beta1*S*IS/N)
    dIAdt = (beta1*S*IA/N)+(beta1*S*IS/N)-(gamma1*IA)-(delta*IA)
    dISdt = (delta*IA)-(gamma2*IS)-(row*IS)
    dRdt = (gamma1*IA)+(gamma2*IS)
    dDdt = (row*IS)
    return dSdt, dIAdt, dISdt, dRdt, dDdt

y0 = S0, IA0, IS0, R0, D0
ret = odeint(deriv, y0, t, args=(N, beta1, beta2, gamma1, gamma2, delta, row))
S, IA, IS, R, D = ret.T


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, (IA+IS)/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, D/1000, 'b', alpha=0.5, lw=2, label='Dead')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,120)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
