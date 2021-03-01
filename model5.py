from pyabc import (ABCSMC,
                   RV, Distribution,
                   MedianEpsilon,
                   LocalTransition,
                   StochasticAcceptor,
                   IndependentNormalKernel,
                   Temperature)
from pyabc.visualization import plot_kde_2d, plot_data_callback
import matplotlib.pyplot as plt
import os
import tempfile
import numpy as np
import scipy as sp


db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))

S0 = 66500000
IA0 = 46000
IS0 = 2400

total_pop = S0 + IA0 + IS0

measurement_data = np.array([646, 603, 568, 398, 555, 708, 603, 703, 717, 564, 446, 370, 732, 691, 774, 705, 580, 498, 444, 813, 782, 798, 799, 767, 523, 558, 858, 843, 1038, 992, 675, 542, 544, 1040, 1018,1059, 1083, 943, 695, 616, 1473, 1339, 1207, 1146, 1161])
measurement_data = measurement_data/total_pop
measurement_times = np.arange(len(measurement_data))

init = np.array([S0, IA0, IS0, 0, 0, 0])
init = init/total_pop

def f(y, t0, beta1, beta2, gamma1, gamma2, delta, echelon, rho, theta1, theta2):
    S, IA, IS, R, D, C = y
    dS = - (S*beta1*IA) - (S*beta2*IS) - (rho*S)
    dIA = (S*beta1*IA) + (S*beta2*IS) - (gamma1*IA) - (delta*IA)
    dIS = (delta*IA) - (gamma2*IS) - (echelon*IS)
    dR = (gamma1*IA) + (gamma2*IS)
    dD = (echelon*IS) + (rho*S)
    dC = (theta1*IA) + (theta2*IS)
    return dS, dIA, dIS, dR, dD, dC

def model(pars):
    sol = sp.integrate.odeint(
             f, init, measurement_times,
             args=(pars["beta1"], pars["beta2"], pars["gamma1"], pars["gamma2"], pars["delta"], pars["echelon"], pars["rho"], pars["theta1"], pars["theta2"]))
    return {"X_2": sol[:,5]}

sigma = 0.02
acceptor = StochasticAcceptor()
kernel = IndependentNormalKernel(var=sigma**2)
eps = Temperature()

parameter_prior = Distribution(
                beta1=RV("uniform", 0, 1),
				beta2=RV("uniform", 0, 1),
				gamma1=RV("uniform", 0, 1),
				gamma2=RV("uniform", 0, 1),
				delta=RV("uniform", 0, 1),
				echelon=RV("uniform", 0, 1),
				rho=RV("uniform", 0, 1),
				theta1=RV("uniform", 0, 1),
                theta2=RV("uniform", 0, 1))
parameter_prior.get_parameter_names()

abc = ABCSMC(models=model,
             parameter_priors=parameter_prior,
             distance_function=kernel,
             acceptor = acceptor,
             eps = eps,
             population_size=100000)

abc.new(db_path, {"X_2": measurement_data});

h = abc.run(minimum_epsilon=0.1, max_nr_populations=5)

fig = plt.figure(figsize=(10,8))
for t in range(h.max_t+1):
    ax = fig.add_subplot(3, np.ceil(h.max_t / 3), t+1)

    ax = plot_kde_2d(
        *h.get_distribution(m=0, t=t), "theta1", "theta2",
        xmin=0, xmax=1, numx=200, ymin=0, ymax=1, numy=200, ax=ax)
    ax.set_title("Posterior t={}".format(t))

fig.tight_layout()
plt.show()
