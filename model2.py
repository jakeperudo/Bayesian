import pyabc
from pyabc.visualization import plot_kde_2d, plot_data_callback
import matplotlib.pyplot as plt
import os
import tempfile
import numpy as np
import scipy as sp
import logging

df_logger = logging.getLogger('Distance')
df_logger.setLevel(logging.DEBUG)
df_logger = logging.getLogger('Acceptor')
df_logger.setLevel(logging.DEBUG)
df_logger = logging.getLogger('Epsilon')
df_logger.setLevel(logging.DEBUG)

std = 0.5
noise_model = np.random.randn
n_t = 5

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))
           

measurement_data = np.array([646, 603, 568, 398, 555, 708, 603, 703, 717, 				      564, 446, 370, 732, 691, 774, 705, 580, 498, 				      444, 813, 782, 798, 799, 767, 523, 558, 858, 				      843, 1038, 992, 675, 542, 544, 1040, 1018,
			      1059, 1083, 943, 695, 616, 1473, 1339, 1207, 				      1146, 1161])
                             
measurement_times = np.arange(len(measurement_data))

init = np.array([66500000, 46000, 2400, 0, 0, 0])

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
             args=(pars["beta1"], pars["beta2"], pars["gamma1"], 			   pars["gamma2"], pars["delta"], pars["echelon"],	 			   pars["rho"], pars["theta1"], pars["theta2"]))
    return {"X_2": sol[:,5]}


acceptor = pyabc.StochasticAcceptor()
temperature = pyabc.Temperature()
distance = lambda x, x0: np.sum((x['y0']-x0['y0'])**2)
kernel = pyabc.IndependentNormalKernel(var=np.array(std**2))
def fvar(p):
    return p['std']**2 * np.ones(len(data['y0']))
kernel_uvar = pyabc.IndependentNormalKernel(var=fvar)
pop_size = 66500000


parameter_prior = pyabc.Distribution(beta1=pyabc.RV("uniform", 0, 1),
				beta2=pyabc.RV("uniform", 0, 1),
				gamma1=pyabc.RV("uniform", 0, 1),
				gamma2=pyabc.RV("uniform", 0, 1),
				delta=pyabc.RV("uniform", 0, 1),
				echelon=pyabc.RV("uniform", 0, 1),
				rho=pyabc.RV("uniform", 0, 1),
				theta1=pyabc.RV("uniform", 0, 1),
                               theta2=pyabc.RV("uniform", 0, 1))
parameter_prior.get_parameter_names()

abc = pyabc.ABCSMC(model, parameter_prior, kernel, eps=temperature, acceptor=acceptor, population_size=pop_size)

abc.new(db_path, {"X_2": measurement_data});









h = abc.run(minimum_epsilon=0.1, max_nr_populations=5)

fig = plt.figure(figsize=(10,8))
for t in range(h.max_t+1):
    ax = fig.add_subplot(3, np.ceil(h.max_t / 3), t+1)

    ax = plot_kde_2d(
        *h.get_distribution(m=0, t=t), "theta1", "theta2",
        xmin=0, xmax=1, numx=200, ymin=0, ymax=1, numy=200, ax=ax)
    ax.scatter([theta1_true], [theta2_true], color="C1",
                label='$\Theta$ true = {:.3f}, {:.3f}'.format(
                    theta1_true, theta2_true))
    ax.set_title("Posterior t={}".format(t))

    ax.legend()
fig.tight_layout()

_, ax = plt.subplots()

def plot_data(sum_stat, weight, ax, **kwargs):
    """Plot a single trajectory"""
    ax.plot(measurement_times, sum_stat['X_2'], color='grey', alpha=0.1)

def plot_mean(sum_stats, weights, ax, **kwargs):
    """Plot mean over all samples"""
    weights = np.array(weights)
    weights /= weights.sum()
    data = np.array([sum_stat['X_2'] for sum_stat in sum_stats])
    mean = (data * weights.reshape((-1, 1))).sum(axis=0)
    ax.plot(measurement_times, mean, color='C2', label='Sample mean')

ax = plot_data_callback(h, plot_data, plot_mean, ax=ax)

plt.plot(true_trajectory, color="C0", label='Simulation')
plt.scatter(measurement_times, measurement_data,
            color="C1", label='Data')
plt.xlabel('Time $t$')
plt.ylabel('Measurement $Y$')
plt.title('Conversion reaction: Simulated data fit')
plt.legend()
plt.show()


