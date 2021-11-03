import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi


# Mean wind-speed
z = 14.5
vb = 30.5  # vb = 30.5 for 50-y RP (as in AMC aerodynamics report PDF page 82).
z0 = 0.01
kt = 0.17
U_bar = kt * np.log(z / z0) * vb * 1.07  # "10min mean is found by multiplying V(z) by 1.07."

# Turbulence
Iu = 0.14
Iv = 0.85 * Iu
Iw = 0.55 * Iu

sigma_u = Iu * U_bar
sigma_v = Iv * U_bar
sigma_w = Iw * U_bar

# This is not the same as real wind time series. These are totally uncorrelated occurrences of wind speeds.
V = np.random.normal(U_bar, sigma_u, 10000000)  # total along wind velocity
v = np.random.normal(0, sigma_v, 10000000)  # total along wind velocity
w = np.random.normal(0, sigma_w, 10000000)  # total along wind velocity

# Total horizontal speed
V_v = np.sqrt(V**2 + v**2)

# Theta angle - angle of attack
thetas = deg(np.arctan(w / V_v))

thetas_between_plus_minus_10 = sum(-10<t<10 for t in thetas)
thetas_between_plus_minus_3 = sum(-3<t<3 for t in thetas)
n_thetas = len(thetas)

from numpy.random import normal
from numpy import mean
from numpy import std
from scipy.stats import norm
import matplotlib
# generate a sample
sample = thetas
# calculate parameters
sample_mean = mean(sample)
sample_std = std(sample)
print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
# define the distribution
dist = norm(sample_mean, sample_std)
# sample probabilities for a range of outcomes
values = [value for value in range(-15, 15+1)]
probabilities = [dist.pdf(value) for value in values]
# plot the histogram and pdf
plt.figure(figsize=(6,3), dpi=300)
plt.hist(sample, bins=300, density=True, alpha=0.5,label='Histogram', color='blue', edgecolor=matplotlib.colors.colorConverter.to_rgba('grey', alpha=.5))
plt.axvspan(-3, 3, facecolor='yellow', alpha=0.3, label=r'Tested domain of $\theta$')
plt.plot(values, probabilities, label='Normal distribution', c='orange', linewidth=3, alpha=0.6)
plt.legend()
plt.xlim([-15,15])
plt.ylim([0,0.1])
plt.grid()
# plt.ylabel('Probability density')
plt.title(r'Probability density function of $\~{\theta}$')
plt.xlabel(r'$\~{\theta}$ [deg]')
plt.tight_layout()

# Plotting
# plt.figure(figsize=(6,5), dpi=300)
# plt.hist(thetas, bins=400, density=True, label='Histogram', alpha=0.6, histtype='step')
# plt.grid()
# plt.ylabel('Probability density')
# plt.title(r'Probability density function of $\theta$')
# plt.xlabel(r'$\~{\theta}$ (or $\theta$ interval) [deg]')
#
# plt.legend()
# plt.xlim([-15,15])
# plt.ylim([0,0.1])
#
# from scipy.stats import gaussian_kde
# gaussian_kde.pdf(thetas)
#
# graph = sns.distplot(thetas, hist=True, kde=True,
#              bins=int(max(thetas)-min(thetas)), color = 'darkblue',
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 4})
# graph.set(xlim=(-15, 15))


