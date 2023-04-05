import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import norm

# %matplotlib inline

mean = 1
standard_deviation = 2

x_values = np.arange(-2, 2, 0.1)
y_values = norm(mean, standard_deviation)

plt.plot(x_values, y_values.pdf(x_values))
plt.show()

plt.plot(x_values, y_values.pdf(x_values))
plt.plot(x_values, [0.2 for xi in x_values], '--', label='k*q(z)')
plt.legend()
plt.show()

def accept_reject():
    while True:
        u = random.uniform(0, 1)
        q = random.uniform(0, 1) * 4 - 2
        x = y_values.pdf(q) / 0.2
        if u <= x:
            return q

samples = []
for i in range(100000):
    samples.append(accept_reject())
plt.plot(x_values, y_values.pdf(x_values))
normed_value = y_values.cdf(2.0) - y_values.cdf(-2.0)
plt.plot(x_values, y_values.pdf(x_values) / normed_value, 'r', label='normed pdf')

plt.plot(x_values, [0.2 for xi in x_values], '--', label='k*q(z)');
plt.hist(samples, bins=20, density=True, label='sampling')
plt.legend()
plt.show()

# 马尔可夫链的平稳分布
transfer_matrix = np.array([[0.2, 0.8, 0.0], [0.2, 0.6, 0.2], [0.1, 0.0, 0.9]], dtype='float32')
dist = np.array([1.0, 0.0, 0.0], dtype='float32')

single = []
inrelation = []
married = []

for i in range(30):
    dist = np.dot(dist, transfer_matrix)
    single.append(dist[0])
    inrelation.append(dist[1])
    married.append(dist[2])

print(dist)

x = np.arange(30)
plt.plot(x, single, label='single')
plt.plot(x, inrelation, label='inrelation')
plt.plot(x, married, label='married')
plt.legend()
plt.show()


dist = np.array([0.4, 0.3, 0.3], dtype='float32')

single = []
inrelation = []
married = []
for i in range(30):
    dist = np.dot(dist, transfer_matrix)
    single.append(dist[0])
    inrelation.append(dist[1])
    married.append(dist[2])

x = np.arange(30)
plt.plot(x, single, label='single')
plt.plot(x, inrelation, label='inrelation')
plt.plot(x, married, label='married')
plt.legend()
plt.show()

print(dist)

def norm_dist_prob(x, mean=1, std=2):
    return norm(mean, std).pdf(x)

n_1 = 1000

T = 50000
pi = [0 for i in range(T)]
t = 2
sigma = 1
while t < T-1:
    t = t + 1
    p_new = norm.rvs(loc=pi[t-1], scale=sigma, size=1, random_state=None)
    alpha = norm_dist_prob(p_new) * norm(p_new, sigma).pdf(pi[t-1])
    u = random.uniform(0, 1)
    if (u < alpha):
        pi[t] = p_new[0]
    else:
        pi[t] = pi[t-1]

plt.scatter(pi[1000:], norm_dist_prob(pi[1000:]), label='target distribution')
num_bins = 50
plt.hist(pi[1000:], num_bins, density=True, facecolor='red', alpha=0.7, label='sample distribution')
plt.legend()
plt.show()