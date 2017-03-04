import time
import fileinput
import pymc as pm
import numpy as np

st = time.time()
K = 2 # number of topics
V = 3 # number of words
D = 3 # number of documents
data = np.array([[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
alpha = np.ones(K)
beta = np.ones(V)
theta = pm.Container([pm.CompletedDirichlet("theta_%s" % i, pm.Dirichlet("ptheta_%s" % i, theta=alpha)) for i in range(D)])
phi = pm.Container([pm.CompletedDirichlet("phi_%s" % k, pm.Dirichlet("pphi_%s" % k, theta=beta)) for k in range(K)])
Wd = [len(doc) for doc in data]
z = pm.Container([pm.Categorical('z_%i' % d, p=theta[d], size=Wd[d], value=np.random.randint(K,size=Wd[d])) for d in range(D)])
w = pm.Container([pm.Categorical("w_%i_%i" % (d,i), p=pm.Lambda('phi_z_%i_%i' % (d,i), lambda z=z[d][i], phi=phi:phi[z]), value=data[d][i], observed=True) for d in range(D) for i in range(Wd[d])])
model = pm.Model([theta, phi, z, w])
mcmc = pm.MCMC(model)
mcmc.sample(1000)
ft = time.time()
print ft-st
print theta.value
print phi.value