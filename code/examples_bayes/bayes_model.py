# -*- coding: utf-8 -*- 

import fileinput
import pymc as pm
import numpy as np

data_tp, data_sp = [], []
for line in fileinput.input("../../data/stationary.txt"):
	part = line.strip().split("\t")
	uid, items = part[0], part[1:]
	if uid == "460029901722027":
		for item in items:
			tm, poi = [int(i) for i in item.split(" ")[0:2]], [int(i) for i in item.split(" ")[4].split(",")]
			data_tp.append(tm)
			data_sp.append(poi)
fileinput.close()
data_tp, data_sp = np.array(data_tp), np.array(data_sp)
print data_tp
print data_sp

prior = pm.Dirichlet('prior', np.array([50.0,50.0]))
state = pm.Container([pm.Categorical('state_%i' % i, p=prior) for i in range(len(data_tp))])
stime = pm.Container([pm.DiscreteUniform('stime_%i' % i, lower=0, upper=23) for i in range(2)])
ftime = pm.Container([pm.DiscreteUniform('ftime_%i' % i, lower=0, upper=23) for i in range(2)])
@pm.deterministic(plot=False)
def mu_s(state=state, stime=stime):
	return np.array([stime[0] if state[i] == 0 else stime[1] for i in xrange(len(data_tp))])
@pm.deterministic(plot=False)
def mu_f(state=state, stime=ftime):
	return np.array([ftime[0] if state[i] == 0 else ftime[1] for i in xrange(len(data_tp))])
obs_s = pm.Normal('obs_s', mu=mu_s, tau=0.1, value=data_tp[:,0], observed=True)
obs_f = pm.Normal('obs_f', mu=mu_f, tau=0.1, value=data_tp[:,1], observed=True)
model = pm.Model([prior, state, stime, ftime, obs_s, obs_f])
mcmc = pm.MCMC(model)
mcmc.sample(100)
print state.value
print stime[0].value, ftime[0].value
print stime[1].value, ftime[1].value

# prior = pm.Dirichlet('prior', np.array([50.0,50.0]))
# state = pm.Container([pm.Categorical('state_%i' % i, p=prior) for i in range(len(data_sp))])
# poi_1 = pm.Container([pm.DiscreteUniform('poi_1_%i' % i, lower=0, upper=100) for i in range(2)])
# poi_2 = pm.Container([pm.DiscreteUniform('poi_2_%i' % i, lower=0, upper=100) for i in range(2)])
# poi_3 = pm.Container([pm.DiscreteUniform('poi_3_%i' % i, lower=0, upper=100) for i in range(2)])
# poi_4 = pm.Container([pm.DiscreteUniform('poi_4_%i' % i, lower=0, upper=100) for i in range(2)])
# poi_5 = pm.Container([pm.DiscreteUniform('poi_5_%i' % i, lower=0, upper=100) for i in range(2)])
# @pm.deterministic(plot=False)
# def mu_1(state=state, poi_1=poi_1):
# 	return np.array([poi_1[0] if state[i] == 0 else poi_1[1] for i in xrange(len(data_sp))])
# @pm.deterministic(plot=False)
# def mu_2(state=state, poi_2=poi_2):
# 	return np.array([poi_2[0] if state[i] == 0 else poi_2[1] for i in xrange(len(data_sp))])
# @pm.deterministic(plot=False)
# def mu_3(state=state, poi_3=poi_3):
# 	return np.array([poi_3[0] if state[i] == 0 else poi_3[1] for i in xrange(len(data_sp))])
# @pm.deterministic(plot=False)
# def mu_4(state=state, poi_4=poi_4):
# 	return np.array([poi_4[0] if state[i] == 0 else poi_4[1] for i in xrange(len(data_sp))])
# @pm.deterministic(plot=False)
# def mu_5(state=state, poi_5=poi_5):
# 	return np.array([poi_5[0] if state[i] == 0 else poi_5[1] for i in xrange(len(data_sp))])
# obs_1 = pm.Normal('obs_1', mu=mu_1, tau=0.1, value=data_sp[:,0], observed=True)
# obs_2 = pm.Normal('obs_2', mu=mu_2, tau=0.1, value=data_sp[:,1], observed=True)
# obs_3 = pm.Normal('obs_3', mu=mu_3, tau=0.1, value=data_sp[:,2], observed=True)
# obs_4 = pm.Normal('obs_4', mu=mu_4, tau=0.1, value=data_sp[:,3], observed=True)
# obs_5 = pm.Normal('obs_5', mu=mu_5, tau=0.1, value=data_sp[:,4], observed=True)
# model = pm.Model([prior, state, poi_1, poi_2, poi_3, poi_4, poi_5, obs_1, obs_2, obs_3, obs_4, obs_5])
# mcmc = pm.MCMC(model)
# mcmc.sample(100)
# print state.value
# print poi_1[0].value, poi_2[0].value, poi_3[0].value, poi_4[0].value, poi_5[0].value
# print poi_1[1].value, poi_2[1].value, poi_3[1].value, poi_4[1].value, poi_5[1].value

prior = pm.Dirichlet('prior', np.array([50.0,50.0]))
state = pm.Container([pm.Categorical('state_%i' % i, p=prior) for i in range(len(data_tp))])
stime = pm.Container([pm.DiscreteUniform('stime_%i' % i, lower=0, upper=23) for i in range(2)])
ftime = pm.Container([pm.DiscreteUniform('ftime_%i' % i, lower=0, upper=23) for i in range(2)])
poi_1 = pm.Container([pm.DiscreteUniform('poi_1_%i' % i, lower=0, upper=100) for i in range(2)])
poi_2 = pm.Container([pm.DiscreteUniform('poi_2_%i' % i, lower=0, upper=100) for i in range(2)])
poi_3 = pm.Container([pm.DiscreteUniform('poi_3_%i' % i, lower=0, upper=100) for i in range(2)])
poi_4 = pm.Container([pm.DiscreteUniform('poi_4_%i' % i, lower=0, upper=100) for i in range(2)])
poi_5 = pm.Container([pm.DiscreteUniform('poi_5_%i' % i, lower=0, upper=100) for i in range(2)])
@pm.deterministic(plot=False)
def mu_s(state=state, stime=stime):
	return np.array([stime[0] if state[i] == 0 else stime[1] for i in xrange(len(data_tp))])
@pm.deterministic(plot=False)
def mu_f(state=state, stime=ftime):
	return np.array([ftime[0] if state[i] == 0 else ftime[1] for i in xrange(len(data_tp))])
@pm.deterministic(plot=False)
def mu_1(state=state, poi_1=poi_1):
	return np.array([poi_1[0] if state[i] == 0 else poi_1[1] for i in xrange(len(data_sp))])
@pm.deterministic(plot=False)
def mu_2(state=state, poi_2=poi_2):
	return np.array([poi_2[0] if state[i] == 0 else poi_2[1] for i in xrange(len(data_sp))])
@pm.deterministic(plot=False)
def mu_3(state=state, poi_3=poi_3):
	return np.array([poi_3[0] if state[i] == 0 else poi_3[1] for i in xrange(len(data_sp))])
@pm.deterministic(plot=False)
def mu_4(state=state, poi_4=poi_4):
	return np.array([poi_4[0] if state[i] == 0 else poi_4[1] for i in xrange(len(data_sp))])
@pm.deterministic(plot=False)
def mu_5(state=state, poi_5=poi_5):
	return np.array([poi_5[0] if state[i] == 0 else poi_5[1] for i in xrange(len(data_sp))])
obs_s = pm.Normal('obs_s', mu=mu_s, tau=0.1, value=data_tp[:,0], observed=True)
obs_f = pm.Normal('obs_f', mu=mu_f, tau=0.1, value=data_tp[:,1], observed=True)
obs_1 = pm.Normal('obs_1', mu=mu_1, tau=2, value=data_sp[:,0], observed=True)
obs_2 = pm.Normal('obs_2', mu=mu_2, tau=2, value=data_sp[:,1], observed=True)
obs_3 = pm.Normal('obs_3', mu=mu_3, tau=2, value=data_sp[:,2], observed=True)
obs_4 = pm.Normal('obs_4', mu=mu_4, tau=2, value=data_sp[:,3], observed=True)
obs_5 = pm.Normal('obs_5', mu=mu_5, tau=1, value=data_sp[:,4], observed=True)
model = pm.Model([prior, state, stime, ftime, poi_1, poi_2, poi_3, poi_4, poi_5, obs_s, obs_f, obs_1, obs_2, obs_3, obs_4, obs_5])
mcmc = pm.MCMC(model)
mcmc.sample(100)
print "state:", state.value
print "stime_0:", stime[0].value, ftime[0].value
print "stime_1:", stime[1].value, ftime[1].value
print "poi_0:", poi_1[0].value, poi_2[0].value, poi_3[0].value, poi_4[0].value, poi_5[0].value
print "poi_1:", poi_1[1].value, poi_2[1].value, poi_3[1].value, poi_4[1].value, poi_5[1].value
