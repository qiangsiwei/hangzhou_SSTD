# -*- coding: utf-8 -*- 

import glob
import math
import json
import random
import fileinput

# 时间粒度为10分钟
# 空间粒度为200米
# 20-24为工作日

def euclidean(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def gauss(y, mu, sigma):
    # print "gauss computation:", y, mu, sigma
    return 1./math.sqrt((sigma[0][0]*sigma[1][1]-sigma[0][1]*sigma[1][0]))*math.exp(-0.5/(sigma[0][0]*sigma[1][1]-sigma[0][1]*sigma[1][0])*(sigma[0][0]*(y[0]-mu[0])**2-(y[0]-mu[0])*(y[1]-mu[1])*(sigma[0][1]+sigma[1][0])+sigma[1][1]*(y[1]-mu[1])**2))
# print gauss([1,1],[0,0],[[1,0],[0,1]])

K, R, data = 5, 20, []

# GMM
def run_baseline1():
    import numpy as np
    from sklearn import mixture

    for line in fileinput.input("../data/stationary.txt"):
        st, ft, gx, gy = int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5]), int(line.strip().split(" ")[6])
        data.append([st,ft,gx,gy])
    fileinput.close()

    likelihood, param = 0, {}
    gmm_temporal = mixture.GMM(covariance_type="full",n_components=K)
    gmm_temporal.fit(np.array([sample[:2] for sample in data]))
    data_spatial = [[] for k in xrange(K)]
    param['gmm_temporal_weights'] = gmm_temporal.weights_.tolist()
    param['gmm_temporal_means'] = gmm_temporal.means_.tolist()
    param['gmm_temporal_covars'] = gmm_temporal.covars_.tolist()
    param['gmm_spatial'] = []
    for i, k in enumerate(gmm_temporal.predict([sample[:2] for sample in data])):
        data_spatial[k].append(data[i])
    for k in xrange(K):
        print '-'*10, k, '-'*10
        gmm_spatial = mixture.GMM(covariance_type="full",n_components=R)
        gmm_spatial.fit(np.array([sample[2:] for sample in data_spatial[k]]))
        param['gmm_spatial'].append({
            'weights': gmm_spatial.weights_.tolist(),
            'means': gmm_spatial.means_.tolist(),
            'covars': gmm_spatial.covars_.tolist()
        })
        for j, r in enumerate(gmm_spatial.predict([sample[2:] for sample in data_spatial[k]])):
            prob = 1.*gmm_temporal.weights_[k]*gmm_spatial.weights_[r]*\
                    gauss(data_spatial[k][j][:2],gmm_temporal.means_[k],gmm_temporal.covars_[k])*\
                    gauss(data_spatial[k][j][2:],gmm_spatial.means_[r],gmm_spatial.covars_[r])
            likelihood += -math.log10(prob)
    print likelihood

    with open('model_save/baseline1.txt','w') as f:
        f.write(json.dumps(param))


def compute_error():
    import numpy as np

    param = json.loads(open('model_save/baseline1.txt','r').read())
    gmm_temporal_weights = param['gmm_temporal_weights']
    gmm_temporal_means = param['gmm_temporal_means']
    gmm_temporal_covars = param['gmm_temporal_covars']
    gmm_spatial = param['gmm_spatial']
    
    # 时间分布
    matrix1 = [[0 for j in xrange(24*6)] for i in xrange(24*6)]
    for line in fileinput.input("../data/stationary.txt"):
        st, ft = int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4])
        matrix1[st][ft] += 1
    fileinput.close()
    matrix2 = [[0 for j in xrange(24*6)] for i in xrange(24*6)]
    for k in xrange(K):
        for st in xrange(24*6):
            for ft in xrange(24*6):
                matrix2[st][ft] += 1.*gmm_temporal_weights[k]*gauss([st,ft],gmm_temporal_means[k],gmm_temporal_covars[k])
    matrix1 = 1.*np.array(matrix1)/np.array(matrix1).sum()
    matrix2 = 1.*np.array(matrix2)/np.array(matrix2).sum()
    print "Temporal reconstruction accuracy:", 1-(abs(matrix1-matrix2)[40:60,100:140].sum()+abs(matrix1-matrix2)[100:140,40:60].sum())

    # 空间分布
    matrix1 = [[0 for j in xrange(150)] for i in xrange(225)]
    for line in fileinput.input("../data/stationary.txt"):
        gx, gy = int(line.strip().split(" ")[5]), int(line.strip().split(" ")[6])
        matrix1[gx][gy] += 1
    fileinput.close()
    matrix2 = [[0 for j in xrange(150)] for i in xrange(225)]
    for k in xrange(K):
        for r in xrange(R):
            for gx in xrange(225):
                for gy in xrange(150):
                    matrix2[gx][gy] += 1.*gmm_temporal_weights[k]*gmm_spatial[k]['weights'][r]*gauss([gx,gy],gmm_spatial[k]['means'][r],gmm_spatial[k]['covars'][r])
    matrix1 = 1.*np.array(matrix1)/np.array(matrix1).sum()
    matrix2 = 1.*np.array(matrix2)/np.array(matrix2).sum()
    print "Spatial reconstruction accuracy:", 1-abs(matrix1-matrix2)[50:90,50:90].sum()


if __name__ == "__main__":
    # run_baseline1()
    compute_error()
