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

K, R, alpha, beta, iter_num, data = 5, 20, 0.2, 0.2, 20, []

def run_model(save=True):
    # 数据准备  
    mu_t = [[48,108], [108,48], [60,128], [128,60], [72,84]]
    sigma_t = [[[36,0],[36,0]], [[36,0],[36,0]], [[36,0],[36,0]], [[36,0],[36,0]], [[36,0],[36,0]]]
    assert K == len(mu_t) and K == len(sigma_t)
    print "Total Cluster: {0}".format(K)

    for line in fileinput.input("../data/stationary.txt"):
        st, ft, gx, gy = int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5]), int(line.strip().split(" ")[6])
        ds = [((st-mu_t[k][0])**2+(ft-mu_t[k][1])**2)**0.5 for k in xrange(K)]
        data.append([st,ft,gx,gy,ds.index(min(ds)),-1])
    fileinput.close()

    # from sklearn import mixture
    # est = mixture.GMM(n_components=R, covariance_type="full")
    # est.fit([session[2:4] for session in data])
    # print [[int(i) for i in list(means)] for means in est.means_]
    rs = [[165, 87], [78, 68], [77, 59], [82, 98], [46, 94], [86, 68], [37, 65], [69, 57], [77, 78], [92, 25], [44, 14], [71, 84], [66, 79], [61, 70], [71, 28], [14, 128], [100, 75], [75, 63], [85, 34], [54, 76]]
    assert R == len(rs)
    print "Total Region: {0}".format(R)

    # 初值选取
    for sl in data:
        rd = [euclidean(sl[2:4], rs[r]) for r in xrange(R)]
        sl[-1] = rd.index(min(rd))

    # 初始化
    L = len(data)
    len_k = [float(len(filter(lambda x:x[4]==k, data))) for k in xrange(K)]
    len_k_r = [[float(len(filter(lambda x:x[4]==k and x[5]==r, data))) \
                    for r in xrange(R)] for k in xrange(K)]

    mu1_t = [[float(sum(map(lambda x:x[0],filter(lambda x:x[4]==k, data)))), \
              float(sum(map(lambda x:x[1],filter(lambda x:x[4]==k, data))))] \
                for k in xrange(K)]
    mu2_t = [[float(sum(map(lambda x:x[0]**2,filter(lambda x:x[4]==k, data)))), \
              float(sum(map(lambda x:x[1]**2,filter(lambda x:x[4]==k, data)))), \
              float(sum(map(lambda x:x[0]*x[1],filter(lambda x:x[4]==k, data))))] \
                for k in xrange(K)]
    mu_t = [[mu1_t[k][0]/len_k[k], mu1_t[k][1]/len_k[k]] for k in xrange(K)]
    sigma_t = [[[mu2_t[k][0]/len_k[k]-mu_t[k][0]**2,\
                (mu2_t[k][2]-mu_t[k][0]*mu1_t[k][1]-mu_t[k][1]*mu1_t[k][0])/len_k[k]+mu_t[k][0]*mu_t[k][1]],\
               [(mu2_t[k][2]-mu_t[k][0]*mu1_t[k][1]-mu_t[k][1]*mu1_t[k][0])/len_k[k]+mu_t[k][0]*mu_t[k][1],\
                mu2_t[k][1]/len_k[k]-mu_t[k][1]**2]] \
                    for k in xrange(K)]

    mu1_k_r = [[[float(sum(map(lambda x:x[2],filter(lambda x:x[4]==k and x[5]==r, data)))), \
                 float(sum(map(lambda x:x[3],filter(lambda x:x[4]==k and x[5]==r, data))))] \
                    for r in xrange(R)] for k in xrange(K)]
    mu2_k_r = [[[float(sum(map(lambda x:x[2]**2,filter(lambda x:x[4]==k and x[5]==r, data)))), \
                 float(sum(map(lambda x:x[3]**2,filter(lambda x:x[4]==k and x[5]==r, data)))), \
                 float(sum(map(lambda x:x[2]*x[3],filter(lambda x:x[4]==k and x[5]==r, data))))] \
                    for r in xrange(R)] for k in xrange(K)]
    mu_k_r = [[[mu1_k_r[k][r][0]/len_k_r[k][r], mu1_k_r[k][r][1]/len_k_r[k][r]] for r in xrange(R)] for k in xrange(K)]
    sigma_k_r = [[[[mu2_k_r[k][r][0]/len_k_r[k][r]-mu_k_r[k][r][0]**2,\
                    (mu2_k_r[k][r][2]-mu_k_r[k][r][0]*mu1_k_r[k][r][1]-mu_k_r[k][r][1]*mu1_k_r[k][r][0])/len_k_r[k][r]+mu_k_r[k][r][0]*mu_k_r[k][r][1]],\
                   [(mu2_k_r[k][r][2]-mu_k_r[k][r][0]*mu1_k_r[k][r][1]-mu_k_r[k][r][1]*mu1_k_r[k][r][0])/len_k_r[k][r]+mu_k_r[k][r][0]*mu_k_r[k][r][1],\
                    mu2_k_r[k][r][1]/len_k_r[k][r]-mu_k_r[k][r][1]**2]] \
                        for r in xrange(R)] for k in xrange(K)]

    # 迭代计算
    def find_index(X, a):
        for i in xrange(len(X)):
            for j in xrange(len(X[0])):
                if X[i][j] == a:
                    return (i, j)

    for iter_curr in xrange(iter_num):
        likelihood = 0
        for i in xrange(L):
            item, old_k, old_r = data[i], data[i][4], data[i][5]
            probs = [[1.*len_k[k]/sum(len_k)*len_k_r[k][r]/sum(len_k_r[k])*\
                        gauss(item[0:2],mu_t[k],sigma_t[k])*\
                        gauss(item[2:4],mu_k_r[k][r],sigma_k_r[k][r]) \
                            for r in xrange(R)] for k in xrange(K)]
            prob_max = max([max(prob) for prob in probs])
            likelihood += -math.log10(sum([sum(prob) for prob in probs]))
            new_k, new_r = find_index(probs, prob_max)
            data[i][4], data[i][5] = new_k, new_r
            len_k[old_k] -= 1; len_k[new_k] += 1
            len_k_r[old_k][old_r] -= 1; len_k_r[new_k][new_r] += 1
            mu1_t[old_k][0] -= item[0]; mu1_t[old_k][1] -= item[1]
            mu1_t[new_k][0] += item[0]; mu1_t[new_k][1] += item[1]
            mu2_t[old_k][0] -= item[0]**2; mu2_t[old_k][1] -= item[1]**2; mu2_t[old_k][2] -= item[0]*item[1]
            mu2_t[new_k][0] += item[0]**2; mu2_t[new_k][1] += item[1]**2; mu2_t[new_k][2] += item[0]*item[1]
            mu_t = [[mu1_t[k][0]/len_k[k], mu1_t[k][1]/len_k[k]] for k in xrange(K)]
            sigma_t = [[[mu2_t[k][0]/len_k[k]-mu_t[k][0]**2,\
                        (mu2_t[k][2]-mu_t[k][0]*mu1_t[k][1]-mu_t[k][1]*mu1_t[k][0])/len_k[k]+mu_t[k][0]*mu_t[k][1]],\
                       [(mu2_t[k][2]-mu_t[k][0]*mu1_t[k][1]-mu_t[k][1]*mu1_t[k][0])/len_k[k]+mu_t[k][0]*mu_t[k][1],\
                        mu2_t[k][1]/len_k[k]-mu_t[k][1]**2]] \
                            for k in xrange(K)]
            mu1_k_r[old_k][old_r][0] -= item[2]; mu1_k_r[old_k][old_r][1] -= item[3]
            mu1_k_r[new_k][new_r][0] += item[2]; mu1_k_r[new_k][new_r][1] += item[3]
            mu2_k_r[old_k][old_r][0] -= item[2]**2; mu2_k_r[old_k][old_r][1] -= item[3]**2; mu2_k_r[old_k][old_r][2] -= item[2]*item[3];
            mu2_k_r[new_k][new_r][0] += item[2]**2; mu2_k_r[new_k][new_r][1] += item[3]**2; mu2_k_r[new_k][new_r][2] += item[2]*item[3];
            mu_k_r = [[[mu1_k_r[k][r][0]/len_k_r[k][r], mu1_k_r[k][r][1]/len_k_r[k][r]] for r in xrange(R)] for k in xrange(K)]
            sigma_k_r = [[[[mu2_k_r[k][r][0]/len_k_r[k][r]-mu_k_r[k][r][0]**2,\
                            (mu2_k_r[k][r][2]-mu_k_r[k][r][0]*mu1_k_r[k][r][1]-mu_k_r[k][r][1]*mu1_k_r[k][r][0])/len_k_r[k][r]+mu_k_r[k][r][0]*mu_k_r[k][r][1]],\
                           [(mu2_k_r[k][r][2]-mu_k_r[k][r][0]*mu1_k_r[k][r][1]-mu_k_r[k][r][1]*mu1_k_r[k][r][0])/len_k_r[k][r]+mu_k_r[k][r][0]*mu_k_r[k][r][1],\
                            mu2_k_r[k][r][1]/len_k_r[k][r]-mu_k_r[k][r][1]**2]] \
                                for r in xrange(R)] for k in xrange(K)]
        print iter_curr, likelihood, len_k

        if save:
            with open('model_save/iter_{0}.txt'.format(str(iter_curr).zfill(2)),'w') as f:
                f.write(json.dumps({"likelihood":likelihood,
                                    "len_k":len_k,
                                    "len_k_r":len_k_r,
                                    "mu_t":mu_t,
                                    "sigma_t":sigma_t,
                                    "mu_k_r":mu_k_r,
                                    "sigma_k_r":sigma_k_r}))

def plot_distribution(iter_curr):
    from pylab import *

    param = json.loads(open('model_save/iter_{0}.txt'.format(iter_curr),'r').read())
    len_k, len_k_r = param['len_k'], param['len_k_r']
    mu_t, sigma_t = param['mu_t'], param['sigma_t']
    mu_k_r, sigma_k_r = param['mu_k_r'], param['sigma_k_r']

    # 时间分布
    plt.figure(figsize=(12,5))
    norm1 = cm.colors.Normalize(vmax=0.0020, vmin=0)
    for c, k in enumerate([4,0,1,2,3]):
        matrix = [[0 for j in xrange(24*6)] for i in xrange(24*6)]
        for ts in xrange(24*6):
            for tf in xrange(24*6):
                matrix[ts][tf] = 1.*(len_k[k]/sum(len_k))*gauss([ts,tf],mu_t[k],sigma_t[k])
        (X, Y), C = meshgrid(np.arange(24*6), np.arange(24*6)), np.array(matrix)
        subplot(2,5,1+c)
        cset = pcolormesh(X, Y, C.T, cmap=cm.get_cmap("OrRd"), norm=norm1)
        plt.axis([0, 24*6, 0, 24*6])
        plt.xticks(np.linspace(0,24*6,7))
        plt.yticks(np.linspace(0,24*6,7))
        if c == 0:
            plt.xlabel('Session start time slot /10min')
            plt.ylabel('Session end time slot /10min')
    cax1 = axes([0.92, 0.54, 0.01, 0.35])
    colorbar(cax=cax1)
    # plt.axis('off')

    # 空间分布
    subplots_adjust(hspace=0.4)
    norm2 = cm.colors.Normalize(vmax=0.0040, vmin=0)
    for c, k in enumerate([4,0,1,2,3]):
        matrix = [[0 for j in xrange(150)] for i in xrange(225)]
        for gx in xrange(225):
            for gy in xrange(150):
                matrix[gx][gy] = 1.*(len_k[k]/sum(len_k))*sum([1.*(len_k_r[k][r]/sum(len_k_r[k]))*gauss([gx,gy],mu_k_r[k][r],sigma_k_r[k][r]) for r in xrange(R)])
        (X, Y), C = meshgrid(np.arange(100), np.arange(100)), np.array(matrix)[20:120,20:120]
        subplot(2,5,6+c)
        cset = pcolormesh(X, Y, C.T, cmap=cm.get_cmap("OrRd"), norm=norm2)
        plt.axis([0, 100-1, 0, 100-1])
        plt.xticks(np.linspace(0,100,6))
        plt.yticks(np.linspace(0,100,6))
        if c == 0:
            plt.xlabel('Longitude grid index /200m')
            plt.ylabel('Latitude grid index /200m')
    subplots_adjust(bottom=0.1, left=0.06, right=0.9, top=0.9)
    cax2 = axes([0.92, 0.09, 0.01, 0.35])
    colorbar(cax=cax2)
    # plt.axis('off')
    # show()
    for postfix in ('eps','png'):
        savefig('../figure/{0}/05.{0}'.format(postfix))

def plot_iteration_likelihood():
    from pylab import *

    iterations, likelihoods = [], []
    for iteration, filename in enumerate(sorted(glob.glob(r"model_save/iter_*.txt"))):
        likelihood = json.loads(open(filename,'r').read()).get("likelihood",0)
        iterations.append(iteration)
        likelihoods.append(likelihood/10**4)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot(iterations, likelihoods, 'k-', label="Likelihood", linewidth=2)
    plt.xlabel('Number for iteration')
    plt.ylabel('$-10^{-4} \\times$ log likelihood')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    # show()
    for postfix in ('eps','png'):
        savefig('../figure/{0}/06.{0}'.format(postfix))

def compute_error(iter_curr):
    import numpy as np

    param = json.loads(open('model_save/iter_{0}.txt'.format(iter_curr),'r').read())
    len_k, len_k_r = param['len_k'], param['len_k_r']
    mu_t, sigma_t = param['mu_t'], param['sigma_t']
    mu_k_r, sigma_k_r = param['mu_k_r'], param['sigma_k_r']

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
                matrix2[st][ft] += 1.*(len_k[k]/sum(len_k))*gauss([st,ft],mu_t[k],sigma_t[k])
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
                    matrix2[gx][gy] += 1.*(len_k[k]/sum(len_k))*(len_k_r[k][r]/sum(len_k_r[k]))*gauss([gx,gy],mu_k_r[k][r],sigma_k_r[k][r])
    matrix1 = 1.*np.array(matrix1)/np.array(matrix1).sum()
    matrix2 = 1.*np.array(matrix2)/np.array(matrix2).sum()
    print "Spatial reconstruction accuracy:", 1-abs(matrix1-matrix2)[50:90,50:90].sum()


if __name__ == "__main__":
    # run_model(save=True)
    plot_distribution(19)
    # plot_iteration_likelihood()
    # compute_error(19)

# Likelihood
# model:     2636999
# baseline1: 3336792
# baseline2: 2644190

# model:
# Temporal reconstruction accuracy: 0.867786928604
# Spatial reconstruction accuracy: 0.746619059294
# baseline1:
# Temporal reconstruction accuracy: 0.861796706542
# Spatial reconstruction accuracy: 0.659659953592
# baseline2:
# Temporal reconstruction accuracy: 0.868461451694
# Spatial reconstruction accuracy: 0.719261119801

