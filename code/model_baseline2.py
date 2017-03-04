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

def run_baseline2():
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
    # est = mixture.GMM(n_components=R, covariacne_type="full")
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
    len_r = [float(len(filter(lambda x:x[5]==r, data))) for r in xrange(R)]
    len_k_r = [[float(len(filter(lambda x:x[4]==k and x[5]==r, data))) for r in xrange(R)] for k in xrange(K)]
    
    mu1_t = [[float(sum(map(lambda x:x[0],filter(lambda x:x[4]==k, data)))), float(sum(map(lambda x:x[1],filter(lambda x:x[4]==k, data))))] for k in xrange(K)]
    mu2_t = [[float(sum(map(lambda x:x[0]**2,filter(lambda x:x[4]==k, data)))), float(sum(map(lambda x:x[1]**2,filter(lambda x:x[4]==k, data)))), float(sum(map(lambda x:x[0]*x[1],filter(lambda x:x[4]==k, data))))] for k in xrange(K)]
    mu_t = [[mu1_t[k][0]/len_k[k], mu1_t[k][1]/len_k[k]] for k in xrange(K)]
    sigma_t = [[[mu2_t[k][0]/len_k[k]-mu_t[k][0]**2,(mu2_t[k][2]-mu_t[k][0]*mu1_t[k][1]-mu_t[k][1]*mu1_t[k][0])/len_k[k]+mu_t[k][0]*mu_t[k][1]],[(mu2_t[k][2]-mu_t[k][0]*mu1_t[k][1]-mu_t[k][1]*mu1_t[k][0])/len_k[k]+mu_t[k][0]*mu_t[k][1],mu2_t[k][1]/len_k[k]-mu_t[k][1]**2]] for k in xrange(K)]
    
    mu1_r = [[float(sum(map(lambda x:x[2],filter(lambda x:x[5]==r, data)))), float(sum(map(lambda x:x[3],filter(lambda x:x[5]==r, data))))] for r in xrange(R)]
    mu2_r = [[float(sum(map(lambda x:x[2]**2,filter(lambda x:x[5]==r, data)))), float(sum(map(lambda x:x[3]**2,filter(lambda x:x[5]==r, data)))), float(sum(map(lambda x:x[2]*x[3],filter(lambda x:x[5]==r, data))))] for r in xrange(R)]
    mu_r = [[mu1_r[r][0]/len_r[r], mu1_r[r][1]/len_r[r]] for r in xrange(R)]
    sigma_r = [[[mu2_r[r][0]/len_r[r]-mu_r[r][0]**2,(mu2_r[r][2]-mu_r[r][0]*mu1_r[r][1]-mu_r[r][1]*mu1_r[r][0])/len_r[r]+mu_r[r][0]*mu_r[r][1]],[(mu2_r[r][2]-mu_r[r][0]*mu1_r[r][1]-mu_r[r][1]*mu1_r[r][0])/len_r[r]+mu_r[r][0]*mu_r[r][1],mu2_r[r][1]/len_r[r]-mu_r[r][1]**2]] for r in xrange(R)]

    # 迭代计算
    for iter_curr in xrange(iter_num):
        likelihood = 0
        for i in xrange(L):
            item, co = data[i], data[i][4]
            # sample R
            prob = [1.*len_k_r[co][r]/len_k[co]*gauss(item[2:4],mu_r[r],sigma_r[r]) for r in xrange(R)]
            ro, rn = item[5], prob.index(max(prob))
            if rn != ro:
                data[i][5] = rn
                len_r[ro] -= 1; len_r[rn] += 1
                len_k_r[co][ro] -= 1; len_k_r[co][rn] += 1
                mu1_r[ro][0] -= item[2]; mu1_r[ro][1] -= item[3]
                mu1_r[rn][0] += item[2]; mu1_r[rn][1] += item[3]
                mu2_r[ro][0] -= item[2]**2; mu2_r[ro][1] -= item[3]**2; mu2_r[ro][2] -= item[2]*item[3]
                mu2_r[rn][0] += item[2]**2; mu2_r[rn][1] += item[3]**2; mu2_r[rn][2] += item[2]*item[3]
                mu_r = [[mu1_r[r][0]/len_r[r], mu1_r[r][1]/len_r[r]] for r in xrange(R)]
                sigma_r = [[[mu2_r[r][0]/len_r[r]-mu_r[r][0]**2,(mu2_r[r][2]-mu_r[r][0]*mu1_r[r][1]-mu_r[r][1]*mu1_r[r][0])/len_r[r]+mu_r[r][0]*mu_r[r][1]],[(mu2_r[r][2]-mu_r[r][0]*mu1_r[r][1]-mu_r[r][1]*mu1_r[r][0])/len_r[r]+mu_r[r][0]*mu_r[r][1],mu2_r[r][1]/len_r[r]-mu_r[r][1]**2]] for r in xrange(R)]
            # sample K
            prob = [1.*len_k_r[k][rn]/L*gauss(item[:2],mu_t[k],sigma_t[k])*gauss(item[2:4],mu_r[rn],sigma_r[rn]) for k in xrange(K)]
            cn = prob.index(max(prob))
            if cn != co:
                data[i][4] = cn
                len_k[co] -= 1; len_k[cn] += 1
                len_k_r[co][rn] -= 1; len_k_r[cn][rn] += 1
                mu1_t[co][0] -= item[0]; mu1_t[co][1] -= item[1]
                mu1_t[cn][0] += item[0]; mu1_t[cn][1] += item[1]
                mu2_t[co][0] -= item[0]**2; mu2_t[co][1] -= item[1]**2; mu2_t[co][2] -= item[0]*item[1]
                mu2_t[cn][0] += item[0]**2; mu2_t[cn][1] += item[1]**2; mu2_t[cn][2] += item[0]*item[1]
                mu_t = [[mu1_t[k][0]/len_k[k], mu1_t[k][1]/len_k[k]] for k in xrange(K)]
                sigma_t = [[[mu2_t[k][0]/len_k[k]-mu_t[k][0]**2,(mu2_t[k][2]-mu_t[k][0]*mu1_t[k][1]-mu_t[k][1]*mu1_t[k][0])/len_k[k]+mu_t[k][0]*mu_t[k][1]],[(mu2_t[k][2]-mu_t[k][0]*mu1_t[k][1]-mu_t[k][1]*mu1_t[k][0])/len_k[k]+mu_t[k][0]*mu_t[k][1],mu2_t[k][1]/len_k[k]-mu_t[k][1]**2]] for k in xrange(K)]
            
            prob_max = 1.*len_k[cn]/sum(len_k)*len_k_r[cn][rn]/sum(len_k_r[cn])*gauss(item[:2],mu_t[cn],sigma_t[cn])*gauss(item[2:4],mu_r[rn],sigma_r[rn])
            likelihood += -math.log10(prob_max)

        print iter_curr, likelihood

    with open('model_save/baseline2.txt','w') as f:
        f.write(json.dumps({"len_k":len_k,
                            "len_k_r":len_k_r,
                            "mu_t":mu_t,
                            "sigma_t":sigma_t,
                            "mu_r":mu_r,
                            "sigma_r":sigma_r}))

def compute_error():
    import numpy as np

    param = json.loads(open('model_save/baseline2.txt','r').read())
    len_k, len_k_r = param['len_k'], param['len_k_r']
    mu_t, sigma_t = param['mu_t'], param['sigma_t']
    mu_r, sigma_r = param['mu_r'], param['sigma_r']

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
                    matrix2[gx][gy] += 1.*(len_k[k]/sum(len_k))*(len_k_r[k][r]/sum(len_k_r[k]))*gauss([gx,gy],mu_r[r],sigma_r[r])
    matrix1 = 1.*np.array(matrix1)/np.array(matrix1).sum()
    matrix2 = 1.*np.array(matrix2)/np.array(matrix2).sum()
    print "Spatial reconstruction accuracy:", 1-abs(matrix1-matrix2)[50:90,50:90].sum()


if __name__ == "__main__":
    # run_baseline2()
    compute_error()
