# -*- coding: utf-8 -*- 

import gzip
import fileinput
import numpy as np
from pylab import *

# 时间粒度为10分钟
# 空间粒度为200米
# 20-24为工作日

def euclidean(p1, p2):
	return 200*((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

# 轨迹连接
def trajectory_concat():
	users = {}
	for day, filename in enumerate(['0820','0821','0822','0823','0824']):
		print filename
		for line in gzip.open("../data/3G/{0}.txt.gz".format(filename)):
			uid, slices = line.strip().split("\t")
			slices = ["{0}:{1}".format(int(sl.split(":")[0])+day*6*24,sl) 
						for sl in slices.split("|")]
			users[uid] = users.get(uid,[])
			users[uid].extend(slices)
	with open("../data/trace.txt","w") as f:
		for uid, slices in users.iteritems():
			f.write("{0}\t{1}\n".format(uid,"|".join(slices)))

# 阈值距离为1000米，时间为1小时，仅筛选出工作日
def stationary_detection():
	min_distance, min_duration, min_session = 1000, 1*60/10, 10
	with open("../data/stationary.txt", "w") as f:
		# line_num = 0
		for line in gzip.open("../data/trace.txt.gz"):
			# line_num += 1
			# print line_num
			uid = line.strip().split("\t")[0]
			session_list, session_current, slices = [], [], [(int(sl.split(":")[0]), \
															  int(sl.split(":")[1]), \
															  sum([int(p.split(",")[0]) for p in sl.split(":")[2].split("-")])/len(sl.split(":")[2].split("-")), \
															  sum([int(p.split(",")[1]) for p in sl.split(":")[2].split("-")])/len(sl.split(":")[2].split("-"))) \
																for sl in line.strip().split("\t")[1].split("|")]
			for sl in slices:
				if len(session_current) == 0:
					session_current = [sl]
				else:
					if euclidean(sl[2:],session_current[-1][2:]) >= min_distance:
						if session_current[-1][0]-session_current[0][0] >= min_duration:
							session_list.append(session_current)
						session_current = [sl]
					else:
						session_current.append(sl)
			if session_current[-1][0]-session_current[0][0] >= min_duration:
				session_list.append(session_current)
			if len(session_list) >= min_session:
				for i in range(1,len(session_list)-1):
					if len(session_list[i]) >= 2 and 1*24*60/10 < session_list[i][-1][0] and session_list[i][0][0] < 6*24*60/10:
						f.write(uid+" "+str(round(float(session_list[i][0][0]%(24*60/10))/(60/10),2))+" "+\
										str(round(float(session_list[i][-1][0]%(24*60/10))/(60/10),2))+" "+\
										str(session_list[i][0][0]%(24*60/10))+" "+\
										str(session_list[i][-1][0]%(24*60/10))+" "+\
										str(sum([session[2] for session in session_list[i]])/len(session_list[i]))+" "+\
										str(sum([session[3] for session in session_list[i]])/len(session_list[i]))+"\n") 

def stationary_statistic():
	matrix = [[0 for j in xrange(24*6)] for i in xrange(24*6)]
	for line in fileinput.input("../data/stationary.txt"):
		matrix[int(line.strip().split(" ")[3])][int(line.strip().split(" ")[4])] += 1
	fileinput.close()
	(X, Y), C = meshgrid(np.arange(24*6), np.arange(24*6)), np.array(matrix)
	# 时间分布
	subplot(1,1,1)
	cset = pcolormesh(X, Y, C.T, cmap=cm.get_cmap("OrRd"))
	plt.axis([0, 24*6-1, 0, 24*6-1])
	colorbar(cset)
	plt.xlabel('Session entering time slot /10min')
	plt.ylabel('Session leaving time slot /10min')
	# show()
	for postfix in ('eps','png'):
		savefig('../figure/{0}/01.{0}'.format(postfix))

	matrix1, matrix2 = [[0 for j in xrange(150)] for i in xrange(225)], [[0 for j in xrange(150)] for i in xrange(225)]
	for line in fileinput.input("../data/stationary.txt"):
		ts, tf, gx, gy = int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5]), int(line.strip().split(" ")[6])
		d1, d2 = ((ts-50)**2+(tf-110)**2)**(1.0/2), ((ts-110)**2+(tf-50)**2)**(1.0/2)
		if d1 <= d2:
			matrix1[gx][gy] += 1
		else:
			matrix2[gx][gy] += 1
	fileinput.close()
	(X, Y), C1, C2 = meshgrid(np.arange(100), np.arange(100)), np.array(matrix1)[20:120,20:120], np.array(matrix2)[20:120,20:120]
	# 空间分布
	plt.figure(figsize=(12,5))
	plt.subplots_adjust(left=0.05,right=1.00)
	subplot(1,2,1)
	cset1 = pcolormesh(X, Y, C1.T, cmap=cm.get_cmap("OrRd"))
	plt.axis([0, 100-1, 0, 100-1])
	colorbar(cset1)
	plt.xlabel('Longitude grid index /200m')
	plt.ylabel('Latitude grid index /200m')
	plt.title('Diurnal')
	subplot(1,2,2)
	cset2 = pcolormesh(X, Y, C2.T, cmap=cm.get_cmap("OrRd"))
	plt.axis([0, 100-1, 0, 100-1])
	colorbar(cset2)
	plt.title('Nocturnal')
	# show()
	for postfix in ('eps','png'):
		savefig('../figure/{0}/02.{0}'.format(postfix))


if __name__ == "__main__":
	# trajectory_concat()
	# stationary_detection()
	stationary_statistic()

