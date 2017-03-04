# -*- coding: utf-8 -*- 

import gzip
import fileinput

# 时间粒度为10分钟
# 空间粒度为200米
# 20-24为工作日

def euclidean(p1, p2):
	return 200*((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

# 轨迹段分布
def plot_segmentation_distribution():
	from pylab import *
	from scipy import interpolate
	from matplotlib.ticker import MultipleLocator, FormatStrFormatter

	fig = plt.figure(figsize=(10,5))
	fig.subplots_adjust(left=0.05,right=0.98)

	users, delta = {}, 2
	for line in fileinput.input("../data/stationary.txt"):
		uid, _, _, st, ft, _, _ = line.strip().split(" ") # 时间粒度为十分钟
		st, ft = int(st), int(ft)
		users[uid] = users.get(uid,[])
		day = users[uid][-1][3]/(24*6)+(users[uid][-1][1]>=st) if users[uid] else 0
		if len(users[uid]) == 0 or (24*6)*day+st - users[uid][-1][3] >= delta*6:
			users[uid].append([st, ft, (24*6)*day+st, (24*6)*(day+(st>ft))+ft, (ft>st)])
	fileinput.close()

	day_total, row_total = 6, 60
	matrixs, row = [[0 for h in xrange(day_total*24*6)] for u in xrange(row_total)], -1
	for uid, slices in users.iteritems():
		if slices[-1][-2] <= (day_total-1)*24*6 or slices[-1][-2] >= day_total*24*6-1:
			continue
		row += 1
		if row == row_total:
			break
		for sl in slices:
			color = -1 if sl[-1] else 1
			for h in xrange(sl[2],sl[3]+1):
				matrixs[row][h] = color
	
	ax = fig.add_subplot(121)
	(X, Y) = meshgrid(np.arange(day_total*24*6), np.arange(row_total))
	C = np.array(matrixs)
	plt.pcolormesh(X, Y, C, cmap='RdBu', vmin=-2, vmax=2)
	plt.xlim(0,day_total*24*6-1)
	plt.ylim(0,row_total-1)
	xmajorLocator = MultipleLocator(24*6)
	xmajorFormatter = FormatStrFormatter('%d')
	ax.xaxis.set_major_locator(xmajorLocator)
	ax.xaxis.set_major_formatter(xmajorFormatter)
	plt.xlabel('Time slot /10min')
	plt.ylabel('User')

	users, delta = {}, 2
	for line in fileinput.input("../data/stationary.txt"):
		uid, _, _, st, ft, _, _ = line.strip().split(" ") # 时间粒度为十分钟
		st, ft = int(st), int(ft)
		users[uid] = users.get(uid,[])
		day = users[uid][-1][3]/(24*6)+(users[uid][-1][1]>=st) if users[uid] else 0
		users[uid].append([st, ft, (24*6)*day+st, (24*6)*(day+(st>ft))+ft, (ft>st)])
	fileinput.close()

	distribution = {}
	for uid, slices in users.iteritems():
		for i in xrange(1,len(slices)):
			interval = (slices[i][2]-slices[i-1][3])/3
			distribution[interval] = distribution.get(interval,0)+1
	distribution = [distribution.get(t,0) for t in xrange(2*12)]
	distribution = [1-1.*sum(distribution[t:])/sum(distribution) for t in xrange(2*12)]
	ax1 = fig.add_subplot(122)
	tck = interpolate.splrep(range(len(distribution)),distribution,s=0)
	xnew = np.arange(0,2*12,0.1)
	ynew = interpolate.splev(xnew,tck,der=0)
	plt.plot(xnew,ynew,'k-',label="Interval",linewidth=2)
	plt.xlim(1,12)
	plt.ylim(0,1.)
	plt.xlabel('Time slot /30min')
	plt.ylabel('CDF')
	# handles, labels = ax1.get_legend_handles_labels()
	# ax1.legend(handles, labels)
	xmajorLocator = MultipleLocator(1)
	xmajorFormatter = FormatStrFormatter('%d')
	ax1.xaxis.set_major_locator(xmajorLocator)
	ax1.xaxis.set_major_formatter(xmajorFormatter)
	# show()
	for postfix in ('eps','png'):
		savefig('../figure/{0}/04.{0}'.format(postfix))

# 阈值距离为1000米，时间为1小时，仅筛选出工作日
def stationary_accurate_detection():
	min_distance, min_duration, max_duration, min_session = 1000, 1*60/10, 1*60/10, 10
	with open("../data/stationary_accurate.txt", "w") as f:
		line_num = 0
		for line in gzip.open("../data/trace.txt.gz"):
			line_num += 1
			print line_num
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
						if session_current[-1][0]-session_current[0][0] >= min_duration and sl[0]-session_current[-1][0] <= max_duration:
							session_list.append(session_current)
						session_current = [sl]
					else:
						session_current.append(sl)
			if len(session_list) >= min_session:
				for i in range(1,len(session_list)-1):
					if len(session_list[i]) >= 2 and 1*24*60/10 < session_list[i][-1][0] and session_list[i][0][0] < 6*24*60/10:
						f.write(uid+" "+str(round(float(session_list[i][0][0]%(24*60/10))/(60/10),2))+" "+\
										str(round(float(session_list[i][-1][0]%(24*60/10))/(60/10),2))+" "+\
										str(session_list[i][0][0]%(24*60/10))+" "+\
										str(session_list[i][-1][0]%(24*60/10))+" "+\
										str(sum([session[2] for session in session_list[i]])/len(session_list[i]))+" "+\
										str(sum([session[3] for session in session_list[i]])/len(session_list[i]))+"\n") 

def segmentation_detection(function="", method="probability"):
	if function == "segment" and not method in ["probability", "median","cut"]:
		exit()

	def uniform_prob(N):
		prob = [0]*(N)
		for i in xrange(N):
			for j in xrange(i,N):
				if (i+j)%2 == 0:
					prob[int(1.0*(i+j)/2)] += 1
				else:
					prob[int(1.0*(i+j)/2-0.5)] += 0.5
					prob[int(1.0*(i+j)/2+0.5)] += 0.5
		return [prob[i]/sum(prob) for i in xrange(N)]

	import random
	global_prob, local_prob, user_prob, valid_set, delta = {}, {}, {}, [], 5*3
	for line in fileinput.input("../data/stationary_accurate.txt"):
		if function == "plot":
			uid, st, ft, _, _, gx, gy = line.strip().split(" ") # 时间粒度为每小时
		elif function == "segment":
			uid, _, _, st, ft, gx, gy = line.strip().split(" ") # 时间粒度为十分钟
		else:
			exit()
		st, ft, gx, gy = int(float(st)), int(float(ft)), int(gx), int(gy)
		global_prob[st] = global_prob.get(st,0)+1
		global_prob[ft] = global_prob.get(ft,0)+1
		local_prob[(gx, gy)] = local_prob.get((gx, gy),{})
		local_prob[(gx, gy)][st] = local_prob[(gx, gy)].get(st,0)+1
		local_prob[(gx, gy)][ft] = local_prob[(gx, gy)].get(ft,0)+1
		user_prob[uid] = user_prob.get(uid,{})
		user_prob[uid][st] = user_prob[uid].get(st,0)+1
		user_prob[uid][ft] = user_prob[uid].get(ft,0)+1
		valid_set.append([uid,st,ft,gx,gy,\
							max(st-random.randint(0,delta),0),\
							min(st+random.randint(0,delta),24*6-1),\
							max(ft-random.randint(0,delta),0),\
							min(ft+random.randint(0,delta),24*6-1)])
	fileinput.close()

	# 概率时间分布（时间粒度为每小时）
	if function == "plot":
		import matplotlib.pyplot as plt
		line, = plt.plot(range(24), [global_prob[h] for h in range(24)], '-', linewidth=2)
		show()
		for gx in xrange(255):
			for gy in xrange(150):
				if (gx, gy) in local_prob and len(local_prob[(gx, gy)]) == 24:
					line, = plt.plot(range(24), [local_prob[(gx, gy)].get(h,0) for h in xrange(24)], '-', linewidth=2)
					show()

	# 切分点预测（时间粒度为十分钟）
	if function == "segment":
		alpha_global, alpha_local, alpha_user, error = 0.4, 0.4, 0.4, 0.0
		for uid,st,ft,gx,gy,stb,ste,ftb,fte in valid_set:
			if method == "probability":
				probs_global = [global_prob.get(h,0)+(global_prob.get(h-1,0)+global_prob.get(h+1,0))*0.5 
									for h in range(stb,ste+1)]
				probs_local = [local_prob.get((gx,gy),{}).get(h,0)+(local_prob.get((gx,gy),{}).get(h-1,0)+local_prob.get((gx,gy),{}).get(h+1,0))*0.5
									for h in range(stb,ste+1)]
				probs_user = [user_prob.get(uid,{}).get(h,0)+(user_prob.get(uid,{}).get(h-1,0)+user_prob.get(uid,{}).get(h+1,0))*0.5
									for h in range(stb,ste+1)]
				probs_global = [(1-alpha_global)*prob/sum(probs_global)+alpha_global/len(probs_global) for prob in probs_global]
				probs_local = [(1-alpha_local)*prob/sum(probs_local)+alpha_local/len(probs_local) for prob in probs_local]
				probs_user = [(1-alpha_user)*prob/sum(probs_user)+alpha_user/len(probs_user) for prob in probs_user]
				probs_uniform = uniform_prob(ste-stb+1)
				probs = [probs_uniform[h]+probs_global[h]+probs_local[h]+probs_user[h] for h in range(ste-stb+1)]
				error += abs(st-(stb+probs.index(max(probs))))
			elif method == "median":
				error += abs(st-round(1.0*(stb+ste)/2,0))
			elif method == "cut":
				error += abs(st-round(1.0*ste,0))

		print "method={0}, MAE={1}".format(method, 10*(error/len(valid_set)))


if __name__ == "__main__":
	plot_segmentation_distribution()
	# stationary_accurate_detection()
	# segmentation_detection("plot")
	# segmentation_detection("segment","probability")
	# segmentation_detection("segment","median")
	# segmentation_detection("segment","cut")

# 2*30min
# method=probability, MAE=5.80136525334
# method=median, MAE=11.3674646669
# method=cut, MAE=29.5907124315
# 3*30min
# method=probability, MAE=8.06143640035
# method=median, MAE=16.5051437362
# method=cut, MAE=44.0982597827
# 4*30min
# method=probability, MAE=11.9503893856
# method=median, MAE=21.4089029901
# method=cut, MAE=58.3183347755
# 5*30min
# method=probability, MAE=15.1259494279
# method=median, MAE=28.3938082877
# method=cut, MAE=72.5562926642

