# -*- coding: utf-8 -*- 

import sys
from operator import add
from pyspark import SparkConf
from pyspark import SparkContext

def extract(line):
	import time
	try:
		part = line.strip().replace('\"','').split(",")
		TTIME, LAC, CI, IMSI = part[1].split(" "), part[3], part[4], part[5]
		pt1, pt2, pt3 = TTIME[0].split("-"), TTIME[1].split("."), TTIME[2]
		year, month, day, hour, minute, second = int("20"+pt1[2]), {"AUG":8}[pt1[1]], int(pt1[0]), int(pt2[0]), int(pt2[1]), int(pt2[2])
		hour = hour if hour != 12 else 0
		hour = hour if pt3 == "AM" else hour+12
		secs = hour*3600+minute*60+second
		key = LAC+" "+CI
		sl = secs/(10*60)
		if bss.has_key(key):
			bs = bss[key]
			lng, lat = bs["lng"], bs["lat"]
			if 120.02<=lng<120.48 and 30.15<=lat<=30.42:
				gx, gy = int((lng-120.02)/(120.48-120.02)*225), int((lat-30.15)/(30.42-30.15)*150)
				return ((IMSI, sl), str(gx)+","+str(gy))
			else:
				return (("", -1), "")
		else:
			return (("", -1), "")
	except:
		return (("", -1), "")

global bss

if __name__ == "__main__":
	import fileinput
	bss = {}
	for line in fileinput.input("hz_base.txt"):
		part = line.strip().split(" ")
		num, lng, lat = part[1]+" "+part[2], float(part[3]), float(part[4])
		bss[num] = {"lng":lng, "lat":lat}
	fileinput.close()
	conf = SparkConf().setMaster('yarn-client') \
					  .setAppName('qiangsiwei') \
					  .set('spark.driver.maxResultSize', "8g")
	sc = SparkContext(conf = conf)
	filename = "0826"
	lines = sc.textFile("hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/hangzhou/original/{0}.csv".format(filename), 1)
	counts = lines.map(lambda x : extract(x)) \
				  .filter(lambda x : x[0][0]!="" and x[0][1]!=-1 and x[1]!="") \
				  .distinct() \
				  .groupByKey() \
				  .map(lambda x : (x[0][0],str(x[0][1])+":"+"-".join(sorted(x[1])))) \
				  .groupByKey() \
				  .map(lambda x : x[0]+"\t"+"|".join([str(it["sl"])+":"+it["gs"] for it in sorted([{"sl":int(line.split(":")[0]),"gs":line.split(":")[1]} for line in x[1]], key=lambda x:x["sl"])]))
	output = counts.saveAsTextFile("./hangzhou/SSTD/3G/{0}.csv".format(filename))
