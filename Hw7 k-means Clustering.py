##John Okechukwu##

import sys
import random

data = sys.argv[1]

f = open(data, 'r')
data = []
# i = 0
l = f.readline()


### Read Data ###


while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))

    data.append(l2)

    l = f.readline()

data_rows = len(data)
data_cols = len(data[0])


f.close()

try:
    k = int(3)
except IndexError:
    print ("input appropriate number of clusters")

    sys.exit()

###initialize m
m = []
col = []
for j in range(0, data_cols, 1):
    col.append(0)

for i in range(0, k, 1):
    m.append(col)

rand = 0

for p in range(0, k, 1):
    rand=random.randrange(0,(data_rows-1))
    m[p] = data[rand]


#class


train_labels = {}
diff = 1

prev = [[0]*data_cols for x in range(k)]
dist =[]
mdist =[]
for p in range(0, k, 1):
    mdist.append(0)
n = []
for p in range(0, k, 1):
    dist.append(0.1)
for p in range(0, k, 1):
    n.append(0.1)
sum_dist =1
classes=[]

while ((sum_dist) > 0):
    for i in range(0,data_rows, 1):
        dist =[]

        for p in range(0, k, 1):
            dist.append(0)
        for p in range(0, k, 1):
            for j in range(0, data_cols, 1):
                dist[p] += ((data[i][j] - m[p][j])**2)
        for p in range(0, k, 1):
            dist[p] = (dist[p])**0.5
        min_dist=0
        min_dist = min(dist)

        for p in range(0, k, 1):
            if(dist[p]==min_dist):
                train_labels[i] = p

                n[p]+=1

                break

    m = [[0]*data_cols for x in range(k)]
    col = []

    for i in range(0, data_rows, 1):
        for p in range(0, k, 1):
            if(train_labels.get(i) == p):
                for j in range(0, data_cols, 1):
                    temp =  m[p][j]
                    temp1 =  data[i][j]
                    m[p][j] = temp + temp1

    for j in range(0, data_cols, 1):
        for i in range(0, k, 1):
            m[i][j] = m[i][j]/n[i]

    classes = [int(x) for x in n]
    n=[0.1]*k

#dist

    mdist = []
    for i in range(0, k, 1):
        mdist.append(0)
    for p in range(0, k, 1):
        for j in range(0, data_cols, 1):
            mdist[p]+=float((prev[p][j]-m[p][j])**2)
        mdist[p] = (mdist[p])**0.5
    prev=m
    sum_dist = 0
    for f in range(0,len(mdist),1):
        sum_dist += mdist[f]




for i in range(0,data_rows, 1):
    print(train_labels[i],i)