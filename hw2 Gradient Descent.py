import sys
import random
import math

datafile = "LeastSquareSampleAll.data"

f = open(datafile, 'r')
data = []
# i = 0
l = f.readline()

#################
### Read Data ###
#################

while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    l2.append(float(1))
    data.append(l2)
    l = f.readline()

rows = len(data)
cols = len(data[0])



f.close()


##### read training labels ####


labelfile = "LeastSquareSample.trainlabels"
f = open(labelfile)
trainlabels = {}
n = [0, 0]
l = f.readline()
while (l != ''):
    a = l.split()
    trainlabels[a[1]] = int(a[0])
    #    trainlabels_size[a[0]] = trainlabels_size[a[0]]+1
    if (trainlabels[a[1]] == 0):
        trainlabels[a[1]] = -1;
    l = f.readline()

    n[int(a[0])] += 1

f.close()

# initialize w
w = []
for j in range(0, cols, 1):
    # print(random.random())
    w.append(0.02 * random.random() - 0.01)


# print(w)


#### calculation of doc product #####


def dotProduct(a, b):
    dp = 0
    for i in range(0, cols, 1):
        dp += a[i] * b[i]
    return dp


# grad
eta = .0000001
error = rows * 10
stop = 1
while ((stop) > 0.0001):
    vect_diff = [0] * cols
    for j in range(0, rows, 1):
        if (trainlabels.get(str(j)) != None):
            dp = dotProduct(w, data[j])
            for k in range(0, cols, 1):
                vect_diff[k] += (trainlabels.get(str(j)) - dp) * data[j][k]

    # update w
    for j in range(0, cols, 1):
        w[j] = w[j] + eta * vect_diff[j]

    prev = error
    error = 0

    # err

    for i in range(0, rows, 1):
        if (trainlabels.get(str(i)) != None):
            # print(dotProduct(w,data[j]))
            error += (trainlabels.get(str(i)) - dotProduct(w, data[i])) ** 2
    print(error)
    if (prev > error):
        stop = prev - error
    else:
        stop = error - prev

# print("w= ")
normw = 0
for i in range(0, (cols - 1), 1):
    normw += w[i] ** 2
print("w ", w)

normw = math.sqrt(normw)

d_orgin = abs(w[len(w) - 1] / normw)

print("Distance to origin = " + str(d_orgin))

# predict
for i in range(0, rows, 1):
    if (trainlabels.get(str(i)) == None):
        dp = dotProduct(w, data[i])
        if (dp > 0):
            print("1 : " + str(i))
        else:
            print("0 : " + str(i))
