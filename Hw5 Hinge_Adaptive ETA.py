import sys
import random
import math

data_file = sys.argv[1]

f = open(data_file, 'r')
data = []
# i = 0
l = f.readline()

#read data

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

# read labels
labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = [0, 0]
l = f.readline()
while (l != ''):
    a = l.split()
    trainlabels[a[1]] = int(a[0])
    if (trainlabels[a[1]] == 0):
        trainlabels[a[1]] = -1;
    l = f.readline()

    n[int(a[0])] += 1

f.close()

w = []
for j in range(0, cols, 1):
    w.append(0.02 * random.random() - 0.01)

def dotProduct(a, b):
    dp = 0
    for i in range(0, cols, 1):
        dp += a[i] * b[i]
    return dp



#iteration (gradient descent)
loss = rows * 10
Current_error = rows * 10
stop = 1
best_eta = None
while ((stop) > 0.001):
    vect_diff = [0] * cols
    for j in range(0, rows, 1):
        if (trainlabels.get(str(j)) != None):
            dp = dotProduct(w, data[j])
            risk = (trainlabels.get(str(j)) * (dotProduct(w, data[j])))
            for k in range(0, cols, 1):
                if (risk < 1):
                    vect_diff[k] += -1 * ((trainlabels.get(str(j))) * data[j][k])
                else:
                    vect_diff[k] += 0
    eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001]
    bestobj = None

    for k in range(0, len(eta_list), 1):
        curr_eta = eta_list[k]
        for m in range(len(data[0])):
            w[m] -= curr_eta * vect_diff[m]
            # prev_err = Current_error
            # Current_error = 0
        obj = 0.0
        for j in range(0, rows, 1):
            if (trainlabels.get(str(j)) != None):
                obj += max(0, 1 - (trainlabels.get(str(j)) * dotProduct(w, data[j])))
        if bestobj == None or obj < bestobj:
            bestobj = obj
            best_eta = curr_eta
        for m in range(len(data[0])):
            w[m] += curr_eta * vect_diff[m]

       # print(eta, str(obj))


    #update w
    for i in range(0, cols, 1):
        w[i] = w[i] - best_eta * vect_diff[i]
    prev = loss
    loss = 0

#loss
    for j in range(0, rows, 1):
        if (trainlabels.get(str(j)) != None):
            loss += max(0, 1 - (trainlabels.get(str(j)) * dotProduct(w, data[j])))
        stop = abs(prev - loss)
print("eta:" + str(best_eta))


normw = 0
for i in range(0, (cols - 1), 1):
    normw += w[i] ** 2
print ("w: ",w)

normw = math.sqrt(normw)
d_orgin = abs(w[len(w) - 1] / normw)
print ("Distance to origin = " + str(d_orgin))

#predict
for i in range(0, rows, 1):
    if (trainlabels.get(str(i)) == None):
        dp = dotProduct(w, data[i])
        if (dp > 0):
            print("1 " + str(i))
        else:
            print("0 " + str(i))
