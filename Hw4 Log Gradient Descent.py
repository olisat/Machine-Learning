
import sys
import random
import math

datafile = sys.argv[1]

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
# print(data)


f.close()
# print(data)
###############################
##### read training labels ####
###############################

labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = [0, 0]
l = f.readline()
while (l != ''):
    a = l.split()
    trainlabels[a[1]] = int(a[0])
    #    trainlabels_size[a[0]] = trainlabels_size[a[0]]+1
    if (trainlabels[a[1]] == 0):
        '''trainlabels[a[1]] = -1;'''
    l = f.readline()

    n[int(a[0])] += 1

f.close()


########## initialize w ##########
# print("this is new code")
w = []
for j in range(0, cols, 1):
    # print(random.random())
    w.append(0.02 * random.random() - 0.01)


# print(w)

#### calculation of doc product #####

def dot_product(a, b):
    dp = 0
    for i in range(0, cols, 1):
        dp += a[i] * b[i]
    return dp


loss = rows * 10
vect_diff = 1
count = 0

stop = 0.001
eta = .001

while ((vect_diff) > stop):
    vect_diff = [0] * cols
    for j in range(0, rows, 1):
        if (trainlabels.get(str(j)) != None):
            dp = dot_product(w, data[j])
            risk = (trainlabels.get(str(j))) - (1 / (1 + (math.exp(-1 * dp))))
            for k in range(0, cols, 1):
                vect_diff[k] += (risk) * data[j][k]
#update w
    for j in range(0, cols, 1):
        w[j] = w[j] + eta * vect_diff[j]
    prev = loss
    loss = 0

#loss
    for j in range(0, rows, 1):
        if (trainlabels.get(str(j)) != None):
            loss += math.log(1 + math.exp((-1 * (trainlabels.get(str(j)))) * (dot_product(w, data[j]))))
        vect_diff = abs(prev - loss)

normw = 0
for i in range(0, (cols - 1), 1):
    normw += w[i] ** 2
print ("w ", w)


normw = math.sqrt(normw)
# print("sqrt")
print("||w||=" + str(normw))
# print("")

d_orgin = (w[len(w) - 1] / normw)

print ("Distance to origin = " + str(d_orgin))

#predict
for i in range(0, rows, 1):
    if (trainlabels.get(str(i)) == None):
        dp = dot_product(w, data[i])
        if (dp > 0):
            print("Class 1 " + str(i))
        else:
            print("Class 0 " + str(i))
            