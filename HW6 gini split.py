import sys
from sys import argv
import random
import math

data = sys.argv[1]
labels = sys.argv[2]
### Read Data ###

f = open(data, 'r')
data = []
l = f.readline()
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


##### read training labels ####

trainlabels = {}
f = open(labels, 'r')
l = f.readline()
numclass = []
numclass.append(0)
numclass.append(0)
while (l != ''):
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    numclass[int(a[0])] = numclass[int(a[0])] + 1
    l = f.readline()
f.close()


##Iteration##
GiniVal = []
split = 0
l3 = [0, 0]
for j in range(0, data_cols, 1):
    GiniVal.append(l3)
temp_val = 0
col = 0

for i in range(0, data_cols, 1):

    listcol = [item[i] for item in data]
    keys = sorted(range(len(listcol)), key=lambda k: listcol[k])
    listcol.sort()

    gini_values = []
    prev_gini = 0
    prev_row = 0
    for k in range(1, data_rows, 1):
        lsize = k
        rsize = data_rows - k
        lp = 0
        rp = 0
        for l in range(0, k, 1):
            if (trainlabels.get(str(l)) != None and trainlabels[keys[l]] == 0):
                lp += 1
        for r in range(k, data_rows, 1):
            if (trainlabels.get(str(k)) != None and trainlabels[keys[k]] == 0):
                rp += 1
        gini = (lsize / data_rows) * (lp / lsize) * (1 - lp / lsize) + (rsize / data_rows) * (rp / rsize) * (
            1 - rp / rsize)
        gini_values.append(gini)

        prev_gini = min(gini_values)
        if (gini_values[k - 1] == float(prev_gini)):
            GiniVal[i][0] = gini_values[k - 1]
            GiniVal[i][1] = k

    if (i == 0):
        temp_val = GiniVal[i][0]
    if (GiniVal[i][0] <= temp_val):
        temp_val = GiniVal[i][0]
        col = i
        split = GiniVal[i][1]
        if (split != 0):
            split = (listcol[split] + listcol[split - 1]) / 2
print("Column:", col, "Split:", split)