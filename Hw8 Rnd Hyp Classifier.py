import sys
import math
import random
import numpy
from sklearn.model_selection import cross_val_score
from sklearn import svm


# dot_product function

def dot_product(w, x):
    cols = len(w)
    dp = 0
    for j in range(cols):
        dp += w[j] * x[j]
    return dp


#####reading from data file and label file.

def getdata():
    datafile=sys.argv[1]
    f = open(datafile, 'r')
    data = []
    l = f.readline()

    while (l != ''):
        a = l.split()
        l2 = []
        for j in range(len(a)):
            l2.append(float(a[j]))
        data.append(l2)
        l = f.readline()

    rows = len(data)
    cols = len(data[0])
    f.close()
    return data

def getlabels():
    trainlabelfile=sys.argv[2]
    f = open(trainlabelfile, 'r')
    trainlabels = {}
    i = 0
    l = f.readline()
    while (l != ''):
        a = l.split()
        trainlabels[int(a[1])] = int(a[0])
        l = f.readline()
    f.close()
    return trainlabels


def gettestdata():
    testdatafile = sys.argv[3]
    f = open(testdatafile, 'r')
    testdata = []

    l = f.readline()

    while (l != ''):
        a = l.split()
        l2 = []
        for j in range(len(a)):
            l2.append(float(a[j]))
        testdata.append(l2)
        l = f.readline()

    rows = len(testdata)
    cols = len(testdata[0])
    f.close()
    return testdata


def compute(data, labels, k):
    rows = len(data)
    cols = len(data[0])



    c_list = numpy.array([0.001,.1,.5,1,100])
    prev_err =100000
    prev_c = 1000000
    prev_zerr = 0
    for f in c_list:
        Z = []
        z_prime = []
        curr_c = f

        clf = svm.LinearSVC(C=curr_c, max_iter=100000)
        for j in range(k):
            # initialize w
            w = [0] * cols
            for j in range(cols):
                w[j] = round(random.uniform(-1, 1), 2)

            z_col = []

            for data_point in data: #new labels using random hyperplane
                dp_data = dot_product(w, data_point)
                if dp_data <= 0:
                    z_col.append(0)
                else:
                    z_col.append(1)
            Z.append(z_col)
        Z = list(zip(*Z))

        data_score = cross_val_score(clf, data, list(labels.values()), cv=2)
        data_err = round(1 - (sum(data_score) / len(data_score)), 2)

      #  print("For k =", k)
       # print('Error on original data:', data_err, ' ')

        Z_score = cross_val_score(clf, Z, list(labels.values()), cv=2)
        Z_error = round(1 - (sum(Z_score) / len(Z_score)), 2)
       # print('error on new feature data=', Z_error)
        if data_err < prev_err:
            prev_err = data_err
            prev_zerr = Z_error
            prev_c = f
        if f == 100:
            print("For k =", k)
            print("Best C:", prev_c)
            print('Error on original data:', prev_err, ' ')
            print('Error on new data=', prev_zerr)

    return (clf, Z)


def predict(clf, newfeaturedata, labels, testdata):
    clf.fit(newfeaturedata, list(labels.values()))
    rows = len(data)
    cols = len(data[0])
    z_col = []
    z_prime = []
    for j in range(k):
        # initialize w
        w = [0] * cols
        for j in range(cols):
            w[j] = round(random.uniform(-1, 1), 2)
        z_primecol = []

        for data_point in newfeaturedata:
            dp_data = dot_product(w, data_point)
            if dp_data <= 0:
                z_primecol.append(0)
            else:
                z_primecol.append(1)
        z_prime.append(z_primecol)
    z_prime = list(zip(*z_prime))

    test_predictions = clf.predict(z_prime)
    print('Prediction on new data:', test_predictions)

if __name__ == '__main__':
    data = getdata()
    labels = getlabels()
    k_list = {0 : 10, 1 : 100, 2 : 1000, 3 : 10000}
    for i in k_list.keys():
        k = k_list.get(i)
        results = compute(data, labels, k)

    clf = results[0]
    new_data = results[1]
    if len(sys.argv) == 4:
        testdata = gettestdata()
        predict(clf, new_data, labels, testdata)





















