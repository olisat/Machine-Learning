
import random
import time
import sys
from sklearn.svm import LinearSVC



####### reading data from args ########

def readfile(x):
    data = []
    c = 0
    try:
        file = sys.argv[x]
    except IndexError:
        print()
        sys.exit()

    with open(file) as datafile:
        for line in datafile:
            c += 1
            if(c % 1 == 0):
                data.append([int(l) for l in line.split()])
        return data

 ### Feature selection ###

def selectFeatures(data, labels):

    linearSVC = LinearSVC(C=.0289, penalty='l1', dual=False).fit(data, [x[0] for x in labels])
    score = linearSVC.coef_
    feature = []
    feature_cols = []
    for k in range(len(score[0])):
        if (score[0][k] != 0.0):
            feature_cols.append(int(k))
            rowdata = []
            for i in range(len(data)):
                rowdata.append(data[i][k])
            feature.append(rowdata)
    bestfeatures_data = [list(map(float, x)) for x in zip(*feature)]
    # ------ writing selected feature data to file ------

    #bestfeaturesdata_file = open('/Users/okechukwu/PycharmProjects/Hw1/project_Best_features_data', 'w')

    print("Number of Features selected from traindata:", len(bestfeatures_data[0]))

    #for i in range(0, len(bestfeatures_data), 1):
        #for j in range(0, len(bestfeatures_data[i]), 1):
            #bestfeaturesdata_file.write(str(int(bestfeatures_data[i][j])) + " ")
        #bestfeaturesdata_file.write('\n')

    #bestfeaturesdata_file.close()

    return bestfeatures_data, feature_cols

##### best feature for test data ######

def getDataBestFT(testdata, feature_cols):

    bestfeature_testdata = []
    totFeatures = len(feature_cols)

    for rows in range(len(testdata)):
        col = []
        k = 0
        for cols in range(len(testdata[0])):  #return data set of test data, with only the best features
            if (k < totFeatures and feature_cols[k] == cols): #append only columns of best features
                col.append(testdata[rows][cols])
                k += 1
        bestfeature_testdata.append(col)

    print("Feature Columns:", feature_cols)

    #bestfeaturestestdata_file = open('/Users/okechukwu/PycharmProjects/Hw1/project_Best_features_testdata', 'w')
    print("Number of Features selected in testdata:", len(bestfeature_testdata[0]))

    #for i in range(0, len(bestfeature_testdata), 1):
        #for j in range(0, len(bestfeature_testdata[i]), 1):
            #bestfeaturestestdata_file.write(str(bestfeature_testdata[i][j]) + " ")
        #bestfeaturestestdata_file.write('\n')

    #bestfeaturestestdata_file.close()

    return bestfeature_testdata

##### classify data ####

def predictLabels(traindata, trainlabels, testdata):
    print("Predicted Data (Classified)", sep='', end = '', flush = True)
    print()
#----- classification ----

    clf = LinearSVC(max_iter=15000000, tol=0.00000001).fit(traindata, [x[0] for x in trainlabels])

    # ------- data prediction------
    predictedlabels = clf.predict((testdata))

    # printing data -------

    #predictedlabels_file = open('/Users/okechukwu/PycharmProjects/Hw1/project_predictedlabels', 'w')

    for i in range(0, len(testdata), 1):
        print(str(predictedlabels[i])+" "+str(i))
        #predictedlabels_file.write(str(predictedlabels[i]) + " " + str(i) + '\n')

    #predictedlabels_file.close()
    return True

def main():
    print("Please ensure file arguements are in the following order: Traindata, TrainLabels, TestData")
    timestart = time.time()
    traindata = readfile(1)
    trainlabels = readfile(2)
    testdata = readfile(3)

    bestfeatures_data, feature_cols = selectFeatures(traindata, trainlabels)  # best features from data
    bestfeature_testdata = getDataBestFT(testdata, feature_cols)    # get data with best features
    ispredicted = predictLabels(bestfeatures_data, trainlabels, bestfeature_testdata)

    if (ispredicted):
        print("#Prediction Successful!!")

    time_elasped = (time.time() - timestart)
    print("Total Run Time: ", int(time_elasped / 60),"Minutes", int(time_elasped % 60), "Seconds")


main()
