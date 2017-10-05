# A skeleton for implementing Naive Bayes Classifier in Python.
# Author: Salem

import numpy
import scipy.stats as stats

trainingFile = "irisTraining.txt"
testingFile = "irisTesting.txt"
Xtrain = numpy.loadtxt(trainingFile)

# Sort training data by class
sortedArr = Xtrain[Xtrain[:, 4].argsort()]

# Make separate arrays for each class
index = 0
i = 0
while i < len(sortedArr):
    if sortedArr[i][4] == -1 and sortedArr[i + 1][4] == 1:
        index = i
    i += 1



# split sorted array into classes
neg = sortedArr[:index]
pos = sortedArr[index:]

#print(neg)
#print("\n\n ..................\n\n")
#print(pos)

# flip axis
neg = numpy.swapaxes(neg, 0, 1)
pos = numpy.swapaxes(pos, 0, 1)

n = Xtrain.shape[0]
d = Xtrain.shape[1] - 1

# Training... Collect mean and standard deviation for each dimension for each class..
# Also, calculate P(C+) and P(C-)

posMeans = []
posSTDs = []

negMeans = []
negSTDs = []

# calculate means for each attribute in positive matrix
posMean1 = numpy.mean(pos[0])
posMean2 = numpy.mean(pos[1])
posMean3 = numpy.mean(pos[2])
posMean4 = numpy.mean(pos[3])
posMeans.extend((posMean1, posMean2, posMean3, posMean4))

# calculate means for each attribute in negative matrix
negMean1 = numpy.mean(neg[0])
negMean2 = numpy.mean(neg[1])
negMean3 = numpy.mean(neg[2])
negMean4 = numpy.mean(neg[3])
negMeans.extend((negMean1, negMean2, negMean3, negMean4))

# calculate standard deviations for each attribute in positive matrix
posSTD1 = numpy.std(pos[0])
posSTD2 = numpy.std(pos[1])
posSTD3 = numpy.std(pos[2])
posSTD4 = numpy.std(pos[3])
posSTDs.extend((posSTD1, posSTD2, posSTD3, posSTD4))

# calculate standard deviations for each attribute in negative matrix
negSTD1 = numpy.std(neg[0])
negSTD2 = numpy.std(neg[1])
negSTD3 = numpy.std(neg[2])
negSTD4 = numpy.std(neg[3])
negSTDs.extend((negSTD1, negSTD2, negSTD3, negSTD4))

# calculate prior probability for classes
posPrior = len(pos) / len(Xtrain)
negPrior = len(neg) / len(Xtrain)

# Testing .....
Xtest = numpy.loadtxt(testingFile)
nn = Xtest.shape[0]  # Number of points in the testing data.

tp = 0  # True Positive
fp = 0  # False Positive
tn = 0  # True Negative
fn = 0  # False Negative

#x = 0
#y = 0
# iterate points in test data
for x in range(0, nn):
    # scores for probability of point falling in positive class and negative class
    posMult = 1.0
    negMult = 1.0
    # class: 1 = positive; -1 = negative
    result = 0
    # iterate attributes in test point
    for y in range(0, len(posMeans)):
        # joint probability of attributes
        posMult *= stats.norm.pdf(Xtest[x][y], posMeans[y], posSTDs[y])
        negMult *= stats.norm.pdf(Xtest[x][y], negMeans[y], negSTDs[y])
    # weight joint probability of attributes by prior probability of each class
    posMult *= posPrior
    negMult *= negPrior

    # our prediction for class of test point
    if posMult > negMult:
        result = 1
    else:
        result = -1

    # actual class of test point
    actual = Xtrain[x][4]

    # comparing our prediction to actual
    if result == 1 and actual == 1:
        tp += 1
    elif result == 1 and actual == -1:
        fp += 1
    elif result == -1 and actual == -1:
        tn += 1
    elif result == -1 and actual == 1:
        fn += 1

print("\nClassification Accuracy: " + str('{accuracy:.2%}'.format(accuracy=((tp + tn )/( tp + tn + fp + fn)))))


print("True Positive: " + str(tp))
print("True Negative: " + str(tn))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))

print("Classification Precision: " + str(tp / (tp + fp)))
print("Classification Recall: " + str(tp / (tp + fn)))