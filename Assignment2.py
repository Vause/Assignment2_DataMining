import numpy
import scipy.stats as stats
import operator
import functools
import sys

trainingFile = sys.argv[1]  #To be able to run for any file
testingFile = sys.argv[2]	#To be able to run for any file
Xtrain = numpy.loadtxt(trainingFile)

def compareProbabilities(arr):
    if arr[0] > arr[1]: return 1
    else: return -1

# n = number of training points, d = dimensions of training points
d = Xtrain.shape[1] - 1

posFlag = Xtrain[:, d] > 0      # indices of all positive records
positive = Xtrain[posFlag, 0: d]     # subarray with just positive records
negFlag = Xtrain[:, d] < 0      # indices of negative records
negative = Xtrain[negFlag, 0: d]     # subarray with just negative records

# calculate means for each element in positive and negative matrix
posMean = numpy.mean(positive, axis=0)
negMean = numpy.mean(negative, axis=0)

#calculate standard deviations for each element in positive and negative matrix
posSD = numpy.std(positive, axis=0)
negSD = numpy.std(negative, axis=0)

# calculate prior probability for classes
priorProbPos = float(len(positive)) / len(Xtrain)
priorProbNeg = float(len(negative)) / len(Xtrain)

# Testing .....
Xtest = numpy.loadtxt(testingFile)
nn = Xtest.shape[0]  # Number of points in the testing data.

tp = fp = tn = fn = 0 #True Positive, False Positive, True Negative, False Negative

#Using stats.norm to calculate probability density function
pdfNormPos = stats.norm.pdf(Xtest[:, 0:d], posMean, posSD)
pdfNormNeg = stats.norm.pdf(Xtest[:, 0:d], negMean, negSD)

#Multiplying the positive and negative prior probabilities by each element in each of the matrices
posPredict = [functools.reduce(operator.mul, x, priorProbPos) for x in pdfNormPos]
negPredict = [functools.reduce(operator.mul, x, priorProbNeg) for x in pdfNormNeg]

#Combining matrices by each element, and converting to a list
bothPredict = list(zip(posPredict, negPredict))

#Comparing each element to determine if tp, tn, fp, or fn
results = list(zip(map(compareProbabilities, bothPredict), Xtest[:,d]))

#Setting values to tp, tn, fp and fn
for x,y in results:
    tp +=1 if x == 1 and y == 1 else 0
    tn +=1 if x == -1 and y == -1 else 0
    fp +=1 if x ==1 and y == -1 else 0
    fn +=1 if x == -1 and y == 1 else 0

print('Accuracy: ' + str(float(tp + tn) / (len(results))) + '\n' + 
'True Positive Count: ' + str(tp) + '\n' + 
'True Negative Count: ' + str(tn) + '\n' + 
'False Positive Count: ' + str(fp) + '\n' + 
'False Negative Count: ' + str(fn) + '\n' + 
'Precision: ' + str(float(tp)/(tp+fp)) + '\n' + 
'Recall: ' + str(float(tp)/(tp+fn)))