"""
Estimate physical activity intensity level according to heart rate utilizing KNN classifier
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt 

# classify activities to three levels of physical intensity according to MET
# light effort(label 1): lying, sitting, standing, ironing(1, 2, 3, 17)
# moderate effort(label 2): vaccum cleaning, descending stairs, walking, nordic walk, cycling(16, 13, 4, 7, 6)
# vigorous effort(label 3): ascending stairs, running, rope jumping(12, 5, 24)
def groupLabels(matrix):
	for i in range(0, np.shape(matrix)[0]):
		if matrix[i, 1] == 1 or matrix[i, 1] == 2 or matrix[i, 1] == 3 or matrix[i, 1] == 17:
			matrix[i, 1] = 1
		elif matrix[i, 1] == 16 or matrix[i, 1] == 13 or matrix[i, 1] == 4 or matrix[i, 1] == 7 or matrix[i, 1] == 6:
			matrix[i, 1] = 2
		elif matrix[i, 1] == 12 or matrix[i, 1] == 5 or matrix[i, 1] == 24:
			matrix[i, 1] = 3
		else:
			matrix[i, 1] = -1

# K: the selected parameter for KNN
# matrix: a matrix of timestamps, intensity label, heartRate
# heartRate: the heart rate to be classified
# return value: predicted intensity lable
def KNNClassifier(K, matrix, heartRate):
	difference = abs(matrix[:, -1] - heartRate)
	indice = np.argsort(difference)		# indice of sorted arrays based on the difference from the given heartrate
	kNearest = []
	# generate an array of k nearest labels
	for i in range(0, K):
		kNearest.append(matrix[indice[i], 1])
	mode, count = stats.mode(kNearest)
	return mode[0]

# K: K of KNN classifier
# train: training dataset
# test: test dataset
# return value: error rate
def errorRate(K, train, test):
	errors = 0
	for i in range(0, np.shape(test)[0]):
		label = KNNClassifier(K, train, test[i, 2])
		if label != test[i, 1]:
			errors = errors + 1
	return errors / np.shape(test)[0]

# K: K of KNN classifier
# fold: define the K-fold cross-calidation
# matrix: datasets
# return value: the mean error rate
def crossValidation(K, fold, matrix):
	interval = np.floor(np.shape(matrix)[0] / fold)
	errorRates = []
	for i in range(0, fold):
		test = matrix[int(i*interval): int(i*interval+interval), :]
		train = np.concatenate((matrix[0: int(i*interval), :], matrix[int(i*interval+interval):, :]))
		err = errorRate(K, train, test)
		errorRates.append(err)
	return np.sum(errorRates) / fold

# pick up an optimal k of KNN based pn cross validation error rate
# Ks: candidates of K for KNN
# matrix: all datasets
# return value: 
# 	Ks[indice[0]]: the optimal K; errRates[indice[0]]: the optimal error rate; errRates: array of cross validation error rates
def chooseK(Ks, matrix):
	errRates = []
	for k in Ks:
		errRates.append(crossValidation(k, 10, matrix))
	indice = np.argsort(errRates)
	return (Ks[indice[0]], errRates[indice[0]], errRates)


if __name__ == '__main__':
	X = np.genfromtxt('data.csv', delimiter=',')
	np.random.shuffle(X)	# shuffle the dataset for better cross validation performance
	print(np.shape(X))
	groupLabels(X)
	Ks = np.arange(0, 305, 5)
	Ks[0] = 1
	K, errRate, errRates = chooseK(Ks, X)
	print(Ks)
	print(K, errRate, errRates)

	plt.xlabel('K')
	plt.ylabel('Error Rate')
	plt.title('10-fold Cross Validation Error Rates of KNN')
	plt.plot(Ks[2:], errRates[2:], 'bo-')
	plt.show()

