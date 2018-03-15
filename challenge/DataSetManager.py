# Author: WeldFire
# Created: 12/20/2016
from pprint import pprint
import tensorflow
import random
import numpy

"""
Creates a random list of IDs to be used in training and validation

IN:
data - dataset that you are wanting to train with
validationRation - the amount of validation data that you would like removed from your training set

OUT:
trainingDataIDs - A list of indicies in the original dataset to be used for training
validationDataIDs - An ordered list of indicies in the original dataset to be used for validation
"""
def _generateValidationandTrainingDataIDSets(data, validationRatio=0.1):
	dataLength = len(data)
	entriesUsedInValidation = int(dataLength * validationRatio)
	
	dataIDs = range(dataLength)
	
	#Get a random sample of data IDs to be used in validation
	validationDataIDs = random.sample(dataIDs, entriesUsedInValidation)
	#Remove the validation IDs from the overall pool of IDs
	trainingDataIDs = list(set(dataIDs)-set(validationDataIDs))
	
	return trainingDataIDs, validationDataIDs
	
	
"""
Normalizes the dataset provided

IN:
data - the data that you want to normalized

OUT:
normalizedData - the normalized data from the input provided
standardDeviation - the calculated standardDeviation to be reused optionally later
average - the calculated average to be reused optionally later
"""
def _normalizeData(data):
	dataArray = numpy.asarray(data, dtype=numpy.float32)
	
	standardDeviation = dataArray.std(axis=0)
	average = dataArray.mean(axis=0)
	
	normalizedData = (dataArray - average)/ (standardDeviation)
	
	return normalizedData, standardDeviation, average
	
"""
Normalizes the dataset provided using precomputed values

IN:
data - the data that you want to normalized
standardDeviation - the precalculated standardDeviation from a previous normalization
average - the precalculated average from a previous normalization

OUT:
normalizedData - the normalized data from the input provided
"""
def _precomputedDataNormalize(data, standardDeviation, average):
	dataArray = numpy.asarray(data, dtype=numpy.float32)
	
	normalizedData = (dataArray - average)/ (standardDeviation)
	
	return normalizedData
	
	
"""
Creates one hot representations for the array provided

IN:
data - the data that you want a one hot representation of
dataSize - the data size of the one hot representation

OUT:
oneHotData - the one hot data from the input provided
"""
def _oneHotData(data, dataSize):
	#Convert the provided array to a numpy array
	numpyDataArray = numpy.array(data).astype(dtype=numpy.uint8)

	#Convert the numpy array into a one hot matrix
	oneHotData = (numpy.arange(dataSize) == numpyDataArray[:, None]).astype(numpy.float32)
	
	return oneHotData

	
"""
Creates two array sets one set of training data and labels and one set of validation data and labels

IN:
data - the dataset that you are wanting to train with
labels - the label set that you are wanting to train your data on
validationRation - the amount of validation data that you would like removed from your training set

OUT:
trainingData - a list of normalized training data excluding validation data
trainingLabels - a list of normalized training labels excluding validation labels
validationData- a list of normalized validation data excluding training data
validationLabels - a list of normalized validation labels excluding training labels
"""
def generateDataSets(data, labels, validationRatio=0.1):
	#Create training data output placeholder variables
	trainingData = []
	trainingLabels = []
	#Create validation data output placeholder variables
	validationData = []
	validationLabels = []
	
	#Normalize our data
	normalizedData,std,avg = _normalizeData(data)
	#Get the Data IDs that we want to use for training and validation
	trainingDataIDs, validationDataIDs = \
		_generateValidationandTrainingDataIDSets(data, validationRatio)
	
	#Shuffle our accesses for randomness
	shuffledDataIndexes = range(len(normalizedData))
	random.shuffle(shuffledDataIndexes)
	
	#For every index in our normalizedData array we want to populate our lists
	for i in shuffledDataIndexes:
		#If the index is in our validation ID set 
		#add the corresponding data and label to their respective arrays
		if i in validationDataIDs:
			validationData.append(normalizedData[i])
			validationLabels.append(labels[i])
		else:
			#Else we add the data and label to the training set!
			trainingData.append(normalizedData[i])
			trainingLabels.append(labels[i])
		
	return trainingData, trainingLabels, validationData, validationLabels