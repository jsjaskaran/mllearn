# Author: WeldFire
# Created: 12/20/2016
from pprint import pprint
import DataSetManager
import FileProcessor
import tensorflow
import TFManager

tensorflow.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
tensorflow.app.flags.DEFINE_integer('trainingIterations', 120, 'Number of iterations to train the network for.')
tensorflow.app.flags.DEFINE_float('learningRate', 0.0005, 'The learning rate for the training session.')
FLAGS = tensorflow.app.flags.FLAGS

# Be verbose?
verbose = FLAGS.verbose
trainingIterations = FLAGS.trainingIterations
learningRate = FLAGS.learningRate

"""
The main function of this program!
"""
def main():	
	#Load the data from our pokemon dataset
	data, labels, TFManager.labelKVP = FileProcessor.loadCleanData('pokemon.csv')
	
	#Get a normalized reference for user input later
	#(Not actually needed for training as it will be called in the DataSetManager)
	normalizedData, TFManager.normalizationStdDev, TFManager.normalizationAvg = \
		DataSetManager._normalizeData(data)
	
	#Seperate and normalize our datasets
	trainingData, trainingLabels, validationData, validationLabels = \
		DataSetManager.generateDataSets(data, labels)
	
	#Execute our tensor to learn and predict!
	TFManager.trainTensor(trainingData, trainingLabels, validationData, validationLabels, trainingIterations, learningRate)

"""
The not as pretty but verbose main function of this program!
"""
def verboseMain():
	#Load the data from our pokemon dataset
	data, labels, TFManager.labelKVP = FileProcessor.loadCleanData('pokemon.csv')
	
	print('All data parsed:')
	pprint(data)
	print('')
	print('All labels parsed:')
	print(labels)
	print('')
	print('')

	#Get a normalized reference for user input later
	#(Not actually needed for training as it will be called in the DataSetManager)
	normalizedData, TFManager.normalizationStdDev, TFManager.normalizationAvg = \
		DataSetManager._normalizeData(data)
	print('Normalized data:')
	pprint(normalizedData)
	print('')
	print('')
	
	trainingDataIDs, validationDataIDs = \
		DataSetManager._generateValidationandTrainingDataIDSets(data)
	#print('The random sample we are using to train with:')
	#print(trainingDataIDs)
	#print('')
	print('The random sample we are using to validate with:')
	print(sorted(validationDataIDs))
	print('')
	print('')
	
	#Seperate and normalize our datasets
	print('The training and validation lists seperated:')
	trainingData, trainingLabels, validationData, validationLabels = \
		DataSetManager.generateDataSets(data, labels)
	
	print('TrainingData:')
	pprint(trainingData)	
	print('TrainingLabels:')
	print(trainingLabels)
	print('ValidationData:')
	pprint(validationData)
	print('ValidationLabels:')
	print(validationLabels)
	print('')
	print('')
	
	print('Validation one hot label data:')
	print(DataSetManager._oneHotData(validationLabels, 19).tolist())
	print('')
	print('')
	
	#print('Test one hot label data:')
	#testLabel = [0,1,2,3,4,5,6,7,8,9]
	#print(DataSetManager._oneHotData(testLabel, 10).tolist())
	#print('')
	#print('')
	
	#Execute our tensor to learn and predict!
	TFManager.trainTensor(trainingData, trainingLabels, validationData, validationLabels, trainingIterations, learningRate)
	

"""
Grab the Python main method hook
"""
if __name__ == '__main__':
	if (verbose):
		verboseMain()
	else:
		main()	
			
	print("-------------------------------------------------")
	print("                                  (\_/)")
	print("Goodbye, thanks for stopping bye! (^-^)/)")
	print("-------------------------------------------------")