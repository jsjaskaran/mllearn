# Author: WeldFire
# Created: 12/20/2016
from pprint import pprint
import tensorflow as tf
import DataSetManager
import FileProcessor

normalizationStdDev = 0.5
normalizationAvg = 300
labelKVP = {}

"""
Generates all of the layers for a tensor

IN:
inputLayer - the tf placeholder defined as your input layer
outputSize - number of nodes (categories) in the output
hiddenLayers - number of nodes for each of the hidden layers

OUT:
out - the output layer
fc1 - fully connected hidden layer 1
fc2 - fully connected hidden layer 2
"""
def _generateTensorLayers(inputLayer, outputSize, hiddenLayers = [256, 256]):
	fc1 = tf.contrib.layers.fully_connected(inputLayer, hiddenLayers[0], activation_fn=tf.nn.relu)
	fc2 = tf.contrib.layers.fully_connected(fc1, hiddenLayers[1], activation_fn=tf.nn.relu)
	out = tf.contrib.layers.fully_connected(fc2, outputSize, activation_fn=None)
	
	return out, fc1, fc2

"""
Gets user input to evaluate the trained tensor

IN:
NONE

OUT:
data - The pokemon stat array from the user
shouldStop - if the user wants to stop
"""
def _getUserInput():
	print("")
	print("Please enter pokemon stats in the following order:")
	print("HP Attack Defense Sp. Atk Sp. Def Speed Generation")
	
	try:
		data = [int(x) for x in raw_input().split()]
		shouldStop = (data[0] == -1)
	except ValueError:		
		print("-------------------------------------------------")
		print("Your input didn't follow the correct standard, please try again!")
		print("-------------------------------------------------")
		data, shouldStop = _getUserInput()
	except IndexError:
		data = []
		shouldStop = True
	
	return data, shouldStop
	
"""
Trains the generated tensor based on the input provided, 
then validates the tensor for accuracy, lastly it asks for user input

IN:
trainingData - The data to train on, this data should already be normalized
trainingLabels - The data labels to train on, the labels will be converted to one hot format
validationData - The data to validate with, this data should already be normalized
validationLabels - The data labels to validate with, the labels will be converted to one hot format
trainingIterations - The number of iterations to train 
learningRate - The learning rate in which to train with

OUT:
None
"""
def trainTensor(trainingData, trainingLabels, validationData, validationLabels, trainingIterations = 120, learningRate = 0.0005):
	#Calculate our input and output neural sizes
	outputSize = max([max(trainingLabels), max(validationLabels)])+1
	inputSize = len(trainingData[0])

	#Print out some statistics about our data 
	#print("input size " + str(inputSize) + " output size: " + str(outputSize))
	#print("train_data length: " + str(len(trainingData)))
	#print("train_data width: " + str(len(trainingData[0])))
	#
	#print("train_labels length: " + str(len(trainingLabels)))
	#print("train_labels width: 1")# + str(len(trainingLabels[0])))
	#
	#print("Validation data length: " + str(len(validationData)))
	#print("Validation data width: " + str(len(validationData[0])))
	#
	#print("Validation labels length: " + str(len(validationLabels)))
	#print("Validation labels width: 1")# + str(len(validationLabels[0])))
	
	#Creates one hot representations for both label sets
	trainingLabels = DataSetManager._oneHotData(trainingLabels, outputSize)
	validationLabels = DataSetManager._oneHotData(validationLabels, outputSize)
	
	#Generate our tensor layers
	hiddenLayers = {}
	X = tf.placeholder(tf.float32, [None, inputSize], name='X')
	Y = tf.placeholder(tf.float32, [None, outputSize], name='Y')
	predictor, hiddenLayers['fc1'], hiddenLayers['fc2'] = _generateTensorLayers(X, outputSize)

	#Create our cost and optimizing functions
	costFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictor, Y))
	optimizingFunction = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(costFunction)
	
	#Init saver class
	saver = tf.train.Saver()
	
	#Init Tensorflow
	init = tf.global_variables_initializer()
	
	#Let the user know what we are training with
	print("We are training with " + str(trainingIterations) + " iterations at a learing rate of " + "{:.2}".format(learningRate))
	
	#Generate our Tensorflow session
	with tf.Session() as session:
		session.run(init)
		
		for iteration in range(trainingIterations):
			losses = 0
			
			for step in range(len(trainingData)):
				#Get Tensorflow to execute our neural net
				_, costReturn, fc1, fc2, out = session.run([optimizingFunction,costFunction, hiddenLayers['fc1'], hiddenLayers['fc2'], predictor], feed_dict={X:[trainingData[step]], Y:[trainingLabels[step]]}) 
				#Keep track of all of our costs
				losses = losses + costReturn
				
				#if step == 0: #Ability to debug as training progresses
				#	print(fc1[0].mean(), fc2[0].mean(), out[0])
					
			#Calculate the average cost
			lossAverage = (losses/len(trainingData))
			
			print("Iteration: " + str(iteration) + " Average Cost: " + "{:.2%}".format(lossAverage))
			
		#Test the trained data
		#If the answer is in the top 5 then we consider it a "win"
		top5Test = tf.nn.in_top_k(predictor, tf.cast(tf.argmax(Y,1), "int32"), 5)
		reducedMeanAccuracy = tf.reduce_mean(tf.cast(top5Test, "float"))
		validationAccuracy = reducedMeanAccuracy.eval({X:validationData, Y:validationLabels})
		
		print("-------------------------------------------------")
		print("Average Prediction Accuracy: " + "{:.2%}".format(validationAccuracy))
		print("-------------------------------------------------")
		
		#Save our trained model to be loaded later if desired
		saver.save(session, 'trainedModel')
		
		#Ask for user if they would like to enter their own data
		print("The network has finished training!")
		print("-------------------------------------------------")
		print("If you would like you can input your own entries against the nueral net!")
		print("Bulbasaur, a Type 1 Grass Pokemon, has the input '45 49 49 65 65 45 1'")
		print("Input -1 for the HP to quit")
		print("-------------------------------------------------")
		
		while (True):
			#Get the users input
			data, shouldStop = _getUserInput()
			
			if(shouldStop):
				#User requested that we stop, so we will leave the while loop
				break
			else:
				#Create an prediction evaluator
				predict = tf.argmax(predictor, 1)
				#Normalize the users first 7 entries with the normalization values we received from normalizing our training data
				oneShotNormalizedData = DataSetManager._precomputedDataNormalize([data[:7]], normalizationStdDev, normalizationAvg)
				
				#Calculate our prediction
				pred = predict.eval({X: oneShotNormalizedData})
				
				#Display the prediction to the user
				print("-------------------------------------------------")
				print("The neural network predicts that your input is of type:")
				#Convert the type ID back to the type 1 name 
				print(FileProcessor.typeFromTrackedTypeID(pred[0], labelKVP))
				print("-------------------------------------------------")