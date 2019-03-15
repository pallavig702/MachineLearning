#Implementing KNN in python 3 from scratch. Test and train data set is provided already. We don't need a split in this case. This is done in case of MNIST dataset

#!/usr/bin/python

import csv
import sys, getopt
import math
from math import sqrt
import operator
import random


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for i in range(length):
		i1=int(instance1[i])
		i2=int(instance2[i])
		distance+=pow((i1-i2),2)
	return(sqrt(distance))

def getNeighbors(trainingSet,testInstance,k):
	dist=[]
	distances=[]
	length = len(testInstance)
	for i in range(length-1):
		if(length-1 == len(trainingSet[i])-1):
			dist=euclideanDistance(testInstance,trainingSet[i],length)
			distances.append((trainingSet[i],dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)-1):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0



def main():
	print("hi")
	trainingSet=[]
	testSet=[]
	predictions=[]
	trainReader = csv.reader(open(sys.argv[1]), delimiter=',')
	for x in trainReader:
		line = list(x)
		newlist=[]
		for y in line:		
			newlist.append(float(y))
		trainingSet.append(newlist)

	testReader = csv.reader(open(sys.argv[2]), delimiter=',')
	for row in testReader:
		line2 = list(row)
		newlist2=[]
		for yt in line2: 
			newlist2.append(float(yt))
		testSet.append(newlist2)

	k=3
	for x in range(len(testSet)):
		neighbors=getNeighbors(trainingSet,testSet[x],k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')


main()
