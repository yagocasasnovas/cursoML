#hw2_classification.py
from __future__ import division
import numpy as np
import sys
import math

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

## can make more functions if required

def set_number_of_x(xtrain):
	
	return len(xtrain[0])

def calculate_mean_std_dev(xtrain,ytrain):
	
	answer = {}
	
	all_classes = {}
	ytrain.astype(int)
	
	xtrain_list = xtrain.tolist()
	
	ytrain_list = ytrain.tolist()
	
	#print xtrain_list
	
	ytrain_unique = set(ytrain)

	ytrain_unique = list(ytrain_unique)

	ytrain_unique = map(int,ytrain_unique)

	for c in ytrain_unique:
		
		
		per_class = []
		
		
		for idx2, y in enumerate(ytrain_list):
			#print str(idx2) + " " + str(idx) + " " + str(c)
			if c == y:
				for idx3, x in enumerate(xtrain_list):
					if idx3 == idx2:
						per_class.append(x)
						
						#raw_input()
		all_classes[c] = per_class

	class_mean = {}
	
	for c in ytrain_unique:
		long1 = len(all_classes[c])
		
		
		means_dd = []
		for kk in range(set_number_of_x(X_train)):
			mean = 0
			mean_q = 0
			
			for q in all_classes[c]:
				
				mean_q = mean_q + q[kk]
			mean = mean_q / long1
			means_dd.append(mean)
		
		class_mean[c] = means_dd
	
	answer['mean'] = class_mean
	
	class_std_dev = {}
	
	for c in ytrain_unique:
		long2 = len(all_classes[c])
		standards_dd = []
		for jj in range(set_number_of_x(X_train)):
			std = 0
			std_q = 0
			for s in all_classes[c]:
				
				std_q = std_q + (q[jj] - class_mean[c][jj])*(q[jj] - class_mean[c][jj])
			std = np.sqrt(std_q/long2)
			standards_dd.append(std)

		class_std_dev[c] = standards_dd
		
	
	answer['standard'] = class_std_dev
	answer['unique'] = ytrain_unique
	
	return answer
	



def prob(x, mean, std):
	e = np.exp(-(x-mean)*(x-mean)/(2*std*std))
	return (1 / (np.sqrt(2*np.pi) * std)) * e



def pluginClassifier(X_train, y_train, X_test):
	# this function returns the required output 
	
	mean_std = calculate_mean_std_dev(X_train,y_train)
	
	final_outputs = []
	
	for x in X_test:
		final_vector = []
		for c in mean_std['unique']:
			pepe = 1
			for idx,k in enumerate(x):
				
				#print c
				#print k
				#print mean_std['mean'][c][idx]
				#print mean_std['standard'][c][idx]
				proba = prob(k,mean_std['mean'][c][idx],mean_std['standard'][c][idx])
				#print proba
				
				pepe = pepe * proba
				
				
				
			final_vector.append(pepe)
		
		final_outputs.append(final_vector)
	
		
	
	
	return final_outputs


final_outputs = pluginClassifier(X_train, y_train, X_test)

np.savetxt("probs_test.csv", final_outputs, delimiter=",")