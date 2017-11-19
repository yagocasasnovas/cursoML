#hw2_classification.py
from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

## can make more functions if required

def set_number_of_x(xtrain):
	
	return len(xtrain[0])

def calculate_mean_std_dev(xtrain,ytrain):
	
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
	
	class_std_dev = {}
	
	for c in ytrain_unique:
		long2 = len(all_classes[c])
		standards_dd = []
		for jj in range(set_number_of_x(X_train)):
			std = 0
			std_q = 0
			for s in all_classes[c]:
				
				std_q = std_q + (q[jj] - )
		
calculate_mean_std_dev(X_train,y_train)



def pluginClassifier(X_train, y_train, X_test):
	# this function returns the required output 
	
	
	pass


final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

#np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file