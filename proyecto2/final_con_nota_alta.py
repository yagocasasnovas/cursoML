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
	
	d = set_number_of_x(X_train)
	
	
	
	#MLE
	
	
	all_classes = {}
	ytrain.astype(int)
	
	xtrain_list = xtrain.tolist()
	#print len(xtrain_list)
	
	ytrain_list = ytrain.tolist()
	
	instances = len(ytrain_list)
	#print instances
	
	#print xtrain_list
	
	ytrain_unique = set(ytrain)

	ytrain_unique = list(ytrain_unique)

	ytrain_unique = map(int,ytrain_unique)

	#MLE
	mle = {}
	for c in ytrain_unique:
		cnt = 0
		for y in ytrain_list:
			if y == c:
				cnt = cnt + 1
		
		mle[c] = cnt/instances
		
	

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
		
		#print all_classes[c]
		#raw_input()
		
		rr = np.zeros([d,d])
		

		
		for x in all_classes[c]:
			
			x_np = np.asarray(x)
			mean_np = np.asarray(answer['mean'][c])
			toro = x_np - mean_np
			
			rr = rr + np.outer(toro,toro)
		
		
		rr = rr / long1
		
		class_std_dev[c] = rr
		
	
	
	answer['standard'] = class_std_dev
	
	
	
	answer['unique'] = ytrain_unique
	
	answer['mle'] = mle
	

	
	return answer
	



def prob(x, mean, std):
	
	inversecov = np.linalg.inv(std)
	toro = x - mean
	ff = np.dot(inversecov,toro)
	
	js = np.dot(toro,ff)
	
	js = (-1) * js / 2
	
	#print js
	ee = np.exp(js)
	#print ee
	determ = np.linalg.det(std)
	#print determ
	determ1 = np.power(determ,-1/2)
	#print determ1
	out = determ1 * mle * ee
	#print out
	#raw_input()
	return out
	
	#e = np.exp(-(x-mean)*(x-mean)/(2*std*std))
	#return (1 / (np.sqrt(2*np.pi) * std*std)) * e



def pluginClassifier(X_train, y_train, X_test):
	# this function returns the required output 
	
	mean_std = calculate_mean_std_dev(X_train,y_train)
	

	
	final_outputs = []
	
	for x in X_test:
		final_vector = []
		for c in mean_std['unique']:
			

			
			proba = prob(x,mean_std['mean'][c],mean_std['standard'][c],mean_std['mle'][c])


			#pepe = pepe * proba
			final_vector.append(proba)
		
		norma = np.linalg.norm(final_vector)
		final_vector = final_vector / norma
		
		final_outputs.append(final_vector)
	
		
	
	
	return final_outputs


final_outputs = pluginClassifier(X_train, y_train, X_test)

np.savetxt("probs_test.csv", final_outputs, delimiter=",")