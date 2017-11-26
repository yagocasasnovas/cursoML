#proyecto3

#5 clusters

#10 iterations

from __future__ import division
import numpy as np
import sys
import math
from random import randint
import copy
import csv
from scipy.stats import multivariate_normal
#import time
#start_time = time.time()

def prob(x, mean, std, size):
	
	inversecov = np.linalg.inv(std)
	
	toro = x - mean

	
	ff = np.dot(inversecov,toro)

	
	js = np.dot(toro,ff)
	
	js = (-1) * js / 2

	ee = np.exp(js)

	determ = np.linalg.det(std)
	#print determ
	determ1 = np.power(determ,1/2)
	#print determ1
	determ2 = np.power(2 * np.pi,size/2)
	determ3 = 1 / (determ1*determ2)
	out = determ3 * ee
	#print out
	#raw_input()

	return out
	
	#e = np.exp(-(x-mean)*(x-mean)/(2*std*std))
	#return (1 / (np.sqrt(2*np.pi) * std*std)) * e


K=5
it=10

centroids = []
centroids_set = set()
threshold = 0.000001
end = 0

X = np.genfromtxt(sys.argv[1], delimiter=",")

number_att = len(X[0])

instances = len(X)

nk = 0
while nk < 5:
	
	l = np.random.randint(0,high=len(X)-1)
	
	if l not in centroids_set:
		centroids_set.add(l)
		centroids.append(l)
		nk = nk + 1



centroids_x = []

for c1 in centroids:
	centroids_x.append(X[c1])


centroids_EM = centroids_x[:]


for iter in range(it):

	#Obtenemos C
	
	
	
	C = {}

	
	for idx, x in enumerate(X):
		cbueno = -1
		min = 1000000000000
		for idy, c in enumerate(centroids_x):
	
		
			dist = np.linalg.norm(x-c)
		

			if min > dist:
				min = dist
				cbueno = idy
		
		
		C[idx] = cbueno
	

	#obtenemos mu

	centroids_x_temp = centroids_x[:]

	centroids_x = []

	for idx, c in enumerate(centroids_x_temp):
		sum_partial = np.zeros(number_att)
		i_c = 0
		for key, value in C.iteritems():
			if value == idx:
				
				sum_partial = sum_partial + X[key]
				i_c = i_c + 1
		sum_partial = sum_partial/i_c
		
		centroids_x.append(sum_partial)
		


	
	kks = 0
	itt = iter + 1
	namefile = "centroids-"+str(itt)+".csv"
	with open(namefile, 'w') as csvfile:
		for aa in centroids_x:
			ee = 0
			for idd in aa:
				if ee != 0:
					csvfile.write(',')
				csvfile.write(str(idd))
				ee = ee + 1
			csvfile.write('\n')

	
	
	kks = 0
	itt1 = iteration + 1
	namefile1 = "pi-"+str(itt1)+".csv"
	with open(namefile1, 'w') as csvfile1:
		for k in range(K):
			csvfile1.write('0.2')
			
			csvfile1.write('\n')
	
	#print centroids_EM
	#raw_input()
	
	itt2 = iteration + 1
	namefile2 = "mu-"+str(itt2)+".csv"
	with open(namefile2, 'w') as csvfile2:
		for c in centroids_EM:
			ee = 0
			for k in range(K):
				if ee != 0:
					csvfile2.write(',')
				csvfile2.write(str(c[k]))
				ee = ee + 1
			csvfile2.write('\n')
	
	itt3 = iteration + 1

	for k in range(K):
		namefile3 = "Sigma-"+str(k+1)+"-"+str(itt3)+".csv"
		with open(namefile3, 'w') as csvfile3:
			
			for n in range(number_att):
				ee = 0
				for nn in range(number_att):
					if ee != 0:
						csvfile3.write(',')
					csvfile3.write(str(sigmas[k][n][nn]))
					ee = ee + 1
				csvfile3.write('\n')


			
			
			
