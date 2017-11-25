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
import time
start_time = time.time()

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
		min = 10000000
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




##########EM GMM



###inicializar means: centroids_EM

### inicializar sigmas dxd donde d = numero de atributos

sigmas = []

factor = 1000

for k in range(K):

	sigmas.append(factor*np.identity(number_att))

##distribution

pi = np.zeros(K)

for k in range(K):
	
	pi[k] = (1/K)



for iteration in range(it):
	
	
	#######################3###E Step

	fi_x_vector = []



	determ = np.linalg.det(sigmas[k])
	

	for i in range(instances):
	
		fi_k_vector = np.zeros(K)
	
		for k in range(K):
			
			
			
			pk = multivariate_normal.pdf(X[i],centroids_EM[k],sigmas[k],allow_singular=True)

		
			pkk = pk * pi[k]
		
			ss = 0
			for k1 in range(K):
				#print str(i) + ' ' + str(k) + ' ' + str(k1) + ' ' + str(len(centroids_EM)) + ' ' + str(len(sigmas))+' ' + str(len(pi))
				#print str(i) + ' ' + str(k) + ' ' + str(k1)
				p = multivariate_normal.pdf(X[i],centroids_EM[k1],sigmas[k1],allow_singular=True)
				pp = p * pi[k1]
			
				ss = ss + pp

			fi_k_vector[k] = pkk/ss
	
		fi_x_vector.append(fi_k_vector)
	



	################################	#M Step

	n_vector = np.zeros(K)
	
	
	for k in range(K):
		
		n_temp_k = 0
		
		for i in range(instances):
		
			n_temp_k = n_temp_k + fi_x_vector[i][k]
		
		n_vector[k] = n_temp_k
	

	#print np.linalg.norm(n_vector)
	#print pi
	

	
	#update pi
		
		ty = n_vector[k]
		ty = ty / instances
		pi[k]=ty
		
		
		
		#update mean
	
		cuscus = 0
		for i in range(instances):
		
			as1 = n_vector[k]*X[i]
		
			cuscus = cuscus + as1
		
		centroids_EM[k] = cuscus/n_vector[k]
		
		
		#update sigma
		cuscus2 = 0
		for i in range(instances):
		
			ps = X[i] - centroids_EM[k]
		
			sw = np.outer(ps,ps.T)
			sw1 = fi_x_vector[i][k]*sw
		
			cuscus2 = cuscus2 + sw1
	
		sigmas[k] = cuscus2/n_vector[k]
		

	#print pi
	#print centroids_EM
	#print sigmas
	#print fi_x_vector
	#raw_input()
	
	
print("--- %s seconds ---" % (time.time() - start_time))

			
			
			
