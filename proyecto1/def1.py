#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import sys
from sklearn.linear_model import Ridge

lmbda = float(sys.argv[1])
sigma2 = float(sys.argv[2])
x_train_file = sys.argv[3]
y_train_file = sys.argv[4]
x_test_file = sys.argv[5]



def crossProd(a,b):
	dimension = len(a)
	c = []
	for i in range(dimension):
		c.append(0)
		for j in range(dimension):
			if j <> i:
				for k in range(dimension):
					if k <> i:
						if k > j:
							c[i] += a[j]*b[k]
						elif k < j:
							c[i] -= a[j]*b[k]
	return c

xtrain_file = pd.read_csv(x_train_file)

xtrain_file = xtrain_file * 1.0

xtrain_file_m = xtrain_file.as_matrix()
print xtrain_file_m[:,0:xtrain_file_m.shape[1]-1]

ytrain_file = pd.read_csv(y_train_file)

ytrain_file = ytrain_file * 1.0

xtest_file = pd.read_csv(x_test_file)

xtest_file = xtest_file * 1.0



def ridge_yago(dataX,Y,alpha):
	
	dimension = dataX.shape
	dimension = dimension[1]
	
	#estandarizar dataX
	
	Xmean = dataX.mean()
	
	
	Xstd = dataX.std()
	dataX = (dataX-Xmean)/Xstd
	

	#normalizar Y
	
	Y1 = Y - Y.mean()
	
	
	dataXX = dataX.as_matrix()
	
	
	
	dataXXT = dataXX.transpose()
	
	
	j1 = np.dot(dataXXT,Y1)
	
	
	j2 = np.dot(dataXXT,dataXX)
	
	gammaMatrix = np.identity(dimension,dtype=float)

	gammaMatrix = alpha * gammaMatrix
	
	j4 = gammaMatrix + j2
	
	j5 = np.linalg.inv(j4)
	
	wrr = np.dot(j5,j1)

	
	
	wrr_a = wrr.tolist()
	
	filename = 'wRR_'+str(alpha)+'.csv'
	file = open(filename,'w')
	
	for w in wrr_a:
		
		file.write(str(w[0])+'\n')
	file.close()
	
	ll = []
	for s in wrr_a:
		ll.append(s[0])
		
	
	return np.array(ll)
	
w = ridge_yago(xtrain_file,ytrain_file,lmbda)

xtest_file_matrix = xtest_file.as_matrix()

for x in xtest_file_matrix:
	
	#print np.dot(x,w)
	a = 1


def epsilon(dataX,Y,w,alpha,sigma_2,testX_vector):
	
	
	dimension = dataX.shape
	
	dimension = dimension[1]
	
	#calcular epsilon(x0)
	
	dataXX = dataX.as_matrix()
	
	tt = np.zeros(dimension)
	for j in dataXX:

		
		dd = np.array(crossProd(j,j),dtype=float)
		tt = tt + crossProd(j,j)
		
	
	t1 = (1/(sigma_2*sigma_2))*(testX_vector + tt)

	gammaMatrix = np.identity(dimension,dtype=float)

	gammaMatrix = alpha * gammaMatrix
	
	t2 = gammaMatrix + t1
	
	
	
	t3 = np.linalg.inv(t2)
	
	
	return t3
	


def get_max(xtest_file_local,xtrain_file_local,ytrain_file_local,w_local,lmbda_local,sigma2_local):
	xtest_file_matrix = xtest_file_local.as_matrix()
	max_sigma = 0
	for xvector_test in xtest_file_matrix:
	
		epsylon1 = epsilon(xtrain_file_local,ytrain_file_local,w_local,lmbda_local,sigma2_local,xvector_test)
	
		r1 = np.dot(epsylon1,xvector_test)
	
		ss0 = np.dot(xvector_test,r1)
	
		if max_sigma < ss0:
			max_sigma = ss0
			max_vector = xvector_test
	
	#calculo y0
	
	
	y_new = np.dot(xvector_test,w_local)
	
	#print xvector_test
	#print w_local
	#print y_new
	#print max_sigma
	#print max_vector


get_max(xtest_file, xtrain_file, ytrain_file, w, 1.0, sigma2)








