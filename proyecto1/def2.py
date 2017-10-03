#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import sys
from sklearn.linear_model import Ridge

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

lmbda = float(sys.argv[1])
sigma2 = float(sys.argv[2])
x_train_file = sys.argv[3]
y_train_file = sys.argv[4]
x_test_file = sys.argv[5]


xtrain_file = pd.read_csv(x_train_file, sep = ',', header=None)

dim = xtrain_file.shape[1]

cols = ['a{}'.format(i) for i in range(0, dim)]

xtrain_file.columns = cols

xtrain_file_original = xtrain_file.copy()

xtrain_file_intercept = xtrain_file_ * 1.0

ytrain_file = pd.read_csv(y_train_file)

ytrain_file = ytrain_file * 1.0

xtest_file = pd.read_csv(x_test_file)

#xtest_file['intercept'] = pd.Series(np.ones(xtrain_file.shape[0]),dtype=float)

xtest_file = xtest_file * 1.0


xtrain_file_matrix = xtrain_file_intercept.as_matrix()

ytrain_file_matrix = ytrain_file.as_matrix()


clf = Ridge(alpha=lmbda,fit_intercept=False)

clf.fit(xtrain_file_matrix, ytrain_file_matrix)




def ridge_yago(dataX,Y,alpha):
	
	dimension = dataX.shape
	dimension = dimension[1]
	
	#estandarizar dataX
	
	Xmean = dataX.mean()
	
	
	Xstd = dataX.std()
	#dataX = (dataX-Xmean)/Xstd
	

	#normalizar Y
	
	#Y = Y - Y.mean()
	
	
	dataXX = dataX.as_matrix()
	
	
	
	dataXXT = dataXX.transpose()
	
	
	j1 = np.dot(dataXXT,Y)
	
	
	
	
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
	
w = ridge_yago(xtrain_file_intercept,ytrain_file,lmbda)




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
	n = 0
	for xvector_test in xtest_file_matrix:
	
		epsylon1 = epsilon(xtrain_file_local,ytrain_file_local,w_local,lmbda_local,sigma2_local,xvector_test)
	
		r1 = np.dot(epsylon1,xvector_test)
	
		ss0 = np.dot(xvector_test,r1)
	
		if max_sigma < ss0:
			max_sigma = ss0
			max_vector = xvector_test
		n = n + 1
	#calculo y0
	
	
	xvector_test = np.append(xvector_test,1.0)
	y_new = np.dot(xvector_test,w_local)
	
	
	filename = 'active_'+str(lmbda_local)+'_'+str(sigma2_local)+'.csv'
	file = open(filename,'a')
	
	file.write(str(n)+',')
	file.close()
	
	print 'iteration'
	print w_local
	print y_new
	print max_sigma
	print max_vector
	print n
	
	output = []
	
	output.append(y_new)
	output.append(max_vector)
	output.append(n)
	
	return output

output = get_max(xtest_file, xtrain_file_original, ytrain_file, w, lmbda, sigma2)

#anyado un 1 final al vector maximo X

oo = np.append(output[1],1.0)

#creo un dataframe a partir del array oo

outt = pd.DataFrame([oo], columns=list(xtrain_file_intercept))

#anyado el vector maximo con el 1 final a X dataframe

xtrain_file_intercept1 = xtrain_file_intercept.append(outt)

#anyado el valor y0 al nuevo Y

outt2 = pd.DataFrame([output[0]], columns=list(ytrain_file))

ytrain_file1 = ytrain_file.append(outt2)

#creo un dataframe a partir del array output (sin unos)

outt2 = pd.DataFrame([output[1]], columns=list(xtrain_file_original))

#anyado el vector max al dataframe X sin 1s

xtrain_file_original1 = xtrain_file_original.append(outt2)

xtrain_file_matrix1 = xtrain_file_intercept1.as_matrix()

ytrain_file_matrix1 = ytrain_file1.as_matrix()

xtest_file1 = xtest_file.drop(xtest_file.index[output[2]-1])

w1 = ridge_yago(xtrain_file_intercept1,ytrain_file1,lmbda)

output = get_max(xtest_file1, xtrain_file_original1, ytrain_file1, w1, lmbda, sigma2)
