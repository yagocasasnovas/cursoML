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

xtrain_file = xtrain_file * 1.0

ytrain_file = pd.read_csv(y_train_file, sep = ',', header=None)

ytrain_file.columns = ['y']

ytrain_file = ytrain_file * 1.0

xtest_file = pd.read_csv(x_test_file, sep = ',', header=None)

xtest_file.columns = cols

#xtest_file['intercept'] = pd.Series(np.ones(xtrain_file.shape[0]),dtype=float)

xtest_file = xtest_file * 1.0

xtrain_file_matrix = xtrain_file.as_matrix()

ytrain_file_matrix = ytrain_file.as_matrix()


clf = Ridge(alpha=lmbda,fit_intercept=False)

clf.fit(xtrain_file_matrix, ytrain_file_matrix)




def ridge_yago(dataX,Y,alpha,dim,l):
	
	
	dataXX = dataX.as_matrix()
	
	dataXXT = dataXX.transpose()
	
	j1 = np.dot(dataXXT,Y)
	
	j2 = np.dot(dataXXT,dataXX)
	
	gammaMatrix = alpha * np.identity(dim,dtype=float)

	j4 = gammaMatrix + j2
	
	j5 = np.linalg.inv(j4)
	
	wrr = np.dot(j5,j1)
	
	wrr_a = wrr.tolist()
	
	if l == 0:
		alpha1 = '{0:.1g}'.format(alpha)
		filename = 'wRR_'+str(alpha1)+'.csv'
	
		file = open(filename,'w')
	
		for w in wrr_a:
		
			file.write(str(w[0])+'\n')
		file.close()
	
	ll = []
	for s in wrr_a:
		ll.append(s[0])
	
	return np.array(ll)
	
w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)


def epsilon(dataX,Y,w,alpha,sigma_2,testX_vector,dim):
	
	dataXX = dataX.as_matrix()
	
	tt = np.zeros(dim)
	
	for j in dataXX:
		
		dd = np.array(crossProd(j,j),dtype=float)
		tt = tt + crossProd(j,j)
		
	t1 = (1/(sigma_2*sigma_2))*(testX_vector + tt)

	gammaMatrix = alpha * np.identity(dim,dtype=float)

	t2 = gammaMatrix + t1
	
	t3 = np.linalg.inv(t2)
	
	return t3
	
def epsilon1(dataX,Y,w,alpha,sigma_2,dim):
	
	dataXX = dataX.as_matrix()
	
	tt = np.zeros(dim)
	
	for j in dataXX:
		
		dd = np.array(crossProd(j,j),dtype=float)
		tt = tt + dd
		
	
	j1 = np.dot(dataXX.transpose(),dataXX)

	#t1 = (1/(sigma_2*sigma_2))*(testX_vector + tt)
	t1 = (1/(sigma_2*sigma_2))*j1

	gammaMatrix = alpha * np.identity(dim,dtype=float)

	t2 = gammaMatrix + t1
	
	t3 = np.linalg.inv(t2)
	
	return t3


def get_max(xtest_file_local,xtrain_file_local,ytrain_file_local,w_local,lmbda_local,sigma2_local,dim,l,hh):
	
	xtest_file_matrix = xtest_file_local.as_matrix()
	
	max_sigma = 0
	
	n = 1
	
	for xvector_test in xtest_file_matrix:

	
		epsylon1 = epsilon1(xtrain_file_local,ytrain_file_local,w_local,lmbda_local,sigma2_local,dim)
	
		r1 = np.dot(epsylon1,xvector_test)
	
		ss0 = np.dot(xvector_test,r1)
	
		if max_sigma < ss0 and n != hh:
			max_sigma = ss0
			max_vector = xvector_test
			
			m = n
		n = n + 1
	
	
	
	#xvector_test = np.append(xvector_test,1.0)
	y_new = np.dot(xvector_test,w_local)
	
	sigma2_local1 = '{0:.1g}'.format(sigma2_local)
	lmbda_local1 = '{0:.1g}'.format(lmbda_local)
	filename = 'active_'+str(lmbda_local1)+'_'+str(sigma2_local1)+'.csv'
	file = open(filename,'a')
	if l == 0:
		file.write(str(m)+',')
	else:
		file.write(str(m))
	file.close()
	
	#print 'iteration'
	#print w_local
	#print y_new
	#print max_sigma
	#print max_vector
	#print m
	
	output = []
	
	output.append(y_new)
	output.append(max_vector)
	output.append(m)
	
	return output

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,-1)

new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,new_vector[2])

new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,new_vector[2])



new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,new_vector[2])



new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,new_vector[2])



new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,new_vector[2])





new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,new_vector[2])




new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,new_vector[2])




new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0,new_vector[2])




new_vector_frame = pd.DataFrame([new_vector[1]], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([new_vector[0]], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#xtest_file = xtest_file.drop(xtest_file.index[new_vector[2]])

#-------

w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)

new_vector = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,1,new_vector[2])






