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


def get_max(xtest_file_local,xtrain_file_local,ytrain_file_local,w_local,lmbda_local,sigma2_local,dim,l):
	
	xtest_file_matrix = xtest_file_local.as_matrix()
	
	max_sigma = 0
	
	n = 0
	
	for xvector_test in xtest_file_matrix:

	
		epsylon1 = epsilon1(xtrain_file_local,ytrain_file_local,w_local,lmbda_local,sigma2_local,dim)
	
		r1 = np.dot(epsylon1,xvector_test)
	
		ss0 = np.dot(xvector_test,r1)
	
		if max_sigma < ss0:
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

new_vector0 = get_max(xtest_file, xtrain_file, ytrain_file, w, lmbda, sigma2,dim,0)

new_vector_frame0 = pd.DataFrame([new_vector0[1]], columns=list(xtrain_file))

xtrain_file1 = xtrain_file.append(new_vector_frame0)

new_y0 = pd.DataFrame([new_vector0[0]], columns=list(ytrain_file))

ytrain_file1 = ytrain_file.append(new_y0)

xtest_file1 = xtest_file.drop(xtest_file.index[new_vector0[2]])

w1 = ridge_yago(xtrain_file1,ytrain_file1,lmbda,dim,1)

#--------------------

new_vector1 = get_max(xtest_file1, xtrain_file1, ytrain_file1, w1, lmbda, sigma2,dim,0)

new_vector_frame1 = pd.DataFrame([new_vector1[1]], columns=list(xtrain_file))

xtrain_file2 = xtrain_file1.append(new_vector_frame1)

new_y1 = pd.DataFrame([new_vector1[0]], columns=list(ytrain_file))

ytrain_file2 = ytrain_file1.append(new_y1)

xtest_file2 = xtest_file1.drop(xtest_file1.index[new_vector1[2]])

w2 = ridge_yago(xtrain_file2,ytrain_file2,lmbda,dim,1)

#--------------------

new_vector2 = get_max(xtest_file2, xtrain_file2, ytrain_file2, w2, lmbda, sigma2,dim,0)

new_vector_frame2 = pd.DataFrame([new_vector2[1]], columns=list(xtrain_file))

xtrain_file3 = xtrain_file2.append(new_vector_frame2)

new_y2 = pd.DataFrame([new_vector2[0]], columns=list(ytrain_file))

ytrain_file3 = ytrain_file2.append(new_y2)

xtest_file3 = xtest_file2.drop(xtest_file2.index[new_vector2[2]])

w3 = ridge_yago(xtrain_file3,ytrain_file3,lmbda,dim,1)

#--------------------

new_vector3 = get_max(xtest_file3, xtrain_file3, ytrain_file3, w3, lmbda, sigma2,dim,0)

new_vector_frame3 = pd.DataFrame([new_vector3[1]], columns=list(xtrain_file))

xtrain_file4 = xtrain_file3.append(new_vector_frame3)

new_y3 = pd.DataFrame([new_vector3[0]], columns=list(ytrain_file))

ytrain_file4 = ytrain_file3.append(new_y3)

xtest_file4 = xtest_file3.drop(xtest_file3.index[new_vector3[2]])

w4 = ridge_yago(xtrain_file4,ytrain_file4,lmbda,dim,1)

#--------------------

new_vector4 = get_max(xtest_file4, xtrain_file4, ytrain_file4, w4, lmbda, sigma2,dim,0)

new_vector_frame4 = pd.DataFrame([new_vector4[1]], columns=list(xtrain_file))

xtrain_file5 = xtrain_file4.append(new_vector_frame4)

new_y4 = pd.DataFrame([new_vector4[0]], columns=list(ytrain_file))

ytrain_file5 = ytrain_file4.append(new_y4)

xtest_file5 = xtest_file4.drop(xtest_file4.index[new_vector4[2]])

w5 = ridge_yago(xtrain_file5,ytrain_file5,lmbda,dim,1)

#--------------------

new_vector5 = get_max(xtest_file5, xtrain_file5, ytrain_file5, w5, lmbda, sigma2,dim,0)

new_vector_frame5 = pd.DataFrame([new_vector5[1]], columns=list(xtrain_file))

xtrain_file6 = xtrain_file5.append(new_vector_frame5)

new_y5 = pd.DataFrame([new_vector5[0]], columns=list(ytrain_file))

ytrain_file6 = ytrain_file5.append(new_y5)

xtest_file6 = xtest_file5.drop(xtest_file5.index[new_vector5[2]])

w6 = ridge_yago(xtrain_file6,ytrain_file6,lmbda,dim,1)

#--------------------

new_vector6 = get_max(xtest_file6, xtrain_file6, ytrain_file6, w6, lmbda, sigma2,dim,0)

new_vector_frame6 = pd.DataFrame([new_vector6[1]], columns=list(xtrain_file))

xtrain_file7 = xtrain_file6.append(new_vector_frame6)

new_y6 = pd.DataFrame([new_vector6[0]], columns=list(ytrain_file))

ytrain_file7 = ytrain_file6.append(new_y6)

xtest_file7 = xtest_file6.drop(xtest_file6.index[new_vector6[2]])

w7 = ridge_yago(xtrain_file7,ytrain_file7,lmbda,dim,1)

#--------------------

new_vector7 = get_max(xtest_file7, xtrain_file7, ytrain_file7, w7, lmbda, sigma2,dim,0)

new_vector_frame7 = pd.DataFrame([new_vector7[1]], columns=list(xtrain_file))

xtrain_file8 = xtrain_file7.append(new_vector_frame7)

new_y7 = pd.DataFrame([new_vector7[0]], columns=list(ytrain_file))

ytrain_file8 = ytrain_file7.append(new_y7)

xtest_file8 = xtest_file7.drop(xtest_file7.index[new_vector7[2]])

w8 = ridge_yago(xtrain_file8,ytrain_file8,lmbda,dim,1)

#--------------------

new_vector8 = get_max(xtest_file8, xtrain_file8, ytrain_file8, w8, lmbda, sigma2,dim,0)

new_vector_frame8 = pd.DataFrame([new_vector8[1]], columns=list(xtrain_file))

xtrain_file9 = xtrain_file8.append(new_vector_frame8)

new_y8 = pd.DataFrame([new_vector8[0]], columns=list(ytrain_file))

ytrain_file9 = ytrain_file8.append(new_y8)

xtest_file9 = xtest_file8.drop(xtest_file8.index[new_vector8[2]])

w9 = ridge_yago(xtrain_file9,ytrain_file9,lmbda,dim,1)

#--------------------

new_vector9 = get_max(xtest_file9, xtrain_file9, ytrain_file9, w9, lmbda, sigma2,dim,1)






