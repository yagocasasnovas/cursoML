#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import sys


kkk = ''

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

xtest_file_matrix  = xtest_file.as_matrix()


def ridge_yago(dataX,Y,alpha,dim,l):
	
	
	dataXX = dataX.as_matrix()
	
	dataXXT = dataXX.transpose()
	
	wrr = np.dot(np.linalg.inv(alpha * np.identity(dim,dtype=float) + np.dot(dataXXT,dataXX)),np.dot(dataXXT,Y))
	
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
	
	output = []
	output.append(np.array(ll))
	output.append(np.dot(dataXXT,dataXX))
	return output
	
w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[0]


epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

ok = set()
sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + str(m)

# numero 2

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)


#numero 3

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)


#numero 4

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)


#numero 5

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)



#numero 6

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)


#numero 7

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)



#numero 8

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)



#numero 9

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)





#numero 10

new_vector_frame = pd.DataFrame([x_def], columns=list(xtrain_file))

xtrain_file = xtrain_file.append(new_vector_frame)

new_y = pd.DataFrame([y_def], columns=list(ytrain_file))

ytrain_file = ytrain_file.append(new_y)

#w = ridge_yago(xtrain_file,ytrain_file,lmbda,dim,1)[0]

epsylon = np.linalg.inv(lmbda * np.identity(dim,dtype=float) + (1/(sigma2*sigma2))*ridge_yago(xtrain_file,ytrain_file,lmbda,dim,0)[1])

sig_val = -1
n = 1
m = 0
for x in xtest_file_matrix:
	
	sigma_new = sigma2*sigma2 + np.dot(x,np.dot(epsylon,x))
	
	if sig_val < sigma_new and n not in ok:
		sig_val = sigma_new
		m = n
		x_def = x
		y_def = np.dot(x,w)
	n = n + 1

ok.add(m)

kkk = kkk + ',' + str(m)
















sigma2_f = '{0:.1g}'.format(sigma2)
lmbda_f = '{0:.1g}'.format(lmbda)

filename = 'active_'+str(lmbda_f)+'_'+str(sigma2_f)+'.csv'
file = open(filename,'a')


file.write(kkk)
file.close()
	





