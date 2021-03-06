#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def fit1(X, y,lmbda):
	dim = X.shape[1]
	y_norm = y - y.mean()
	print X.mean()
	X_Norm = (X - X.mean())/(X.std())
	#print X_Norm
	C = X_Norm.T.dot(X_Norm) + lmbda*np.eye(X_Norm.shape[1])
	w = np.linalg.inv(C).dot(X_Norm.T.dot(y_norm))
	#print w

def ridge_yago(dataX,Y,alpha):
	
	#dataX['ones'] = [1,1,1,1,1,1,1,1,1,1]
	#cols = dataX.columns.tolist()
	#cols = cols[-1:] + cols[:-1]
	#dataX = dataX[cols]
	#print dataX
	#print dataX.mean()
	#print dataX.std()
	dim = dataX.shape[1]
	dataNorm = (dataX - dataX.mean())/(dataX.std())
	
	#print dataNorm
	
	#print dataNorm
	
	
	y_norm = Y - Y.mean()
	
	dataXX = dataNorm.as_matrix()
	
	dataXXT = dataXX.transpose()
	j1 = np.dot(dataXXT,y_norm)
	j2 = np.dot(dataXXT,dataXX)
	gammaMatrix = np.identity(dim,dtype=float)
	
	gammaMatrix = alpha * gammaMatrix
	j4 = gammaMatrix + j2
	j41 = np.add(gammaMatrix,j2)
	j5 = np.linalg.inv(j41)
	wrr = np.dot(j5,j1)
	return wrr



dX_m = np.matrix([[1,2],[3,4],[5,6],[0,0],[-1,-3],[-2,2],[0,-8],[4,3],[3,7],[-1,-3]])
dY_m = np.array([20,22,33,32,25,48,39,21,19,15])
dY_m.shape = (10,1)

matrixA={}
matrixA['x1']=[1,3,5,0,-1,-2,0,4,3,-1]
matrixA['x2']=[2,4,6,0,-3,2,-8,3,7,-3]
matrixA['y']=[20,22,33,32,25,48,39,21,19,15]

dX = pd.DataFrame(matrixA)

dY = pd.Series([20,22,33,32,25,48,39,21,19,15])

a = 1

predictors = ['x1','x2']

print ridge_yago(dX[predictors],dY,a)



ridgereg = Ridge(alpha=a,normalize=True)
ridgereg.fit(dX[predictors],dY)

#print ridgereg.coef_





def fit(X, y, alpha=0):

	
	#print y
	X = np.hstack((np.ones((X.shape[0], 1)), X))

	G = alpha * np.eye(X.shape[1])

	G[0, 0] = 0  # Don't regularize bias
	

	#self.params = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)),np.dot(X.T, y))
	s = np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G))
	j = np.dot(X.T, y)
	#print np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)),np.dot(X.T, y.T))
	tt = np.dot(s,j)
	#print tt


fit(dX_m,dY_m,a)

fit1(dX_m,dY_m,a)






