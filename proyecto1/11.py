#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random

from sklearn.linear_model import Ridge



def ridge_yago(dataX,Y,alpha):
	
	dimension = dataX.shape
	dimension = dimension[1]
	
	print type(Y)
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

	print wrr
	
	wrr_a = wrr.tolist()
	print wrr_a
	
	ll = []
	for s in wrr_a:
		ll.append(s[0])
		
	
	return np.array(ll)


#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducability
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])


for i in range(2,16):  #power of 1 is already there
	colname = 'x_%d'%i      #new var will be x_power
	data[colname] = data['x']**i
#print data.head()

predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

ridgereg = Ridge(alpha=1.0,normalize=True)

ridgereg.fit(data[predictors],data['y'])

y_pred = ridgereg.predict(data[predictors])

print ridgereg.coef_


ridge_yago(data[predictors],data['y'],1.0)