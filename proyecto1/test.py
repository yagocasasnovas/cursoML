#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

alpha_number = 10
alpha_nubmer = 1

def ridge_yago(dataX,Y,alpha):
	
	dataNorm = (dataX - dataX.mean())/(dataX.std())
	
	print dataX
	print data.mean()
	print dataX - dataX.mean()
	
	y_norm = Y - Y.mean()
	dataXX = dataNorm.as_matrix()
	dataXXT = dataXX.transpose()
	j1 = np.dot(dataXXT,y_norm)
	j2 = np.dot(dataXXT,dataXX)
	gammaMatrix = np.identity(15,dtype=float)
	gammaMatrix = alpha * gammaMatrix
	j4 = gammaMatrix + j2
	j41 = np.add(gammaMatrix,j2)
	j5 = np.linalg.inv(j41)
	wrr = np.dot(j5,j1)
	return wrr



#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducability
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')

for i in range(2,16):  #power of 1 is already there
	colname = 'x_%d'%i      #new var will be x_power
	data[colname] = data['x']**i
#print data.head()

	
	
#Initialize a dataframe to store the results:
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define the powers for which a plot is required:
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}



from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
	#Fit the model
	ridgereg = Ridge(alpha=alpha,normalize=True)
	ridgereg.fit(data[predictors],data['y'])
	y_pred = ridgereg.predict(data[predictors])
	
	#Check if a plot is to be made for the entered alpha
	if alpha in models_to_plot:
		plt.subplot(models_to_plot[alpha])
		plt.tight_layout()
		plt.plot(data['x'],y_pred)
		plt.plot(data['x'],data['y'],'.')
		plt.title('Plot for alpha: %.3g'%alpha)
	
	#Return the result in pre-defined format
	rss = sum((y_pred-data['y'])**2)
	ret = [rss]
	ret.extend([ridgereg.intercept_])
	ret.extend(ridgereg.coef_)
	return ret
	
	
#Initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
alpha_ridge = [1e-4]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,alpha_nubmer)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)
coef_matrix_ridge1 = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}

X_yago = data[predictors]
yy = data['y']

for i in range(alpha_nubmer):
	print "------------------"
	print alpha_ridge[i]
	#coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
	kk =  ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
	print kk

	#coef_matrix_ridge1.iloc[i,] = ridge_yago(X_yago,yy,alpha_ridge[i])
	print ridge_yago(X_yago,yy,alpha_ridge[i])
	
	
#Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
pd.set_option('display.max_columns', None)
#print(coef_matrix_ridge)
#print(coef_matrix_ridge1)

#coef_matrix_ridge.apply(lambda x: sum(x.values==0),axis=1)
