#https://academo.org/demos/3d-surface-plotter/?expression=x*x*x-50*y*y&xRange=-50%2C+50&yRange=-50%2C+50&resolution=45
#https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import cm


from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data


alpha = 2

X = np.array(range(-50,51,1))
Y = np.array(range(-50,51,1))

np.random.seed(10)  #Setting seed for reproducability



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

X = range(-50, 51, 2)
Y = range(-50, 51, 2)

X, Y = np.meshgrid(X, Y)

Z = (X*X*X - 50*Y*Y )

R = 100000*np.random.random((51, 51))

Z = Z + R



ax.scatter(X,Y,Z,c='r', marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()




from sklearn.linear_model import Ridge
#def ridge_regression(data, predictors, alpha, models_to_plot={}):
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
    
    
    
    
    
    