
import numpy as np
import sys
import csv
import pandas
from scipy.sparse.linalg import svds
from scipy.stats import multivariate_normal
import time



a = [1.0,2.0,3.0]
b = [4.0,5.0,6.0]

a = np.array(a)
b = np.array(b)


j = np.outer(a,a.T)

k = np.outer(b,b.T)


print j+k


c = np.column_stack((a,b))



print np.dot(c,c.T)

