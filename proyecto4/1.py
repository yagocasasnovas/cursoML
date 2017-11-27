### proyecto 4

import numpy as np
import sys
import csv
import pandas
from scipy.sparse.linalg import svds
from scipy.stats import multivariate_normal

input_file = sys.argv[1]


X = np.genfromtxt(input_file, delimiter=",")



with open(input_file, 'rb') as f:
	reader = csv.reader(f)
	your_list = list(reader)
	

new = []

for item in your_list:
	
	dict1 = {}
	
	
	dict1['u'] = int(item[0])
	dict1['v'] = int(item[1])
	dict1['r'] = float(item[2])
	
	new.append(dict1)
	
	
hola = pandas.read_csv(input_file,header=None,names=['UserID', 'MovieID', 'Rating'],usecols=[0,1,2])
	

print hola.head()
print hola['UserID'][0]

print hola['UserID'].min()
print hola['UserID'].max()
minmovie = hola['MovieID'].min()
maxmovie = hola['MovieID'].max()



hola2 = hola.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)





U, sigma, Vt = svds(hola2, k = 5)

sigma = np.diag(sigma)


### iniciar V

vj = []

cov1 = np.identity(5)

cov1 = 0.5*cov1
print cov1

for v in range(minmovie,maxmovie):
	
	vj.append(np.random.multivariate_normal(np.zeros(5),cov1))
	



for iteration in range(50):
	
	## user update
	
	s = 0
	
	for user in hola['UserID']:
		a = 1
		
		




























	
	