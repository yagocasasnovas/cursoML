### proyecto 4

import numpy as np
import sys
import csv
import pandas
from scipy.sparse.linalg import svds
from scipy.stats import multivariate_normal
import time



input_file = sys.argv[1]


gamma = 2.0
gammainv = 1/gamma

cov_sigma = 0.1

d = 5 
	
	
hola = pandas.read_csv(input_file,header=None,names=['UserID', 'MovieID', 'Rating'],usecols=[0,1,2])
	


minuser = hola['UserID'].min()
maxuser = hola['UserID'].max()
minmovie = hola['MovieID'].min()
maxmovie = hola['MovieID'].max()



hola2 = hola.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
hola3 = hola.pivot(index = 'MovieID', columns ='UserID', values = 'Rating').fillna(0)






### iniciar V



cov1 = np.identity(d)


cov1 = gammainv*cov1

vj = {}


v_length = len(hola['MovieID'].unique())
u_length = len(hola['UserID'].unique())

#print v_length

list_of_users = hola['UserID'].unique()
list_of_movies = hola['MovieID'].unique()

for v1 in list_of_movies:
	
	
	#vj.append(np.random.multivariate_normal(np.zeros(5),cov1))
	vj[v1] = np.random.multivariate_normal(np.zeros(5),cov1)


vj = pandas.DataFrame.from_dict(vj)

#print vj.head()
#raw_input()

## iniciar U 

ui = {}

for u1 in list_of_users:
	
	#ui.append(np.zeros(d))
	ui[u1]=np.zeros(d)

ui = pandas.DataFrame.from_dict(ui)

lambda_vector = []


for iteration in range(50):
	
	#print "iteration"
	print iteration
	start_time = time.time()
	#print iteration
	## user update
	
	for i in list_of_users:
	
		
		user_movies = hola[hola['UserID']==i]
		
		vjs = vj[user_movies['MovieID']]
		
		
		suma2 = np.dot(vjs,vjs.T) + gamma*cov_sigma*np.identity(d)
		
		
		denom = np.linalg.inv(suma2)
		

		ms = hola3[i][user_movies['MovieID']]

		
		ty = np.dot(vjs,ms)
		
		ui[i] = denom.dot(ty)
		
		
		
	print("--- %s seconds user update ---" % (time.time() - start_time))
	
	
	
	###movie update
	start_time = time.time()

	
	for j in list_of_movies:
		
		
		movie_users = hola[hola['MovieID']==j]
		
		uis = ui[movie_users['UserID']]
		#print movie_users['Rating']

		suma2 = np.dot(uis,uis.T) + gamma*cov_sigma*np.identity(d)
		
		
		
		denom = np.linalg.inv(suma2)

		
		

		ms1 = hola2[j][movie_users['UserID']]

		
		
		
		
		ty1 = np.dot(uis,ms1)
		vj[j] = denom.dot(ty1)

		
	print("--- %s seconds movie update ---" % (time.time() - start_time))
	####objective function
	start_time = time.time()
	#primer sumando
	
	
	
	sf = (hola2 - np.dot(ui.T,vj))
	
	
	power = lambda x: x*x
	
	sf = sf.apply(power)
	
	sf = sf.sum().sum()
	
	sumatorio = sf /  (2.0*cov_sigma)
	
	
	norma = lambda x: (gamma/2.0)*np.linalg.norm(x)*np.linalg.norm(x)
	
	ui_1 = ui.apply(norma)
	
	sumatorio2 = ui_1.sum()
	
	vj_1 = vj.apply(norma)
	
	sumatorio3 = vj_1.sum()
	

	lambda1 = sumatorio + sumatorio2 + sumatorio3
	lambda1 = lambda1*(-1.0)
	
	lambda_vector.append(lambda1)
	itt = iteration+1
	
	print("--- %s seconds lambda update ---" % (time.time() - start_time))
	
	if itt in [10,25,50]:
		namefile2 = "U-"+str(itt)+".csv"
		with open(namefile2, 'w') as csvfile2:
			for u in ui:
				
				ee = 0
				for l in ui[u]:
					if ee != 0:
						csvfile2.write(',')
					csvfile2.write(str(l))
					ee = ee + 1
				csvfile2.write('\n')
		csvfile2.close()
		namefile3 = "V-"+str(itt)+".csv"
		with open(namefile3, 'w') as csvfile3:
			for v in vj:
				ee = 0
				for l in vj[v]:
					if ee != 0:
						csvfile3.write(',')
					csvfile3.write(str(l))
					ee = ee + 1
				csvfile3.write('\n')
		csvfile3.close()
namefile4 = "objective.csv"
with open(namefile4, 'w') as csvfile4:
	for la in lambda_vector:
		csvfile4.write(str(la))
		csvfile4.write('\n')

csvfile4.close()

#print("--- %s seconds ---" % (time.time() - start_time))


















	
	