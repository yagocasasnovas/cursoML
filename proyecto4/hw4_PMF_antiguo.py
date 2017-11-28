### proyecto 4

import numpy as np
import sys
import csv
import pandas
from scipy.sparse.linalg import svds
from scipy.stats import multivariate_normal



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







### iniciar V



cov1 = np.identity(d)


cov1 = gammainv*cov1

vj = {}


v_length = len(hola['MovieID'].unique())
u_length = len(hola['UserID'].unique())



for v1 in hola['MovieID']:
	
	
	#vj.append(np.random.multivariate_normal(np.zeros(5),cov1))
	vj[v1] = np.random.multivariate_normal(np.zeros(5),cov1)

## iniciar U 

ui = {}

for u1 in hola['UserID']:
	
	#ui.append(np.zeros(d))
	ui[u1]=np.zeros(d)


lambda_vector = []


for iteration in range(50):
	
	## user update
	print iteration
	for i in hola['UserID']:
	
		
		##numerador
		
		suma = np.zeros(d)
		s = (d,d)
		suma2 = np.zeros(s)
		
		user_movies = hola[hola['UserID']==i]
		
		for j in user_movies['MovieID']:

			fv = vj[j]
			M = hola2.ix[i,j]
			suma = suma + M*fv
			suma2 = suma2 + np.outer(fv,fv.T)
		
		
		suma2 = suma2 + gamma*cov_sigma*np.identity(d)
		
		
		denom = np.linalg.inv(suma2)
		
		
		ui[i] = denom.dot(suma)
		
	###movie update
	

	
	for j in hola['MovieID']:
		
		
		
		##numerador
		
		suma = np.zeros(d)
		
		s = (d,d)
		suma2 = np.zeros(s)
		
		movie_users = hola[hola['MovieID']==j]
		
		
		for i in movie_users['UserID']:

			
			fu = ui[i]
			M = hola2.ix[i,j]
			
			suma2 = suma2 + np.outer(fu,fu.T)
			suma = suma + M*fu
		
		
		suma2 = suma2 + gamma*cov_sigma*np.identity(d)
		
		
		denom = np.linalg.inv(suma2)
		
		
		vj[j] = denom.dot(suma)
		
		
		
		
	
	####objective function
	
	#primer sumando
	
	sumatorio = 0.0
	
	for i in hola['UserID']:
		
		for j in hola['MovieID']:
			
			dd = np.dot(ui[i],vj[j])
			
			mm = hola2.ix[i,j]
			
			tt = (mm - dd)*(mm - dd)
			
			tt = tt / (2.0*cov_sigma)
			
			sumatorio = sumatorio + tt
			
	
	sumatorio2 = 0.0
	
	for i in hola['UserID']:
		
		
		sumatorio2 = sumatorio2 + (gamma/2.0) * np.linalg.norm(ui[i])


	sumatorio3 = 0.0
	
	for j in hola['MovieID']:
		
		
		sumatorio3 = sumatorio3 + (gamma/2.0) * np.linalg.norm(vj[j])

	lambda1 = sumatorio + sumatorio2 + sumatorio3
	lambda1 = lambda1*(-1.0)
	print lambda1
	lambda_vector.append(lambda1)
	itt = iteration+1
	if itt in [10,25,50]:
		namefile2 = "U-"+itt+".csv"
		with open(namefile2, 'w') as csvfile2:
			for u in ui:
				ee = 0
				for l in range(len(u)):
					if ee != 0:
						csvfile2.write(',')
					csvfile2.write(str(u[l]))
					ee = ee + 1
				csvfile2.write('\n')
		namefile3 = "V-"+itt+".csv"
		with open(namefile3, 'w') as csvfile3:
			for v in vj:
				ee = 0
				for l in range(len(v)):
					if ee != 0:
						csvfile3.write(',')
					csvfile3.write(str(v[l]))
					ee = ee + 1
				csvfile3.write('\n')
	
namefile4 = "objective.csv"
with open(namefile4, 'w') as csvfile4:
	for la in lambda_vector:
		csvfile4.write(str(la))
		csvfile4.write('\n')






















	
	