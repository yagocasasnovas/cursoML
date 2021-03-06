### proyecto 4

import numpy as np
import sys
import csv
import pandas
from scipy.sparse.linalg import svds
from scipy.stats import multivariate_normal
import time
start_time = time.time()


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
	
	#print iteration
	## user update
	
	for i in list_of_users:
	
		##numerador
		
		suma = np.zeros(d)
		s = (d,d)
		suma2 = np.zeros(s)
		
		user_movies = hola[hola['UserID']==i]
		
		for j in user_movies['MovieID']:

			fv = vj[j]
			
			
			#M = hola2.ix[i,j]
			suma = suma + hola2.ix[i,j]*fv
			
			
			suma2 = suma2 + np.outer(fv,fv.T)
			
			
		
		
		#suma2 = suma2 + gamma*cov_sigma*np.identity(d)
		
		
		#denom = np.linalg.inv(suma2 + gamma*cov_sigma*np.identity(d))
		
		
		ui[i] = np.linalg.inv(suma2 + gamma*cov_sigma*np.identity(d)).dot(suma)
		
	###movie update
	

	
	for j in list_of_movies:
		
		
		
		##numerador
		
		suma = np.zeros(d)
		
		s = (d,d)
		suma2 = np.zeros(s)
		
		movie_users = hola[hola['MovieID']==j]
		
		
		for i in movie_users['UserID']:

			
			fu = ui[i]
			#M = hola2.ix[i,j]
			
			suma2 = suma2 + np.outer(fu,fu.T)
			suma = suma +  hola2.ix[i,j]*fu
		
		
		#suma2 = suma2 + gamma*cov_sigma*np.identity(d)
		
		
		#denom = np.linalg.inv(suma2 + gamma*cov_sigma*np.identity(d))
		
		
		vj[j] = np.linalg.inv(suma2 + gamma*cov_sigma*np.identity(d)).dot(suma)
		
		
		
		
	
	####objective function
	
	#primer sumando
	
	sumatorio = 0.0
	
	for i in list_of_users:
		
		for j in list_of_movies:
			
			#dd = np.dot(ui[i],vj[j])
			
			#mm = hola2.ix[i,j]
			
			#tt = (mm - dd)*(mm - dd)
			
			#tt = tt / (2.0*cov_sigma)
			
			tt = (hola2.ix[i,j]-np.dot(ui[i],vj[j]))*(hola2.ix[i,j]-np.dot(ui[i],vj[j])) / (2.0*cov_sigma)
			
			sumatorio = sumatorio + tt
			
	
	sumatorio2 = 0.0
	
	for i in list_of_users:
		
		
		sumatorio2 = sumatorio2 + (gamma/2.0) * np.linalg.norm(ui[i])


	sumatorio3 = 0.0
	
	for j in list_of_movies:
		
		
		sumatorio3 = sumatorio3 + (gamma/2.0) * np.linalg.norm(vj[j])

	lambda1 = sumatorio + sumatorio2 + sumatorio3
	lambda1 = lambda1*(-1.0)
	
	lambda_vector.append(lambda1)
	itt = iteration+1
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

print("--- %s seconds ---" % (time.time() - start_time))


















	
	