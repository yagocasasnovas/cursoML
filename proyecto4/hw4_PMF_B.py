### proyecto 4
##version 10


import numpy as np
import sys
import csv
import pandas
from scipy.sparse.linalg import svds
from scipy.stats import multivariate_normal
import time

#um = vj[hola[hola['UserID']==i]['MovieID']]
#ur = vj[hola[hola['UserID']==i]['Rating']]


def update_u(um,ur,gamma,cov_sigma,d,x):
	
	suma2 = np.dot(um,um.T) + gamma*cov_sigma*np.identity(d)
	denom = np.linalg.inv(suma2)
	ty = np.dot(vjs,ur)
	return denom.dot(ty1)

input_file = sys.argv[1]

gamma = 2.0
gammainv = 1/gamma

cov_sigma = 0.1

d = 5 
	
	
hola = pandas.read_csv(input_file,header=None,names=['UserID', 'MovieID', 'Rating'],usecols=[0,1,2])
#hola1 = pandas.read_csv(input_file,header=None,names=['MovieID','UserID', 'Rating'],usecols=[0,1,2])




hola2 = hola.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)


power = lambda x: x*x

norma = lambda x: (gamma/2.0)*np.linalg.norm(x)*np.linalg.norm(x)

tr = lambda x: update_u(vj[hola[hola['UserID']==x]['MovieID']],vj[hola[hola['UserID']==x]['Rating']],gamma,cov_sigma,d)
	
	

### iniciar V



cov1 = np.identity(d)


cov1 = gammainv*cov1

vj = {}


#v_length = len(hola['MovieID'].unique())
#u_length = len(hola['UserID'].unique())

#print v_length

list_of_users = hola['UserID'].unique()
list_of_movies = hola['MovieID'].unique()


for v1 in list_of_movies:
	
	
	#vj.append(np.random.multivariate_normal(np.zeros(5),cov1))
	vj[v1] = np.random.multivariate_normal(np.zeros(5),cov1)


vj = pandas.DataFrame.from_dict(vj)


vhola = hola.copy()


vhola = vhola.drop('Rating',1)

vhola['0'] = 0.0
vhola['1'] = 0.0
vhola['2'] = 0.0
vhola['3'] = 0.0
vhola['4'] = 0.0

vectorss = ['0','1','2','3','4']
vectorss1 = ['MovieID','0','1','2','3','4']

for index, row in vhola.iterrows():

	
	vhola.loc[index,vectorss] = np.random.multivariate_normal(np.zeros(5),cov1)
	


#print vhola.head()

#vhola2 = vhola.pivot(index = 'UserID', columns ='MovieID').fillna(0)



## iniciar U 

ui = {}

for u1 in list_of_users:
	
	#ui.append(np.zeros(d))
	ui[u1]=np.zeros(d)



ui = pandas.DataFrame.from_dict(ui)

uhola = hola.copy()

uhola['0'] = 0.0
uhola['1'] = 0.0
uhola['2'] = 0.0
uhola['3'] = 0.0
uhola['4'] = 0.0

uhola = uhola.drop('Rating',1)





lambda_vector = []


user_movies = []





for iteration in range(50):
	
	#print "iteration"
	print iteration
	start_time = time.time()
	start_time1 = time.time()
	#print iteration
	## user update
	
	
	
	for i in list_of_users:
	
		
		#user_movies = hola[hola['UserID']==i]
		
		#vjs = vj[user_movies['MovieID']]
		
		#print vj[hola[hola['UserID']==i]['MovieID']]
		
		#print hola[hola['UserID']==i]['MovieID']
		#raw_input()
		
		#print vjs
		
		vjs = vhola[vhola['UserID']==i][vectorss].T
		
		#raw_input()
		
		suma2 = np.dot(vjs,vjs.T) + gamma*cov_sigma*np.identity(d)
		
		
		denom = np.linalg.inv(suma2)
		

		#ms = hola[hola['UserID']==i]['Rating']
		#print ms
		
		ms = hola2.ix[i][hola[hola['UserID']==i]['MovieID']]
		#print ms
		#raw_input()
		
		
		
		
		ty = np.dot(vjs,ms)
		
		#ui[i] = denom.dot(ty)
		
		
		uhola.loc[uhola.UserID == i,vectorss] = denom.dot(ty)
		
		
		
		
		
		
		
		
	print("--- %s seconds user update ---" % (time.time() - start_time))
	
	
	
	###movie update
	start_time = time.time()

	
	for j in list_of_movies:
		
		
		
		movie_users = hola[hola['MovieID']==j]
		
		uis = ui[movie_users['UserID']]

		suma2 = np.dot(uis,uis.T) + gamma*cov_sigma*np.identity(d)
		
		
		
		denom = np.linalg.inv(suma2)

		
		

		ms1 = movie_users['Rating']

		
		
		
		
		ty1 = np.dot(uis,ms1)
		vj[j] = denom.dot(ty1)

		
	print("--- %s seconds movie update ---" % (time.time() - start_time))
	####objective function
	start_time = time.time()
	#primer sumando
	
	
	
	sf = (hola2 - np.dot(ui.T,vj))
	
	
	
	
	sf = sf.apply(power)
	
	sf = sf.sum().sum()
	
	sumatorio = sf /  (2.0*cov_sigma)
	
	
	
	
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
		
		
	print("--- %s seconds iteration update ---" % (time.time() - start_time1))
		
		
		
namefile4 = "objective.csv"
with open(namefile4, 'w') as csvfile4:
	for la in lambda_vector:
		csvfile4.write(str(la))
		csvfile4.write('\n')

csvfile4.close()

#print("--- %s seconds ---" % (time.time() - start_time))


















	
	