### proyecto 4
##version 10


import numpy as np
import sys
import csv
import pandas
from scipy.sparse.linalg import svds
from scipy.stats import multivariate_normal
import time

start_time = time.time()

V_matrices = []
U_matrices = []

yago = 7.0

start_time = time.time()

lambda_vector = []

input_file = sys.argv[1]


gamma = 2.0
gammainv = 1/gamma

cov_sigma = 1/10.0

d = 5 
	
	
hola = pandas.read_csv(input_file,header=None,names=['UserID', 'MovieID', 'Rating'],usecols=[0,1,2])

print len(hola)

hola2 = hola.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)

power = lambda x: x*x

norma = lambda x: (gamma/2.0)*np.linalg.norm(x)*np.linalg.norm(x)


### iniciar V

cov1 = np.identity(d)
mean = np.identity(d)

cov1 = gammainv*cov1

vj = {}


#v_length = len(hola['MovieID'].unique())
#u_length = len(hola['UserID'].unique())

#print v_length

list_of_users = hola['UserID'].unique()
list_of_movies = hola['MovieID'].unique()

len_users = len(list_of_users)
len_movies = len(list_of_movies)

print len_users
print len_movies

for v1 in list_of_movies:
	
	
	#vj.append(np.random.multivariate_normal(np.zeros(5),cov1))
	vj[v1] = np.random.multivariate_normal(5*np.ones(d),cov1)
	#vj[v1] = np.ones(5)


vj = pandas.DataFrame.from_dict(vj)


## iniciar U 

ui = {}

for u1 in list_of_users:
	
	#ui.append(np.zeros(d))
	ui[u1]=np.zeros(d)

ui = pandas.DataFrame.from_dict(ui)

user_movies = {}



for i in list_of_users:
	
	user_movies[i] = hola[hola['UserID']==i]['MovieID'].tolist()

movie_users = {}

for j in list_of_movies:
	
	movie_users[j] = hola[hola['MovieID']==j]['UserID'].tolist()




for iteration in range(50):
	
	#print "iteration"
	#print iteration
	#start_time = time.time()
	#start_time1 = time.time()
	#print iteration
	## user update
	
	for i in list_of_users:
	
		#user_movies = hola[hola['UserID']==i]
		#print i
		vjs = vj[user_movies[i]]
		#print vjs
		

		
		suma2 = np.dot(vjs,vjs.T) + gamma*cov_sigma*np.identity(d)
		
		
		denom = np.linalg.inv(suma2)
		

		#ms = user_movies['Rating']
		ms = hola2.ix[i][user_movies[i]]
		
		
		ty = np.dot(vjs,ms)
		
		ssq = denom.dot(ty) + yago*np.ones(d)
		
		ssq = ssq/np.linalg.norm(ssq)
		
		
		ui[i] = ssq
		
		
		
		
		
		
		
		
	#print("--- %s seconds user update ---" % (time.time() - start_time))
	
	U_matrices.append(ui.T)
	
	
	###movie update
	#start_time = time.time()

	
	for j in list_of_movies:
		
	
		
		#movie_users = hola[hola['MovieID']==j]
		
		uis = ui[movie_users[j]]

		suma2 = np.dot(uis,uis.T) + gamma*cov_sigma*np.identity(d)
		
		
		
		denom = np.linalg.inv(suma2)

		
		

		ms1 = hola2.ix[movie_users[j]][j]

		
		
		
		
		ty1 = np.dot(uis,ms1)
		ssq1 = denom.dot(ty1) + yago*np.ones(d)
		
		ssq1 = ssq1/np.linalg.norm(ssq1)

		vj[j] = ssq1 
	
	V_matrices.append(vj.T)
	#print("--- %s seconds movie update ---" % (time.time() - start_time))
	####objective function
	#start_time = time.time()
	#primer sumando
	
	
	
	sf = (hola2 - np.dot(ui.T,vj))
	
	
	
	
	sf = sf.apply(power)
	
	sf = sf.sum().sum()
	
	sumatorio = sf /  (2.0*cov_sigma)
	
	#print sumatorio
	
	#ui_1 = ui.apply(norma)
	

	#print len(list_of_users)
	
	
	#sumatorio2 = ui_1.sum()
	
	#print sumatorio2
	#raw_input()
	#vj_1 = vj.apply(norma)
	
	#sumatorio3 = vj_1.sum()
	
	#print sumatorio3

	lambda1 = sumatorio + len_users*1.0 + len_movies*1.0
	lambda1 = lambda1*(-1.0)
	#print lambda1
	lambda_vector.append(lambda1)
	itt = iteration+1
	
	#print("--- %s seconds iteration update ---" % (time.time() - start_time))
	

print("--- %s seconds finish ---" % (time.time() - start_time))

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")

		
namefile4 = "objective.csv"
with open(namefile4, 'w') as csvfile4:
	for la in lambda_vector:
		csvfile4.write(str(la))
		csvfile4.write('\n')

csvfile4.close()

#print("--- %s seconds ---" % (time.time() - start_time))


















	
	