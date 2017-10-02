class RidgeRegression(object):  
	def __init__(self, lmbda=0.1):
		self.lmbda = lmbda

	def fit(self, X, y):
		C = X.T.dot(X) + self.lmbda*numpy.eye(X.shape[1])
		self.w = numpy.linalg.inv(C).dot(X.T.dot(y))))

	def predict(self, X):
		return X.dot(self.w)

	def get_params(self, deep=True):
		return {"lmbda": self.lmbda}

	def set_params(self, lmbda=0.1):
		self.lmbda = lmbda
		return self
		
		
Xy = numpy.loadtxt(data_dir + "winequality-white.csv", delimiter=",", skiprows=1)

X = Xy[:, 0:-1]
X = scale(X)

y = Xy[:, -1]
y -= y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)