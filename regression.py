import theano
import theano.tensor as T
import numpy as np
import cPickle

def pickleSave(obj, filename):
	f = file(filename, 'wb')
	cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

class regression(object):

	'''Implementation of Linear Regression'''
	
	def __init__(self, data):
		self.data = data
		self.n = self.data.getShape()[0][0]
		self.X = T.dmatrix('X')
		self.Y = T.dmatrix('Y')
		self.w = theano.shared(np.random.randn(self.data.getShape()[0][1]), name='w')
		self.b = theano.shared(0., name='b')
		self.model = self.X * self.w + self.b
		self.cost = (1.0 / (2.0*self.n))*T.sum((self.Y - self.model)**2)


	def train(self, alpha, steps):
		gw = T.grad(self.cost, self.w)
		gb = T.grad(self.cost, self.b)
		updates = ((self.w, self.w - alpha * gw), (self.b , self.b - alpha*gb))
		training = theano.function(inputs=[self.X, self.Y], outputs=[self.cost], updates=updates, allow_input_downcast=True)
		for i in range(steps):
			err = training(self.data.getX(), self.data.getY())
			print (i, err)
		output = (self.b.get_value(), self.w.get_value())
		pickleSave(output, 'weights.pickle')

	def predict(self, dat):
		prediction = theano.function(inputs=[self.X], outputs=[self.model])
		predictions = prediction(dat)
		pickleSave(predictions, 'predictions.pickle')

class ridge(regression):

	'''Implementation of Ridge Regression'''

	def __init__(self, data, reg_par):
		regression.__init__(self, data)
		self.reg_par = reg_par
		self.cost = self.cost + self.reg_par*(T.sum(self.w**2) + self.b**2)

class lasso(regression):

	'''Implementation of Lasso Regression'''

	def __init__(self, data, reg_par):
		regression.__init__(self, data)
		self.reg_par = reg_par
		self.cost = self.cost + self.reg_par*(T.sum(abs(self.w)) + abs(self.b))

class elastic(regression):

	'''Implementation of Elastic Nets'''

	def __init__(self, data, reg_par, mix_par):
		regression.__init__(self, data)
		self.reg_par = reg_par
		self.mix_par = mix_par
		self.cost = self.cost + self.reg_par*self.mix_par*(T.sum(abs(self.w)) + abs(self.b)) + self.reg_par*(1 - self.mix_par)*(T.sum(self.w**2) + self.b**2)

class logistic(regression):

	'''Implementation of Logistic Regression'''
	def __init__(self, data):
	regression.__init__(self, data)
	self.model = 1 / (1 + T.exp(-T.dot(self.X, self.w) - self.b))
	self.cost = (1.0 / (2.0*self.n))*T.sum(-self.Y * T.log(self.model) - (1-self.Y) * T.log(1-self.model))






