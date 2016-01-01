import numpy as np

class datum(object):
	
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def getX(self):
		return self.x

	def getY(self):
		return self.y

	def getShape(self):
		return (self.x.shape, self.y.shape)
