import numpy as np
class Variable:
	"""docstring for Variable"""
	def __init__(self, data):
		self.data = data
class Function:
	"""docstring for ClassName"""
	def __call__(self, input):
		x=input.data
		y=self.forward(x)
		output=Variable(y)
		return output
	def forward(self,x):
		raise NotImplementedError()
class Square(Function):
	def forward(self,x):
		return x**2

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)