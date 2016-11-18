import numpy as np
import random
class network(object):
	def __init__(self,sizes):
		self.sizes = sizes
		self.layers = len(sizes)
		self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
		self.biases = [np.random.randn(x,1) for x in sizes[1:]]

 	def feedforward(self,x):
 		""" caculate the output of the network"""
 		if len(x) != self.sizes[0]:
 			raise "input amount isn't match the network"
 		y = x.reshape(-1,1)
 		for w,b in zip(self.weights,self.biases):
 			y = sigmoid(np.dot(w,y)+b)
 		return y

 	
	def SGD(self,epoch,mini_batchs_sizes,eta,training_data,test_data=None):
		L = len(test_data)
		for j in xrange(epoch):
			random.shuffle(training_data)
			n = len(training_data)
			for i in xrange(0,n,mini_batchs_sizes):
				if i+2*mini_batchs_sizes> n:
					mini_batch = training_data[i:]
				else:
					mini_batch = training_data[i:i+mini_batchs_sizes]
				self.update_paremeter(mini_batch,eta)
			if test_data:
				right = self.evaluate(test_data)
				print "epoch {0} : {1}/{2}".format(j,right,L)
			else:
				print "epoch {0} complete".format(j)
	
	def update_paremeter(self,mini_batch,eta):
		m = len(mini_batch)
		delta_b = [np.zeros(b.shape) for b in self.biases]
		delta_w = [np.zeros(w.shape) for w in self.weights]
		for sam in mini_batch:
			(graded_b,graded_w) = self.backprop(sam[0],sam[1])
			delta_b = [nb+ngb for nb,ngb in zip(delta_b,graded_b)]
			delta_w = [nw+ngw for nw,ngw in zip(delta_w,graded_w)]
		self.biases = [b-eta*nb/m for b,nb in zip(self.biases,delta_b)]
		self.weights = [w-eta*nw/m for w,nw in zip(self.weights,delta_w)]

	
	def backprop(self,x,y):
		n = len(y)
		cell_out = [x]
		Zs = []
		for w,b in zip(self.weights,self.biases):
			z = np.dot(w,x)+b
 			x = sigmoid(z)
 			cell_out.append(x)
 			Zs.append(z)
 		cell_out.pop(-1)
		graded_w = [np.zeros(w.shape) for w in self.weights]
		graded_b = [np.zeros(b.shape) for b in self.biases]
		ceta = (x-y)*sigmoid_derivative(Zs[-1])     #BP1
		graded_b[-1] = ceta    #BP3
		graded_w[-1] = np.dot(ceta,cell_out[-1].T)   #BP4
		for i in xrange(1,self.layers-1):
			ceta = np.dot(self.weights[-1*i].T,ceta)*sigmoid_derivative(Zs[-i-1])  #BP2
			graded_b[-i-1] = ceta    #BP3
			graded_w[-i-1] = np.dot(ceta,cell_out[-i-1].T)   #BP4
		return (graded_b,graded_w)

	def evaluate(self,test_data):
		result = [(np.argmax( self.feedforward(x)),y) for (x,y) in test_data]
		return sum([int(x == y) for (x,y) in result])

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
	return sigmoid(z)*(1-sigmoid(z))


if __name__ == "__main__":
	import mnist_loader
	tra,val,tes = mnist_loader.load_data_wrapper()
	net = network([784,30,10,10])
	net.SGD(10,10,3.0,tra,tes)