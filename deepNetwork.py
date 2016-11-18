import numpy as np
import random
import json 
import sys

class QuadraticCost(object):
	@staticmethod
	def fn(a,y):
		return (0.5*np.linalg.norm(a-y)**2)
	@staticmethod
	def delta(z,a,y):
		return (x-y)*sigmoid_derivative(z)

class CrossEntropyCost(object):
	@staticmethod
	def fn(a,y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
	@staticmethod
	def delta(z,a,y):
		return (a-y)

class network(object):
	def __init__(self,sizes,cost=CrossEntropyCost):
		self.sizes = sizes
		self.layers = len(sizes)#.random.randn
		self.default_weight_init()
		self.CostMethod = cost

	def default_weight_init(self):
		sizes = self.sizes
		self.weights = [np.random.randn(x,y)/np.sqrt(y) for x,y in zip(sizes[1:],sizes[:-1])]
		self.biases = [np.zeros((x,1)) for x in sizes[1:]]

	def large_weight_init(self):
		sizes = self.sizes
		self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
		self.biases = [np.zeros((x,1)) for x in sizes[1:]]

 	def feedforward(self,x):
 		""" caculate the output of the network"""
 		if len(x) != self.sizes[0]:
 			raise "input amount isn't match the network"
 		y = x.reshape(-1,1)
 		for w,b in zip(self.weights,self.biases):
 			y = sigmoid(np.dot(w,y)+b)
 		return y

 	
	def SGD(self,epoch,mini_batchs_sizes,eta,training_data,lmbda=0.0,
			test_data=None,evaluate_accuracy=False,evaluate_cost=False,
			training_accuracy=False,training_cost=False):
		L = len(test_data)
		Len_tra_data = len(training_data)
		for j in xrange(epoch):
			random.shuffle(training_data)
			n = len(training_data)
			for i in xrange(0,n,mini_batchs_sizes):
				if i+2*mini_batchs_sizes> n:
					mini_batch = training_data[i:]
				else:
					mini_batch = training_data[i:i+mini_batchs_sizes]
				self.weights = [w*(1-eta*lmbda/n) for w in self.weights]
				self.update_paremeter(mini_batch,eta)
			print "epoch {0}:".format(j)
			if test_data:
				if evaluate_accuracy:
					right = self.accuracy(test_data)
					print "evaluate accuracy-> {0}/{1}".format(right,L)
				if evaluate_cost:
					Cost = self.total_cost(test_data,lmbda)
					print "evaluate cost-> {0}".format(Cost)
			if training_accuracy:
				right = self.accuracy(training_data)
				print "training accuracy-> {0}/{1}".format(right,Len_tra_data)
			if training_cost:
				Cost = self.total_cost(training_data,lmbda)
				print "training cost-> {0}".format(Cost)
			else:
				print "complete"
	
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
		ceta = self.CostMethod.delta(Zs[-1],x,y)   #BP1
		graded_b[-1] = ceta    #BP3
		graded_w[-1] = np.dot(ceta,cell_out[-1].T)   #BP4
		for i in xrange(1,self.layers-1):
			ceta = np.dot(self.weights[-1*i].T,ceta)*sigmoid_derivative(Zs[-i-1])  #BP2
			graded_b[-i-1] = ceta    #BP3
			graded_w[-i-1] = np.dot(ceta,cell_out[-i-1].T)   #BP4
		return (graded_b,graded_w)

	def accuracy(self,data):
		result = [(np.argmax( self.feedforward(x)),np.argmax(y)) for (x,y) in data]
		return sum([int(x == y) for (x,y) in result])

	def total_cost(self,data,lmbda):
		cost = 0
		n = len(data)
		for x,y in data:
			a = self.feedforward(x)
			cost += self.CostMethod.fn(a,y)/n
		cost += 0.5*lmbda/n*sum(np.linalg.norm(w)**2 for w in self.weights)
		return cost
			
	def save(self,filename):
		data = {"sizes":self.sizes,
				"cost":str(self.CostMethod.__name__),
				"weights":[w.tolist() for w in self.weights],
				"biases":[b.tolist() for b in self.biases],}
		f = open(filename,'w')
		json.dump(data,f)
		f.close()

def load(filename):
	f = open(filename)
	data = json.load(f)
	cost = getattr(sys.modules[__name__],data["cost"])
	net = network(data["sizes"],cost=cost)
	net.weights = [np.array(w) for w in data["weights"]]
	net.biases = [np.array(b) for b in data["biases"]]
	return net

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
	return sigmoid(z)*(1-sigmoid(z))

if __name__ == "__main__":
	import mnist_loader
	tra,val,tes = mnist_loader.load_data_wrapper()
	#net = network([784,30,10],)
	net = load("001")
	net.SGD(10,10,1,tra,lmbda=0.5,test_data=tes,
		evaluate_accuracy=True,evaluate_cost=True,
		training_accuracy=True,training_cost=True)
	net.save('001')