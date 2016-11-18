import numpy as np
import cPickle
import gzip
def load_data():
	f = gzip.open(r'mnist.pkl.gz','rb')
	training_data,validation_data,test_data = cPickle.load(f)
	f.close()
	return (training_data,validation_data,test_data)

def load_data_wrapper():
	tr_d,va_d,te_d = load_data()
	training_inputs = [np.reshape(x,(784,1)) for x in tr_d[0]]
	training_reaults = [vectorized_result(y) for y in tr_d[1]]
	training_data = zip(training_inputs,training_reaults)
	validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
	validation_data = zip(validation_inputs,va_d[1])
	test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
	test_reaults = [vectorized_result(y) for y in te_d[1]]
	test_data = zip(test_inputs,test_reaults)
	return (training_data,validation_data,test_data)

def vectorized_result(j):
	e = np.zeros((10,1))
	e[j] = 1.0
	return e

if __name__ == "__main__":
	tra,val,tes = load_data_wrapper()
	print type(tra[0][0])
	print tra[0][0].shape