import numpy as np
def sigmoid(z):
	g= np.divide(1,(1+ np.exp(-z)))
	return g;
	