import numpy as np

def randInitializeWeights(L_in,L_out):
	W = np.zeros((L_out,L_in+1))
	Init_epsilon = 0.12;

	W = np.random.rand(L_out,1+L_in)*2*Init_epsilon - Init_epsilon

	return W	