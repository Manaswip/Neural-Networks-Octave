import numpy as np

def computeNumericalGradient(J,Theta):

	numgrad = np.zeros((Theta.shape))
	per = np.zeros((Theta.shape))

	e = 1e-4;

	for i in range(Theta.size):
		per(i) = e;
		loss1 = J(Theta-per);
		loss2 = J(Theta+per);
		numgrad(p) = (loss2-loss1) / (2*e);
		per(p)=0;

	return numgrad;



