from numpy import linalg as LA
import debugInitializeWeights as d
import nnCostFunction as n
import computeNumericalGradient as c
import numpy as np

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;
lambda1 = 0;

# We generate some 'random' test data
Theta1 = d.debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = d.debugInitializeWeights(num_labels, hidden_layer_size);
# Reusing debugInitializeWeights to generate X
X  = d.debugInitializeWeights(m, input_layer_size - 1);
y  = np.mod(np.arange(1,m+1),num_labels) ;


nn_params = np.concatenate([(np.transpose(Theta1)).ravel(),(np.transpose(Theta2)).ravel()])

costFunct = lambda p:n.nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1,0);
gradFunct = lambda p:n.nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1,1);
grad = gradFunct(nn_params);

numgrad = c.computeNumericalGradient(costFunct, nn_params);

diff = LA.norm(numgrad-grad)/LA.norm(numgrad+grad);

print diff

