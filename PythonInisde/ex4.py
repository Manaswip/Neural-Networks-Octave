import numpy as np
import nnCostFunction as n
import randInitializeWeights as p
import scipy.io as sio
import ExtractingData as e

# Initializing variables
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

#Loading data
data = sio.loadmat('ex4data1.mat')
X = data.get('X')
y = data.get('y')
m = X.shape[0]

#Loading weights
Theta1 = p.randInitializeWeights(400,25)
Theta2 = p.randInitializeWeights(25,10)
nn_params = np.concatenate([(np.transpose(Theta1)).ravel(),(np.transpose(Theta2)).ravel()])

#Cost Function
lambda1 = 0;

J = n.nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1);