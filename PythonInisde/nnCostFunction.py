from __future__ import division
import numpy as np
import sigmoid as s
import scipy.io as sio
import sigmoidGradient as si

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_classes,X,y,lambda1,condition):
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
    	((input_layer_size + 1),hidden_layer_size));

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
    ((hidden_layer_size + 1),num_classes));

# Getting back the weights
    Theta1 = np.transpose(Theta1);
    Theta2 = np.transpose(Theta2);
    new_unrolled_Theta = np.concatenate([(np.transpose(Theta1[:,1:])).ravel(),(np.transpose(Theta2[:,1:])).ravel()])
# Number of trainig data sets
    m = X.shape[0];
    Theta1_grad = np.zeros(Theta1.shape);
    Theta2_grad = np.zeros(Theta2.shape);
# Concatenating bias unit
    X = np.concatenate((np.ones((m,1)),X),axis=1)
# Number of features
    n = X.shape[1];

    new_y = np.zeros((num_classes,m))

    for i in range(0,m):
    	new_y[y.item(i),i] = 1;

    first_layer_activation = X;
    second_layer_activation = s.sigmoid(np.dot(Theta1,np.transpose(X)));
# appending bias unit
    second_layer_activation = np.concatenate((np.ones((1,m)),second_layer_activation));
    hypFunction = s.sigmoid(np.dot(Theta2,second_layer_activation));

    first_half = np.sum(np.multiply(new_y,np.log(hypFunction)));
    second_half =np.sum(np.multiply((1-new_y),np.log(1-hypFunction)));

    J = ((-1.0/m)*(first_half+second_half)) + (lambda1/(2*m) *(np.sum(np.multiply(new_unrolled_Theta,new_unrolled_Theta))));
#Calculating delta terms using back propogation
    delta1_grad = np.zeros(Theta1.shape);
    delta2_grad = np.zeros(Theta2.shape);

    for i in range(0,m):
    	a1 = X[i,:][np.newaxis];
    	a2 = s.sigmoid(np.dot(Theta1,np.transpose(a1)));
    	a2 = np.concatenate((np.ones((1,1)),a2));
    	a3 = s.sigmoid(np.dot(Theta2,a2));

    	delta3 = a3 - np.transpose(new_y[:,i][np.newaxis]);
    	delta = np.dot((np.transpose(Theta2)),delta3);
    	delta2 = np.multiply(delta[1:],si.sigmoidGradient(np.dot(Theta1,np.transpose(a1))));

    	delta1_grad = delta1_grad + np.dot(delta2,a1);
    	delta2_grad = delta2_grad + np.dot(delta3,np.transpose(a2));

    Theta1_grad = (1/m) * delta1_grad;
    Theta2_grad = (1/m) * delta2_grad;

    Theta1_grad[:,1:] =  Theta1_grad[:,1:] + (Theta1[:,1:]) * (lambda1/m);
    Theta2_grad[:,1:] = Theta2_grad[:,1:] +  (Theta2[:,1:]) * (lambda1/m);

    grad = np.concatenate([(np.transpose(Theta1_grad)).ravel(),(np.transpose(Theta2_grad)).ravel()]);

    if condition==0:
        return J
    elif condition==1:
        return grad













