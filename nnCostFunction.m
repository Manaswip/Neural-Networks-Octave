function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%appending boas unit(ones) to X
X = [ones(m,1) X];

%Number of features
n = size(X,2);

%Converting y's to matrix
new_y = zeros(num_labels,m);
first_half = zeros(num_labels,m);
second_hlaf = zeros(num_labels,m);

for i=1:m,
	new_y(y(i),i)=1;
end;

first_layer_activation = zeros(hidden_layer_size,m);
first_layer_activation = sigmoid(Theta1*X');

%appending bias unit(ones) to first layer activations	
first_layer_activation 	= [ones(1,m) ;first_layer_activation];

hypFunction = sigmoid(Theta2*first_layer_activation);

first_half = sum(sum(new_y.*log(hypFunction)),2);
second_half = sum(sum((1-new_y).*log(1-hypFunction)),2);

% Calculating new unrolled Theta by removing bias units.
new_Theta1 =  Theta1(:,2:n);
new_Theta2 = Theta2 (:,2:(hidden_layer_size+1));
new_unrolled_Theta = [new_Theta1(:); new_Theta2(:)];

J = ((-1/m)*(first_half+second_half)) + (lambda/(2*m) *(sum(new_unrolled_Theta.*new_unrolled_Theta)));

% Calculating partial derivative terms using backpropagation algorithm

delta1_grad = zeros(size(Theta1));
delta2_grad = zeros(size(Theta2));


for t=1:m,
	a1 = X(t,:);
	a2 = sigmoid(Theta1*a1');
	a2 = [ones(1,1); a2];
	a3 = sigmoid(Theta2*a2);

	delta3 = a3 - new_y(:,t);
	delta = Theta2'*delta3; 
	delta2 = (delta(2:end, :)).* sigmoidGradient(Theta1*a1');

	delta1_grad = delta1_grad + delta2 * a1;
	delta2_grad = delta2_grad + delta3 * a2';
end;


Theta1_grad = (1/m) * delta1_grad;
Theta2_grad = (1/m) * delta2_grad;

Theta1_grad(:,2:end) =  Theta1_grad(:,2:end) + (Theta1(:,2:end)) * (lambda/m);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) +  (Theta2(:,2:end)) * (lambda/m);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
