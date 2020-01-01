function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;

% regularize theta by removing first value
theta_reg = [0;theta(2:end, :);];
J = (1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*theta_reg'*theta_reg;

% The gradient of the cost function consist of two parts, for j = 0 and j bigger then 0
% in case J = 0, the regularized term disappear. So we can create a vector theta_reg where theta_reg(1) = 0
% Otherwise we can calculate seperately j = 0   then the rest and make a new gradient from these parts.


grad = (1/m)*(X'*(h-y)+lambda*theta_reg);









% =========================================================================

grad = grad(:);

end
