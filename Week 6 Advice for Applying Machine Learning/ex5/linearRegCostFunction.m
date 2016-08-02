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
thetaToSum = theta;
thetaToSum(1) = 0; %Note that you should not regularize the theta(1) term.
hepothesis = X*theta;
sm = hepothesis-y;
tosm=sm.^2;
ger= (lambda/(2*m))*sum(thetaToSum.^2);
J = sum(tosm)*(1/(2*m))+ger;


%grad part
theta(1) = 0 ;
coreInSegma = (sm'*X)./m;
generalization = ((lambda/m).*theta)';
grad = coreInSegma+generalization;










% =========================================================================

grad = grad(:);

end
