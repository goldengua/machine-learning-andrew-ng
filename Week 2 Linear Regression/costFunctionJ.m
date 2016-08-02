function J = costFunctionJ(X,y,theta)
%X is the 'design matrinx' containing our traning examples
%y is the class lables
m = size(X,1); %# of traning examples
predictions = X*theta; % predictions of the hypothesis on all m examples
sqrError = (predictions - y).^2; %squared Errors
J=(1/(2*m)) * sum(sqrError);