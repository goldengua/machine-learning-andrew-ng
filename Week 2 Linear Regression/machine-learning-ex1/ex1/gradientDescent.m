function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % loop solution
    temp = theta;
    for t = 1: length(temp)
       
        predictions = X*theta;
        currentFeatureVector = X(:, t);
        deltaVector = (predictions - y);
        toSum = deltaVector.*currentFeatureVector;
        
        temp(t) = theta(t) -(alpha/m)*sum(toSum);
    end
    %vectorization in one itteration:
    %the hypothisys minus y. this is a vertical vector
    deltas = (X*theta)- y;
    % in order to make the vectorication we set the dimentions strait:
    % deltas is 92*1 now, X is 92*2.
    % vectorization means that I want a vector that will represent two
    % sums:
    % one for each feacher... deltas * each feacher (like in the loop solution
    % the the vectorization will automatically summerize each
    % moltiplication between deltas and each feacher to a single value:
      
    theta = theta -(alpha/m).*((deltas)'*X)'
    
    %can see that the loop solution (temp) is the same as the
    %vectorization(theta)
theta = temp;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
