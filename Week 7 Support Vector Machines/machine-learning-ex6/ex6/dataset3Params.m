function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
x1 = 100;
x2 = 100;

CSteps = [0.03;0.3;3;30;300;3000]
SSteps = CSteps./3;


currmin = 1000;
for cStep = 1:size(CSteps)
    for SStep = 1:size(SSteps)
    C =  CSteps(cStep);
    sigma = SSteps(SStep);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    rtn = mean(double(predictions ~= yval));
    
    if rtn<currmin
        minc = C;
        mins = sigma;
        currmin = rtn;
         model = svmTrain(X, y, minc, @(x1, x2) gaussianKernel(x1, x2, mins));
        visualizeBoundary(X, y, model);
    end
end
end
C = minc;
sigma = mins;







% =========================================================================

end
