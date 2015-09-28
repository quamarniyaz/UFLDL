%% Softmax Exercise
inputSize = 28 * 28; 
numClasses = 10;     
lambda = 1e-4; 

images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10;
inputData = images;

%{
DEBUG = false; 
if DEBUG
    inputSize = 8;
		load('rand.mat') ;
		load('rlab.mat') ;
		load('rtheta.mat') ;    
end
[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
if DEBUG
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                    inputSize, lambda, inputData, labels), theta);

    disp([numGrad(1:10) grad(1:10)]); 
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
end
%}
options.maxIter = 100;
options.Method  = 'lbfgs' ;
options.useMex  = 0      ;
softmaxModel    = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);

images = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
labels(labels==0) = 10; 
inputData = images;
[pred] = softmaxPredict(softmaxModel, inputData);
acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
