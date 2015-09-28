function [] = checkStackedAECost()
inputSize = 4;
hiddenSize = 5;
lambda = 0.01;
data   = randn(inputSize, 5);
labels = [ 1 2 1 2 1 ];
numClasses = 2;

stack = cell(2,1);
stack{1}.w = 0.1 * randn(3, inputSize);
stack{1}.b = zeros(3, 1);
stack{2}.w = 0.1 * randn(hiddenSize, 3);
stack{2}.b = zeros(hiddenSize, 1);
softmaxTheta = 0.005 * randn(hiddenSize * numClasses, 1);

[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ softmaxTheta ; stackparams ];


[cost, grad] = stackedAECost(stackedAETheta, inputSize, hiddenSize, ...
                             numClasses, netconfig, ...
                             lambda, data, labels);

numgrad = computeNumericalGradient( @(x) stackedAECost(x, inputSize, ...
                                        hiddenSize, numClasses, netconfig, ...
                                        lambda, data, labels), ...
                                        stackedAETheta);

disp([numgrad grad]); 

disp('Norm between numerical and analytical gradient (should be less than 1e-9)');
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff);
            
            
