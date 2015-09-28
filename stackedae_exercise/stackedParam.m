%==========STEP 0========================================================
inputSize 		= 784  ;
numClasses 		= 10   ;
hiddenSizeL1	= 200  ;    
hiddenSizeL2	= 200  ;    
sparsityParam = 0.1  ;                       
lambda 				= 3e-3 ;
maxIter       = 15  ;                
beta 					= 3    ;            

%==========STEP 1========================================================
trainData = loadMNISTImages('mnist/train-images-idx3-ubyte')   ;
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte') ;
trainLabels(trainLabels == 0) = 10 ;
trainLabels = trainLabels' ;
clk1 = clock() ;
%==========STEP 2========================================================
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);
opttheta  = sae1Theta     ;
addpath minFunc/
options.Method  = 'lbfgs' ; 
options.maxIter = maxIter ;	
options.useMex  = 0       ; 
options.display = 'on'    ;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                              sae1Theta, options);

sae1OptTheta = opttheta ;
fprintf("Optimal theta for first layer has been learnt.\n") ;
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
opttheta  = sae2Theta     ;
addpath minFunc/
options.Method  = 'lbfgs' ; 
options.maxIter = maxIter ;	
options.useMex  = 0       ; 
options.display = 'on'    ;
patches = sae1Features    ;

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              sae2Theta, options);
sae2OptTheta = opttheta   ;
fprintf("Optimal theta for second layer has been learnt.\n") ;

%==========STEP: 3=====================================================
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
softmaxModel 	 = struct;  
lambda         = 1e-4 ;
options.maxIter= 100  ;
softmaxModel 	 = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options) ;
saeSoftmaxOptTheta = softmaxModel.optTheta(:) ;

%% STEP 5: Finetune softmax model
% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
% -------------------------------------------------------------------------
[optTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, hiddenSizeL2, ...
                                              numClasses, netconfig, ...
                                              lambda, trainData, trainLabels), ...
                              stackedAETheta, options);
stackedAEOptTheta = optTheta ;
clk2 = clock() ;
fprintf("Total elapsed time: %0.3f\n",etime(clk2,clk1));
%%======================================================================
%% STEP 6: Test 
testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
[m, n] = size(stackedAETheta) ;
fprintf('size of stackedAETheta %d %d\n',m, n ) ;
[m, n] = size(stackedAEOptTheta) ;
fprintf('size of stackedAEOptTheta %d %d\n',m, n ) ;
[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);


