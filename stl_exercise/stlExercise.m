%% stlExercise
inputSize  		= 28 * 28;
numLabels  		= 5		 ;
hiddenSize 		= 200    ;
sparsityParam	= 0.1    ; 
lambda 			  = 3e-3	 ;	      
beta 			    = 3		 ;    
maxIter 	 	  = 400    ;

mnistData   = loadMNISTImages('train-images-idx3-ubyte');
mnistLabels = loadMNISTLabels('train-labels-idx1-ubyte');

labeledSet    = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet  = find(mnistLabels >= 5);
unlabeledData = mnistData(:, unlabeledSet);

numTrain    = round(numel(labeledSet)/2);
trainSet    = labeledSet(1:numTrain)    ;
testSet     = labeledSet(numTrain+1:end);
trainData   = mnistData(:, trainSet)    ;
trainLabels = mnistLabels(trainSet)' + 1; 
testData    = mnistData(:, testSet)     ;
testLabels  = mnistLabels(testSet)' + 1 ;


theta = initializeParameters(hiddenSize, inputSize);
opttheta = theta ; 
addpath minFunc/
options.Method  = 'lbfgs'; 
options.maxIter = maxIter;	
options.useMex  = 0  ; 
options.display = 'on';
patches = unlabeledData ;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);
fprintf("Optimal theta has been learnt.\n") ;

W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);
testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

softmaxModel = struct;  
lambda = 1e-4 ;
options.maxIter = 100;
inputSize = size(trainFeatures, 1) ;
numClasses = 5 ;
inputData = trainFeatures ;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, trainLabels, options);
fprintf("Softmax training is completed.\n") ;

[pred] = softmaxPredict(softmaxModel, testFeatures);
acc = mean(testLabels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
