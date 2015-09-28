function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
	softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, 		hiddenSize);
	stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
	softmaxThetaGrad = zeros(size(softmaxTheta));
	stackgrad = cell(size(stack));
	for d = 1:numel(stack)
		stackgrad{d}.w = zeros(size(stack{d}.w));
	  stackgrad{d}.b = zeros(size(stack{d}.b));
	end

	cost = 0; 
	numCases = size(data, 2);
	groundTruth = full(sparse(labels, 1:numCases, 1));

	a_1 = data ;
	
	z_2 = stack{1}.w * a_1 + repmat(stack{1}.b, 1, numCases) ;
	a_2 = sigmoid(z_2) ;
	
	z_3 = stack{2}.w * a_2 + repmat(stack{2}.b, 1, numCases) ;	
	a_3 = sigmoid(z_3) ;

	z_4 = softmaxTheta * a_3 ;
	M   = z_4 ;
	M = bsxfun(@minus, M, max(M, [], 1));
	expM = exp(M) ;
	h = bsxfun(@rdivide, expM, sum(expM)) ;
	
	decay = (lambda * softmaxTheta(:)'*softmaxTheta(:))/2 ;
	cost = (-1*sum(sum((groundTruth .* log(h)))))/numCases + decay ;
	
	delta_4 = -1 *(groundTruth-h) ;
	delta_3 = (softmaxTheta' * delta_4).*fprime(z_3) ;
	tmpW    = stack{2}.w ;
	delta_2 = (tmpW' * delta_3).*fprime(z_2) ;
	
	softmaxThetaGrad = (-1* (groundTruth -h) * a_3')/numCases + lambda*softmaxTheta ;
	stackgrad{2}.w = delta_3 * a_2'/numCases;
	stackgrad{1}.w = delta_2 * a_1'/numCases ;
	stackgrad{2}.b = sum(delta_3, 2)/numCases ;
	stackgrad{1}.b = sum(delta_2, 2)/numCases ;
	grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];
	%grad = [softmaxThetaGrad(:) ; stackgrad{1}.w ; stackgrad{2}.w ; stackgrad{1}.b ; stackgrad{2}.b ] ;
  [m, n] = size(grad) ;
	fprintf('Size of grad: %d %d\n', m, n) ;
end

function sigm = sigmoid(x)
    sigm = 1./(1 + exp(-x));
end

function sigmp = fprime(x)
	sigmp = sigmoid(x).*(1-sigmoid(x)) ;
end
