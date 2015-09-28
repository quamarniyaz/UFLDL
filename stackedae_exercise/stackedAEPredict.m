function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

W1    = stack{1}.w ;
b1    = stack{1}.b ;
W2    = stack{2}.w ;
b2    = stack{2}.b ;

m = size(data, 2) ;
a_1 = data ;

z_2 = W1*data + repmat(b1, 1, m) ;
a_2 = sigmoid(z_2) ;

z_3 = W2*a_2 + repmat(b2, 1, m)  ;
a_3 = sigmoid(z_3) ;


M = softmaxTheta * a_3 ;
M = bsxfun(@minus, M, max(M, [], 1));
expM = exp(M) ;
h = bsxfun(@rdivide, expM, sum(expM)) ;
[maxP, ind] = max(h, [], 1) ;
pred = ind ;
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
