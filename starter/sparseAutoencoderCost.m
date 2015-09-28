function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

	W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
	W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
	b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
	b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

	cost = 0;
	W1grad = zeros(size(W1)); 
	W2grad = zeros(size(W2));
	b1grad = zeros(size(b1)); 
	b2grad = zeros(size(b2));

	rhocap = zeros(size(hiddenSize, 1)) ;
	m = size(data,2) ;
	for i=1:m
		X = data(:,i) ;
		a_1 = X ;
		
		z_2 = W1 * X + b1 ;
		a_2 = sigmoid(z_2);
		rhocap = rhocap + a_2 ;
	
		z_3 = W2 * a_2 + b2 ;
		a_3 = sigmoid(z_3)  ;
		cost = cost  + ((a_3-X)'*(a_3-X))/2 ;
	end

	rhocap = rhocap / m ;
	KL 		 = sparsityParam*log(sparsityParam./rhocap) + (1-sparsityParam)*log((1-sparsityParam)./(1-rhocap)) ;
	w_ij = [ W1(:) ; W2(:) ] ;
	cost = cost/m + (lambda/2)*(w_ij'*w_ij) + beta * sum(KL)  ;

	for i=1:m
		X = data(:,i) ;
		a_1 = X ;
		z_2 = W1 * X + b1 ;
		a_2 = sigmoid(z_2);
		z_3 = W2 * a_2 + b2 ;
		a_3 = sigmoid(z_3)  ;
		sigGrad = a_3 .* (1-a_3) ; 
		delta_3 = -(X - a_3 ) .* sigGrad ;
		sigGrad = a_2 .* (1-a_2) ;
		delta_2 = ( W2' * delta_3 + beta*(-1*(sparsityParam./rhocap) + (1 - sparsityParam)./(1-rhocap))) .* sigGrad ;
	
		W1grad = W1grad + delta_2 * a_1' ;
		W2grad = W2grad + delta_3 * a_2' ;
	
		b1grad = b1grad + delta_2 ;
		b2grad = b2grad + delta_3 ;    
	end
      
	W1grad = W1grad/m + lambda*W1 ;
	W2grad = W2grad/m + lambda*W2 ; 
	b1grad = b1grad/m ;
	b2grad = b2grad/m ;
	
	grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
	
end

function sigm = sigmoid(x)
	sigm = 1 ./ (1 + exp(-x));
end

