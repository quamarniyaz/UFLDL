function [pred] = softmaxPredict(softmaxModel, data)

	theta = softmaxModel.optTheta;  
	pred = zeros(1, size(data, 2));

	M = theta * data ;
	M = bsxfun(@minus, M, max(M, [], 1));
	expM = exp(M) ;
	h = bsxfun(@rdivide, expM, sum(expM)) ;
	[maxP, ind] = max(h, [], 1) ;
	pred = ind ;
end

