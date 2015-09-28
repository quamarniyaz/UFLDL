function [] = checkNNTest()
	load('data.mat') ;
	load('theta.mat')
	X = data(:,1:3) ;
	[value, grad] = sparseAutoencoderCost(theta, 3, 2, 3e-3, 0.1, 3, X);
	numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, 3, 2, 3e-3, 0.1, 3, X), theta);
	disp([numgrad(1:10) grad(1:10)]); 
	fprintf('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');
	diff = norm(numgrad-grad)/norm(numgrad+grad);
	disp(diff); 
	fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');
end

