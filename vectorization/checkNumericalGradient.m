function [] = checkNumericalGradient()
	x = [4; 10];
	[value, grad] = simpleQuadraticFunction(x);
	numgrad = computeNumericalGradient(@simpleQuadraticFunction, x);
	disp([numgrad grad]);
	fprintf('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');
	diff = norm(numgrad-grad)/norm(numgrad+grad);
	disp(diff); 
	fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');
end

function [value,grad] = simpleQuadraticFunction(x)
	value 	 = x(1)^2 + 3*x(1)*x(2);
	grad 		 = zeros(2, 1);
	grad(1)  = 2*x(1) + 3*x(2);
	grad(2)  = 3*x(1);
end
