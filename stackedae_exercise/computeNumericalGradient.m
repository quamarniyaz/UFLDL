function numgrad = computeNumericalGradient(J, theta)
	numgrad = zeros(size(theta));
	n = size(theta,1) ;
	EPSILON = .0001 ;
	for i=1:n
		theta(i) = theta(i) + EPSILON ;
		J_plus   = J(theta) ;		
		theta(i) = theta(i) - 2 * EPSILON ;
		J_minus  = J(theta) ;
		numgrad(i) = (J_plus - J_minus)/(2 * EPSILON ) ;
		theta(i) = theta(i) + EPSILON ;
	end
end
