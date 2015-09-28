%% Sparse Auto-encoder
visibleSize   = 8*8   ;    
hiddenSize    = 25    ;      
sparsityParam = 0.01  ;    
lambda        = 0.0001;           
beta          = 3     ;                 

patches = sampleIMAGES;
display_network(patches(:,randi(size(patches,2),200,1)),8);
printf("Patches are loaded. Press any key to train the network ....\n") ;
pause ;

%{
[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, patches);
%}

theta = initializeParameters(hiddenSize, visibleSize);
addpath minFunc/
options.Method   = 'lbfgs'; 
options.maxIter  = 400    ;	
options.useMex   = 0      ; 
options.display  = 'on'   ;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);


W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 
print -djpeg weights.jpg 


