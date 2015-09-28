%% Vectorization
visibleSize   = 28*28 ;    
hiddenSize    = 196   ;      
sparsityParam = 0.1   ;    
lambda        = 3e-3  ;           
beta          = 3     ;                 

images = loadMNISTImages('train-images-idx3-ubyte') ;
labels = loadMNISTLabels('train-labels-idx1-ubyte') ;
ind = randperm(size(images, 2)) ;
patches =  images(:,ind(1:10000)) ;
fprintf("Images are loaded. Press any key to continue ...\n") ;
pause ;
display_network(patches(:,1:100)) ;
disp(labels(ind(1:10))) ;

theta = initializeParameters(hiddenSize, visibleSize);
addpath minFunc/
options.Method 	= 'lbfgs'; 
options.maxIter = 400   ;
options.useMex  = 0     ; 
options.display = 'on'  ;
[opttheta, cost]= minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 
print -djpeg weights.jpg  ;
