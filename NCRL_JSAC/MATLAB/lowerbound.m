function lb = lowerbound(E,lambda,h)
n = floor(E/h^2);
lb = sum(lambda./(n+1));