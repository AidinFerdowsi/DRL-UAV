sigma_2 = 10^(-13);
B = 1;
S = 10;
beta = 0.00002;
h = 80;
%%
e_max = 1;
e_min = 0.1;

E_max = e_max * beta/(2^(S/B)-1)/sigma_2;
E_min = e_min * beta/(2^(S/B)-1)/sigma_2;
lb = zeros(10000,10);
for i=1:10000
    for n = 1:10
        E = E_min + rand(n,1) * (E_max-E_min);
        lambda = rand(n,1);
        lambda = lambda/sum(lambda);
        lb(i,n) = lowerbound(E,lambda,h);
    end
end
lb = mean(lb);
%%
lb = zeros(10000,10);
n=4;
for e = 1:10
    for i=1:10000
        e_max = 0.05 + e*0.1;
        
        e_min = 0.05 + (e-1) * 0.1;

        E_max = e_max * beta/(2^(S/B)-1)/sigma_2;
        E_min = e_min * beta/(2^(S/B)-1)/sigma_2;

        E = E_min + rand(n,1) * (E_max-E_min);
        lambda = rand(n,1);
        lambda = lambda/sum(lambda);
        lb(i,e) = lowerbound(E,lambda,h);
    end
end
lb = mean(lb);