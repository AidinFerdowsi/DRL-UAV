function [s_prev,act,target] = sampler(Q,M,batch_size)

sampled = datasample(M,batch_size,'Replace',false);

state_size = length(sampled{1}{1});
s_prev=zeros(state_size,1,1,batch_size);
act = zeros(4,1,1,batch_size);
target = zeros(batch_size,1);
gamma = 0.1;



for i=1:batch_size
    max = -inf;
    dxi = sampled{i}{5}{1};
    dyi = sampled{i}{5}{2};
    E = sampled{i}{5}{3};
    lambda = sampled{i}{5}{4};
    for a = 1:length(dxi)
        action = [dxi(a);dyi(a);E(a);lambda(a)];
        q_temp = Q.predict([sampled{i}{3};action]);
        if isnan(q_temp)
            q_temp
        end
        if q_temp >= max
            max = q_temp;
            ai = action;
        end
    end
    target(i) = gamma * max + sampled{i}{4};
    act(:,:,:,i) = ai;
    s_prev(:,:,:,i) = sampled{i}{1};
end