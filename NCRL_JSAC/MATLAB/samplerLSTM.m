function [state,action,target] = samplerLSTM(Q,M,batch_size,num_nodes)

sampled = datasample(M,batch_size,'Replace',false);

state = cell(batch_size,1);

target = zeros(batch_size,num_nodes);
action = zeros(batch_size,1);
gamma = 1;



for i=1:batch_size
    q = zeros(1,num_nodes);
    if size(sampled{i}{3},2) > 1 %not terminal
        q = Q.predict({[sampled{i}{3}]});
    end
    MAX = max(q);
    target(i,sampled{i}{2}) = gamma *  MAX + sampled{i}{4};
    state{i} = sampled{i}{1};
    action(i) = sampled{i}{2}(end);
end