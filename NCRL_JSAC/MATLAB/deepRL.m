% function deepRL
clear all;
num_nodes = 20;

state_per_node = 4; % state_per_node = x,y,energy,lambda:4
const_size = 5; %dest_x,dest_y,vmax_x, vmax_y, tau:5

M = {}; %replay memory 
M_cap = 20000;

num_initial_episodes = 50;
num_training_episodes = 1000;
num_episode_eval = 20;
target_update_episode = 5;
training_period = 1;

EPS = logspace(-0.3,-2,num_training_episodes);
lr = logspace(-1,-3,num_training_episodes);

% Max env values:
Tmax = 1200;
Tmin = 600;
vMax = 30;
vMin = 20;
Emax = 100000;
Emin = 10000;
xmax = 1000;
ymax = 1000;
hmin = 70;
hmax = 90;
%% initial random episode collector
for e = 1:num_initial_episodes
    fprintf("Initial episodes: %i\n",e)
    terminal = 0;
    T = rand(1) * (Tmax-Tmin) + Tmin;
    vmax = rand(1) * (vMax-vMin) + vMin;
    E = rand(1,num_nodes) * (Emax-Emin) + Emin;
    x0 = rand(1) * xmax;
	y0 = rand(1) * ymax;
    xf = rand(1) * xmax;
    yf = rand(1) * ymax;
    xi = rand(1,num_nodes) * xmax;
    yi = rand(1,num_nodes) * ymax; 
    dxi = xi - x0;
    dyi = yi - y0;
    dxf = xf - x0;
    dyf = yf - y0; 
    h = rand(1) * (hmax-hmin) + hmin;
    lambda = rand(1,num_nodes);
    lambda = lambda/sum(lambda);
    actions = [];
    sum_act = zeros(state_per_node,1,1);
    state_const = zeros(const_size,1,1);
    u = [];
    state_const(1,1,1) = dxf;
    state_const(2,1,1) = dyf;
    state_const(3,1,1) = vmax;
    state_const(4,1,1) = vmax;
    state_const(5,1,1) = T;
    P = ones(1,num_nodes) / num_nodes;
   while terminal == 0
%       action = randi(num_nodes);
      action = randsample((1:num_nodes), 1, true, P );
      P(action) = P(action)/num_nodes;
      P = P/sum(P);
      if ~isempty(u)
        [~,~,~,reward_before] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      else
         reward_before = 1;
      end
      sum_before = sum_act;
      u = [u,action];
      sum_act(1,1,1) = sum_act(1,1,1) + dxi(action);
      sum_act(2,1,1) = sum_act(2,1,1) + dyi(action);
      sum_act(3,1,1) = sum_act(3,1,1) + E(action);
      sum_act(4,1,1) = sum_act(4,1,1) + lambda(action);
      [~,~,~,reward_after] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      if isnan(reward_after) | (reward_after == Inf)
        reward_after = 1;
        terminal = 1;
      end
      sum_before_norm = (sum_before - [-xmax;-ymax;Emin;0])./[2*xmax;2*ymax;Emax-Emin;1];
      state_const_norm = (state_const - [-xmax;-ymax;0;0;Tmin])./[2*xmax;2*ymax;vmax;vmax;Tmax-Tmin];
      sum_act_norm = (sum_act - [-xmax;-ymax;Emin;0])./[2*xmax;2*ymax;Emax-Emin;1];
      traj = {[sum_before_norm;state_const_norm],action,[sum_act_norm;state_const_norm],reward_before - reward_after,{(dxi+xmax)/(2*xmax),(dyi+ymax)/(2*ymax),(E-Emin)/(Emax-Emin),lambda,T}};
      M{length(M) + 1} = traj;
   end
   u
end
%%
pointer = length(M) + 1;
Q = QNetwork(state_per_node,const_size);
options = trainingOptions('sgdm', ...
                        'MaxEpochs',1, ...
                        'Shuffle','every-epoch', ...
                        'MiniBatchSize',1);
Qnet = trainNetwork(rand(13,1,1,1),rand(1),Q,options);   
targetQnet = Qnet;
batch_size = 256;
%%
avg_reward_arr = [];
for e = 1:num_training_episodes
    fprintf("Training episodes: %i\n",e)
    terminal = 0;
    T = rand(1) * (Tmax-Tmin) + Tmin;
    vmax = rand(1) * (vMax-vMin) + vMin;
    E = rand(1,num_nodes) * (Emax-Emin) + Emin;
    x0 = rand(1) * xmax;
	y0 = rand(1) * ymax;
    xf = rand(1) * xmax;
    yf = rand(1) * ymax;
    xi = rand(1,num_nodes) * xmax;
    yi = rand(1,num_nodes) * ymax; 
    dxi = xi - x0;
    dyi = yi - y0;
    dxf = xf - x0;
    dyf = yf - y0; 
    h = rand(1) * (hmax-hmin) + hmin;
    lambda = rand(1,num_nodes);
    lambda = lambda/sum(lambda);
    actions = zeros(length(dxi),4,1,1);
    for i=1:length(dxi)
        actions(i,1,1,1) = dxi(i);
        actions(i,2,1,1) = dyi(i);
        actions(i,3,1,1) = E(i);
        actions(i,4,1,1) = lambda(i);
    end
    sum_act = zeros(state_per_node,1,1);
    state_const = zeros(const_size,1,1);
    u = [];
    state_const(1,1,1) = dxf;
    state_const(2,1,1) = dyf;
    state_const(3,1,1) = vmax;
    state_const(4,1,1) = vmax;
    state_const(5,1,1) = T;
    
    P = ones(1,num_nodes) / num_nodes;
   while terminal == 0
      if rand(1)>EPS(e)
          max = -inf;
            for a = 1:length(dxi)
                q_temp = targetQnet.predict([sum_before;state_const;squeeze(actions(a,:,:,:))']);
                if q_temp > max
                    max = q_temp;
                    action = a;
                end
            end
      else
          action = randsample((1:num_nodes), 1, true, P );
          P(action) = P(action)/num_nodes;
          P = P/sum(P);
%           action = randi(num_nodes);
      end
      if ~isempty(u)
        [~,~,~,reward_before] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      else
         reward_before = 1;
      end
      sum_before = sum_act;
      u = [u,action];
      sum_act(1,1,1) = sum_act(1,1,1) + dxi(action);
      sum_act(2,1,1) = sum_act(2,1,1) + dyi(action);
      sum_act(3,1,1) = sum_act(3,1,1) + E(action);
      sum_act(4,1,1) = sum_act(4,1,1) + lambda(action);
      [~,~,~,reward_after] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      if isnan(reward_after) | (reward_after == Inf)
        reward_after = 1;
        terminal = 1;
      end
      sum_before_norm = (sum_before - [-xmax;-ymax;Emin;0])./[2*xmax;2*ymax;Emax-Emin;1];
      state_const_norm = (state_const - [-xmax;-ymax;0;0;Tmin])./[2*xmax;2*ymax;vmax;vmax;Tmax-Tmin];
      sum_act_norm = (sum_act - [-xmax;-ymax;Emin;0])./[2*xmax;2*ymax;Emax-Emin;1];
      traj = {[sum_before_norm;state_const_norm],action,[sum_act_norm;state_const_norm],reward_before - reward_after,{(dxi+xmax)/(2*xmax),(dyi+ymax)/(2*ymax),(E-Emin)/(Emax-Emin),lambda,T}};
      M{pointer} = traj;
      pointer = mod(pointer,M_cap)+1;
      
   end
   if mod(e,training_period) == 0
      options = trainingOptions('sgdm', ...
                        'MaxEpochs',1, ...
                        'Shuffle','every-epoch', ...
                        'InitialLearnRate', lr(e),...
                        'MiniBatchSize',batch_size);
      [s_prev,act,target] = sampler(Qnet,M,batch_size);
      Qnet = trainNetwork([s_prev;act],target,layerGraph(Qnet),options);
   end
   if mod(e,num_episode_eval) == 0
      avg_reward = eval_policy(targetQnet,100,Tmax,Tmin,vMax,vMin,Emax,Emin,xmax,ymax,hmin,hmax,num_nodes,state_per_node,const_size)
      avg_reward_arr = [avg_reward_arr,avg_reward];
   end
   if mod(e,target_update_episode) == 0
      targetQnet = Qnet;
   end
   u
end