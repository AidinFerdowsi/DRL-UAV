% function deepRL
clear all;
num_nodes = 4;

state_per_node = 4; % state_per_node = x,y,energy,lambda:4
const_size = 5; %dest_x,dest_y,vmax_x, vmax_y, tau:5

M = {}; %replay memory 
M_cap = 100000;

AE_data = {};

batch_size = 128;

num_initial_episodes = 10000;
num_training_episodes = 1000;
num_episode_eval = 10;
target_update_episode = 5;
training_period = 1;

EPS = logspace(-1,-2,num_training_episodes);
lr = logspace(-1,-2,num_training_episodes);

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
    E = rand(num_nodes,1) * (Emax-Emin) + Emin;
    x0 = rand(1) * xmax;
	y0 = rand(1) * ymax;
    xf = rand(1) * xmax;
    yf = rand(1) * ymax;
    xi = rand(num_nodes,1) * xmax;
    yi = rand(num_nodes,1) * ymax; 
    dxi = xi - x0;
    dyi = yi - y0;
    dxf = xf - x0;
    dyf = yf - y0; 
    h = rand(1) * (hmax-hmin) + hmin;
    lambda = rand(num_nodes,1);
    lambda = lambda/sum(lambda);
    max_penalty = floor(E/h^2).*lambda/min(lambda);
    penalty = zeros(num_nodes,1);
    state_const = zeros(num_nodes + 4,1);
    u = [];
    for i = 1:num_nodes
        state_const(i,1) = lambda(i);
    end
    state_const((num_nodes) + 1,1) = (T - Tmin) / (Tmax - Tmin);
    state_const((num_nodes) + 2,1) = (vmax - vMin ) / (vMax - vMin);
    state_const((num_nodes) + 3,1) = (vmax - vMin ) / (vMax - vMin);
    state_const((num_nodes) + 4,1) = (h - hmin ) / (hmax - hmin);
    P = lambda;
    t_before = [];
    x_before = [];
    y_before = [];
    e_before = (E - Emin)/(Emax-Emin);
   while terminal == 0
%       action = randi(num_nodes);
      action = randsample((1:num_nodes), 1, true, P );
      P(action) = P(action)/num_nodes;
      P = P/sum(P);
      if ~isempty(u)
          t_before = t_after;
          x_before = x_after;
          y_before = y_after;
          e_before = e_after;
          aoi_before = aoi_after;
%         [t_before,x_before,y_before,e_before,reward_before] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      else
         aoi_before = 1;
      end
      u = [u,action];
      [t_after,x_after,y_after,e_after,aoi_after] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      if isnan(aoi_after) | (aoi_after == Inf)
        t_after = [];
        x_after = [];
        y_after = [];
        e_after = (E - Emin)/(Emax-Emin);
        for i=1:num_nodes
            penalty(i) = lambda(i) * sum(u == i);
        end
        aoi_after = sum(max_penalty - penalty);
        terminal = 1;
        sequential_input_after = [[0,t_after];e_after; [([dxf;dxi] +xmax)/(2*xmax),x_after] ;[([dyf;dyi] +ymax)/(2*ymax),y_after];state_const.*ones(num_nodes+4,1)];
        aoi_before
      else
        t_after = (t_after)/T;
        e_after = (e_after - Emin) / (Emax-Emin);
        x_after = (x_after - [xf;xi] + xmax) / (2*xmax);
        y_after = (y_after - [yf;yi] + ymax) / (2*ymax);
        sequential_input_after = [[0,t_after];e_after; [([dxf;dxi] +xmax)/(2*xmax),x_after] ;[([dyf;dyi] +ymax)/(2*ymax),y_after];state_const.*ones(num_nodes+4,length(u)+1)];
      end
      sequential_input_before = [[0,t_before];e_before; [([dxf;dxi] +xmax)/(2*xmax),x_before] ;[([dyf;dyi] +ymax)/(2*ymax),y_before];state_const.*ones(num_nodes+4,length(u))];
      traj = {sequential_input_before,u,sequential_input_after,aoi_before - aoi_after};
      M{length(M) + 1} = traj;
      
      AE_data{length(AE_data) + 1} = sequential_input_before;
   end
   u
end
%%
pointer = length(M) + 1;
Q = QNetwork_LSTM(num_nodes);
options = trainingOptions('sgdm', ...
                        'MaxEpochs',1, ...
                        'Shuffle','every-epoch', ...
                        'MiniBatchSize',1);
Qnet = trainNetwork({rand(num_nodes*4 + 7,2)},rand(1,num_nodes),Q,options);   
targetQnet = Qnet;
%%
avg_reward_arr = [];
for e = 1:num_training_episodes
    fprintf("Training episodes: %i\n",e)
    terminal = 0;
    T = rand(1) * (Tmax-Tmin) + Tmin;
    vmax = rand(1) * (vMax-vMin) + vMin;
    E = rand(num_nodes,1) * (Emax-Emin) + Emin;
    x0 = rand(1) * xmax;
	y0 = rand(1) * ymax;
    xf = rand(1) * xmax;
    yf = rand(1) * ymax;
    xi = rand(num_nodes,1) * xmax;
    yi = rand(num_nodes,1) * ymax; 
    dxi = xi - x0;
    dyi = yi - y0;
    dxf = xf - x0;
    dyf = yf - y0; 
    h = rand(1) * (hmax-hmin) + hmin;
    lambda = rand(num_nodes,1);
    lambda = lambda/sum(lambda)
    max_penalty = floor(E/h^2).*lambda/min(lambda);
    penalty = zeros(num_nodes,1);
    state_const = zeros(num_nodes + 4,1);
    u = [];
    for i = 1:num_nodes
        state_const(i,1) = lambda(i);
    end
    state_const((num_nodes) + 1,1,1) = (T - Tmin) / (Tmax - Tmin);
    state_const((num_nodes) + 2,1,1) = (vmax - vMin ) / (vMax - vMin);
    state_const((num_nodes) + 3,1,1) = (vmax - vMin ) / (vMax - vMin);
    state_const((num_nodes) + 4,1,1) = (h - hmin ) / (hmax - hmin);
    P = lambda;
    t_before = [];
    x_before = [];
    y_before = [];
    e_before = (E - Emin)/(Emax-Emin);
    sequential_input_after = [[0,t_before];e_before; [([dxf;dxi] +xmax)/(2*xmax),x_before] ;[([dyf;dyi] +ymax)/(2*ymax),y_before];state_const.*ones(num_nodes+4,1)];
   while terminal == 0
      
      if rand(1)>EPS(e)
          q_temp = targetQnet.predict({sequential_input_after});
          [~,action] = max(q_temp);
      else
          action = randsample((1:num_nodes), 1, true, P );
          P(action) = P(action)/num_nodes;
          P = P/sum(P);
%           action = randi(num_nodes);
      end
      if ~isempty(u)
          t_before = t_after;
          x_before = x_after;
          y_before = y_after;
          e_before = e_after;
          aoi_before = aoi_after;
%         [t_before,x_before,y_before,e_before,reward_before] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      else
         aoi_before = 1;
      end
      u = [u,action];
      [t_after,x_after,y_after,e_after,aoi_after] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      if isnan(aoi_after) | (aoi_after == Inf)
        t_after = [];
        x_after = [];
        y_after = [];
        e_after = (E - Emin)/(Emax-Emin);
        for i=1:num_nodes
            penalty(i) = lambda(i) * sum(u == i);
        end
        aoi_after = sum(max_penalty - penalty);
        terminal = 1;
        sequential_input_after = [[0,t_after];e_after; [([dxf;dxi] +xmax)/(2*xmax),x_after] ;[([dyf;dyi] +ymax)/(2*ymax),y_after];state_const.*ones(num_nodes+4,1)];
      else
        t_after = (t_after)/T;
        e_after = (e_after - Emin) / (Emax-Emin);
        x_after = (x_after - [xf;xi] + xmax) / (2*xmax);
        y_after = (y_after - [yf;yi] + ymax) / (2*ymax);
        sequential_input_after = [[0,t_after];e_after; [([dxf;dxi] +xmax)/(2*xmax),x_after] ;[([dyf;dyi] +ymax)/(2*ymax),y_after];state_const.*ones(num_nodes+4,length(u)+1)];
      end
      sequential_input_before = [[0,t_before];e_before; [([dxf;dxi] +xmax)/(2*xmax),x_before] ;[([dyf;dyi] +ymax)/(2*ymax),y_before];state_const.*ones(num_nodes+4,length(u))];
      traj = {sequential_input_before,u,sequential_input_after,aoi_before - aoi_after};
      M{pointer} = traj;
      pointer = mod(pointer,M_cap)+1;
      
   end
   u
   aoi_before
   if mod(e,training_period) == 0
      options = trainingOptions('sgdm', ...
                        'MaxEpochs',1, ...
                        'Shuffle','every-epoch', ...
                        'InitialLearnRate', lr(e),...
                        'MiniBatchSize',batch_size);
      [state_action,target] = samplerLSTM(Qnet,M,batch_size,num_nodes);
      Qnet = trainNetwork(state_action,target,layerGraph(Qnet),options);
   end
   if mod(e,num_episode_eval) == 0
      avg_reward = eval_policy_LSTM(targetQnet,10,Tmax,Tmin,vMax,vMin,Emax,Emin,xmax,ymax,hmin,hmax,num_nodes,state_per_node,const_size)
      avg_reward_arr = [avg_reward_arr,avg_reward];
   end
   if mod(e,target_update_episode) == 0
      targetQnet = Qnet;
   end
end