function avg_reward = eval_policy(Qnet,num_episodes,Tmax,Tmin,vMax,vMin,Emax,Emin,xmax,ymax,hmin,hmax,num_nodes,state_per_node,const_size)
avg_reward = 0;
parfor e = 1:num_episodes
    fprintf("Evaluating policy: %i\n",e)
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
    reward = 1;
   while terminal == 0
        max = -inf;
        sum_before = sum_act;
        sum_before_norm = (sum_before - [-xmax;-ymax;Emin;0])./[2*xmax;2*ymax;Emax-Emin;1];
        state_const_norm = (state_const - [-xmax;-ymax;0;0;Tmin])./[2*xmax;2*ymax;vmax;vmax;Tmax-Tmin];
        for a = 1:length(dxi)
            sum_act_norm = (squeeze(actions(a,:,:,:))' - [-xmax;-ymax;Emin;0])./[2*xmax;2*ymax;Emax-Emin;1];
            q_temp = Qnet.predict([sum_before_norm;state_const_norm;sum_act_norm]);
            if q_temp > max
                max = q_temp;
                action = a;
            end
        end
      u = [u,action];
      sum_act(1,1,1) = sum_act(1,1,1) + dxi(action);
      sum_act(2,1,1) = sum_act(2,1,1) + dyi(action);
      sum_act(3,1,1) = sum_act(3,1,1) + E(action);
      sum_act(4,1,1) = sum_act(4,1,1) + lambda(action);
      [~,~,~,reward_after] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      if isnan(reward_after) | (reward_after == Inf)
        terminal = 1;
      else
          reward = reward_after;
      end 
   end
   reward
   avg_reward = avg_reward + reward;
end
avg_reward = avg_reward/num_episodes;