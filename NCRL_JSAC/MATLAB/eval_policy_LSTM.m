function avg_aoi = eval_policy_LSTM(targetQnet,num_episodes,Tmax,Tmin,vMax,vMin,Emax,Emin,xmax,ymax,hmin,hmax,num_nodes,state_per_node,const_size)
avg_aoi = 0;
for e = 1:num_episodes
    fprintf("Evaluating policy: %i\n",e)
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
    lambda = rand(1,num_nodes);
    lambda = lambda/sum(lambda)
    state_const = zeros(num_nodes + 4,1);
    u = [];
    for i = 1:num_nodes
        state_const(i,1) = lambda(i);
    end
    state_const((num_nodes) + 1,1,1) = (T - Tmin) / (Tmax - Tmin);
    state_const((num_nodes) + 2,1,1) = (vmax - vMin ) / (vMax - vMin);
    state_const((num_nodes) + 3,1,1) = (vmax - vMin ) / (vMax - vMin);
    state_const((num_nodes) + 4,1,1) = (h - hmin ) / (hmax - hmin);
    t_before = [];
    x_before = [];
    y_before = [];
    e_before = (E - Emin)/(Emax-Emin);
    sequential_input_after = [[0,t_before];e_before; [([dxf;dxi] +xmax)/(2*xmax),x_before] ;[([dyf;dyi] +ymax)/(2*ymax),y_before];state_const.*ones(num_nodes+4,1)];
   while terminal == 0

       q_temp = targetQnet.predict({sequential_input_after});
       
       [~,action] = max(q_temp);

     
      if ~isempty(u)
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
        aoi_after = 1;
        terminal = 1;
        sequential_input_after = [[0,t_after];e_after; [([dxf;dxi] +xmax)/(2*xmax),x_after] ;[([dyf;dyi] +ymax)/(2*ymax),y_after];state_const.*ones(num_nodes+4,1)];
      else
        t_after = (t_after)/T;
        e_after = (e_after - Emin) / (Emax-Emin);
        x_after = (x_after - [xf;xi] + xmax) / (2*xmax);
        y_after = (y_after - [yf;yi] + ymax) / (2*ymax);
        sequential_input_after = [[0,t_after];e_after; [([dxf;dxi] +xmax)/(2*xmax),x_after] ;[([dyf;dyi] +ymax)/(2*ymax),y_after];state_const.*ones(num_nodes+4,length(u)+1)];
      end
   end
   u
   aoi_before
   avg_aoi = avg_aoi + aoi_before;
end
avg_aoi = avg_aoi/num_episodes;