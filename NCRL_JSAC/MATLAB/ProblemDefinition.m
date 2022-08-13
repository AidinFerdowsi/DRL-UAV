clear all;
close all;
h = 1;
vmax = 0.5;
x0 = 0 ; y0 = 5; xf = 10 ; yf = 5;
xi = 5 ; yi = 10; 
cvx_opt_init = Inf;
for E = 500:50:500
for T = 50:10:50
for vmax = 0.1:0.1:0.1
    opt = [];
    for n = 1:20
      [t_temp,x_temp,y_temp,cvx_optval_temp] = AoIminimizer(T,vmax,E,n,x0,y0,xf,yf,xi,yi,h);
      if cvx_optval_temp == Inf
          break;
      end
      if cvx_optval_temp <= cvx_opt_init
          t= t_temp;
          x = x_temp;
          y = y_temp;
          cvx_optval =  cvx_optval_temp;
          cvx_opt_init = cvx_optval_temp;
      end
      opt = [opt,cvx_optval] ; 
    end

    plot(opt);
    hold on;
end
end
end
figure;
plot([x0,x,xf],[y0,y,yf],'marker','o');