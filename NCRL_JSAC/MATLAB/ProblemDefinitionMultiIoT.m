clear all;
% close all;

sigma_2 = 10^(-13);
B = 1;
S = 10;
beta = 0.00002;
e_max = 1;
e_min = 0.1;

E_max = e_max * beta/(2^(S/B)-1)/sigma_2;
E_min = e_min * beta/(2^(S/B)-1)/sigma_2;

h = 80;
x0 = 0 ; y0 = 500; xf = 1000 ; yf = 500;
% xi = [0,0,10,10] ; yi = [0,10,0,10]; 
%xi = [6.5557    7.2292    5.3121]*100; yi = [8.9292    7.0322    2.1203]*100;
xi  = [5.0730,994.9421]; yi = [5.1606,994.5752];
% xi = rand(1,5) * 10 ; yi = rand(1,5) * 10; 
ni = length(xi);
cvx_opt_init = Inf;
perms = permn([0,1,2],6);
% perms = randi(5,1,10);
% perms = [1,2,3,4,5,1,1];
[iters,~] = size(perms);
f = waitbar(0,'Please wait...');
E = [3.6413e+04,3.5510e+04];   
T = 900;
vmax = 25;
lambda = [0.4020,0.5980];
%%
opt = [];
    waitbar(0,f,'Starting the optimization ...');
for i = 1:iters
      i
      waitbar(i/iters,f,sprintf('Progress %0.2f percent ...',i/iters*100));
      u= perms(i,perms(i,:) ~= 0);
      if isempty(u)
      	continue;
      end
      [t_temp,x_temp,y_temp,e,cvx_optval_temp] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
      if (cvx_optval_temp == Inf) || (isnan(cvx_optval_temp))
          continue;
      end
      opt = [opt,cvx_optval_temp] ; 
      if cvx_optval_temp <= cvx_opt_init
          t= t_temp;
          x = x_temp;
          y = y_temp;
          uopt = perms(i,:);
          cvx_optval =  cvx_optval_temp;
          cvx_opt_init = cvx_optval_temp;
      end
end
close(f);
%%
figure;
n1 = plot(xi(1),yi(1),'color','g','marker','p','linestyle','none','LineWidth',4);
hold on;
n2 = plot(xi(2),yi(2),'color','r','marker','p','linestyle','none','LineWidth',4);
n3 = plot(xi(3),yi(3),'color','b','marker','p','linestyle','none','LineWidth',4);
plot([x0,x,xf],[y0,y,yf], 'color','k','LineStyle','--');
u3 = plot(x(1),y(1),'marker','o','color','b','linestyle','none','LineWidth',4);
u2 = plot(x(2),y(2),'marker','o','color','r','linestyle','none','LineWidth',4);
u1 = plot(x(3),y(3),'marker','o','color','g','linestyle','none','LineWidth',4);
strt = plot(x0,y0,'marker','s','color','m','linestyle','none','LineWidth',4);
endd = plot(xf,yf,'marker','s','color','y','linestyle','none','LineWidth',4);
grid on;
xlim([min([x0,x,xf,xi])-50,max([x0,x,xf,xi])+50]);
ylim([min([y0,y,yf,yi])-50,max([y0,y,yf,yi])+50]);
xlabel('x axis (m)','FontSize',13,'Interpreter','latex');
ylabel('y axis (m)','FontSize',13,'Interpreter','latex');
legend([n1,n2,n3,u1,u2,u3,strt,endd],{'Node 1','Node 2','Node 3','Node 1 update location',...
    'Node 2 update location','Node 3 update location','Initial location','Final location'},...
    'Interpreter','latex','FontSize',13,'Location','best');
removewhitespace;
%% 
E = [31552.0861069276, 36155.4900574205, 32098.664907186, 33486.7819590322,48770.4603858191,30933.863584163,47335.1842219002,30985.785444386,30185.1206022113,38111.1267055511];
xi = [994.457705000000,999.122327000000,993.758756000000,997.723795000000,993.793880000000];
xi = (xi - 990) * 10 + 900;
xi = [[0.807817069000000,6.71327126000000,2.59495096000000,9.06450661000000,6.47253822000000]*10,xi];
yi = [995.014064620000,998.434451490000,990.272583790000,996.229902710000,996.213504570000];
yi = (yi - 990) * 10 + 900;
yi = [[4.50290628000000,6.11631693000000,3.46874343000000,9.76792237000000,5.44373638000000]*10,yi];
lambda =[0.0189038134564852, 0.194257712941318, 0.136039714331600, 0.00970733987021267, 0.0431297840066249, 0.147028336844155, 0.0910918664819089, 0.169341934248843, 0.0278833773041252, 0.162616120514727];
u = [6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5];
[t_temp,x_temp,y_temp,e,cvx_optval_temp] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);