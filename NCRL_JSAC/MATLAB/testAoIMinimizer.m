clear all;
T = 180;
vmax = 10;
E = [1000000];
x0 = 5;
y0 = 100;
xf = 1000;
yf = 879;
xi = 0;
yi = 1000;
h = 80;
lambda = [1];
u = [1];
aoi_arr = [];
%%
aoi = 1;
while ~isnan(aoi) & (aoi ~= Inf)
    [~,~,~,~,aoi] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
    aoi_arr = [aoi_arr,aoi];
    u = [u,1];
end
aoi_arr = aoi_arr(1:end-1);
%%
subplotfill(1,2,1);
plot(aoi_arr,'linewidth',2,'color','b','marker','s');
hold on;
plot(6,aoi_arr(6),'linewidth',2,'color','r','markersize',10,'marker','x','linestyle','none');
xlabel('Number of updates','Interpreter','latex','fontsize',13);
ylabel('NWAoI','Interpreter','latex','fontsize',13);
grid on;
legend({'Brute force','NCRL'},'Interpreter','latex','fontsize',13);
removewhitespace;
%%
u = ones(1,6);
[t6,x6,y6,e6,NWAoI6] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
u = ones(1,12);
[t12,x12,y12,e12,NWAoI12] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
%%
subplotfill(1,2,2);
n1 = plot(xi,yi,'color','g','marker','p','linestyle','none','LineWidth',4);
hold on;
t6 = plot([x0,x6,xf],[y0,y6,yf], 'color','k','LineStyle','--');
u6 = plot(x6,y6,'marker','o','color','k','linestyle','none','LineWidth',4);
t12 = plot([x0,x12,xf],[y0,y12,yf], 'color','b','LineStyle','-.');
u12 = plot(x12,y12,'marker','o','color','b','linestyle','none','LineWidth',4);
strt = plot(x0,y0,'marker','s','color','m','linestyle','none','LineWidth',4);
endd = plot(xf,yf,'marker','s','color','y','linestyle','none','LineWidth',4);
grid on;
xlim([min([x0,x6,x12,xf,xi])-50,max([x0,x6,x12,xf,xi])+50]);
ylim([min([y0,y6,y12,yf,yi])-50,max([y0,y6,y12,yf,yi])+50]);
xlabel('x axis (m)','FontSize',13,'Interpreter','latex');
ylabel('y axis (m)','FontSize',13,'Interpreter','latex');
legend([n1,t6,u6,t12,u12,strt,endd],{'Node location','\bf{u}_6 policy trajectory',...
    '${\bf{u}}_6$ policy update locations','${\bf{u}}_{12}$ policy trajectory',...
    '${\bf{u}}_{12}$ policy update locations','Initial location','Final location'},...
    'Interpreter','latex','FontSize',13,'Location','best');
removewhitespace;