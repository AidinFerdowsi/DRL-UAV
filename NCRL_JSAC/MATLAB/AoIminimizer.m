function [t,x,y,cvx_optval] = AoIminimizer(T,vmax,E,n,x0,y0,xf,yf,xi,yi,h)

cvx_begin

variable t(1,n);
variable x(1,n);
variable y(1,n);

minimize(sum(t.^2) +  (T - sum(t))^2)

for i = 1:n
    t(i) >= 0;
    t(i) <= T;
end

for i = 2:n-1
    x(i+1) - x(i)<= t(i)*vmax;
    x(i+1) - x(i)>= -t(i)*vmax;
    y(i+1) - y(i)<= t(i)*vmax;
    y(i+1) - y(i)>= -t(i)*vmax;
    
%     (x(i+1) - x(i))^2 + (y(i+1) - y(i))^2<= t(i)^2*vmax^2;
end

x(1) - x0 <= t(1)*vmax;
x(1) - x0 >= -t(1)*vmax;
y(1) - y0 <= t(1)*vmax;
y(1) - y0 >= -t(1)*vmax;
% (x(1) - x0).^2 + (y(1) - y0).^2 <= t(1)^2*vmax^2;

xf - x(n) <= t(1)*vmax;
xf - x(n) >= -t(1)*vmax;
yf - y(n) <= t(1)*vmax;
yf - y(n) >= -t(1)*vmax;

% (xf - x(n))^2 + (yf - y(n))^2 <= t(n)^2*vmax^2;
sum((x - xi).^2) + sum((y - yi).^2 + n * h^2) <= E;
cvx_end

