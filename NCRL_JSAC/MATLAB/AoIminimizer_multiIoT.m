function [t,x,y,e,cvx_optval] = AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,lambda,u)


cvx_begin quiet;

n = length(u);
ni = length(xi);
% lambda = 1/ni * ones(1,ni);
variable t(1,n);
variable x(1,n);
variable y(1,n);

obj = 0;
for i = 1:ni
    indexes = find(u == i);
    if ~isempty(indexes)
        obj = obj + lambda(i) * (t(indexes(1))^2 +  (T - t(indexes(end)))^2);
        if length(indexes)>1
            for j = 1 : length(indexes) - 1
                obj = obj + lambda(i) * (t(indexes(j+1))-t(indexes(j)))^2; 
            end
        end
    else
        obj = obj + lambda(i) * T^2;
    end
end

obj = obj/T^2;
minimize(obj)

for i = 1:n
    t(i) >= 0;
    t(i) <= T;
end

for i = 1:n-1
    x(i+1) - x(i)<= (t(i+1)- t(i))*vmax;
    x(i+1) - x(i)>= -(t(i+1)- t(i))*vmax;
    y(i+1) - y(i)<= (t(i+1)- t(i))*vmax;
    y(i+1) - y(i)>= -(t(i+1)- t(i))*vmax;
    
%     (x(i+1) - x(i))^2 + (y(i+1) - y(i))^2<= t(i)^2*vmax^2;
end

x(1) - x0 <= t(1)*vmax;
x(1) - x0 >= -t(1)*vmax;
y(1) - y0 <= t(1)*vmax;
y(1) - y0 >= -t(1)*vmax;
% (x(1) - x0).^2 + (y(1) - y0).^2 <= t(1)^2*vmax^2;

xf - x(n) <= (T - t(n))*vmax;
xf - x(n) >= -(T - t(n))*vmax;
yf - y(n) <= (T - t(n))*vmax;
yf - y(n) >= -(T - t(n))*vmax;

% (xf - x(n))^2 + (yf - y(n))^2 <= t(n)^2*vmax^2;

for i=1:ni
    indexes = (u == i);
    if sum(indexes) > 0
        sum((x(indexes) - xi(i)).^2) + sum((y(indexes) - yi(i)).^2 + sum(indexes)*h^2) <= E(i);
    end
end
cvx_end;
e = zeros(ni,n+1);
e(:,1) = E;
for i=1:n
    e(:,i + 1) = e(:,i);
    e(u(i),i+1) = e(u(i),i) - ((x(i) - xi(u(i))).^2 + (y(i) - yi(u(i))).^2 + h^2);
end

