function [v,u] = upper_bound_minSpeed(T,E,x0,y0,xf,yf,xi,yi,h)


nmax = floor(E./h^2)-2;
update_t = T./(nmax+1);
u = zeros(1,sum(nmax));
t = zeros(1,sum(nmax));

for i=1:sum(nmax)
    [min_t,index_t] = min(update_t);
    t(i) = min_t;
    u(i) = index_t;
    update_t(index_t) = update_t(index_t)+ T./(nmax(index_t)+1);
end
ni = length(xi);
n = length(u);


cvx_begin;





% lambda = 1/ni * ones(1,ni);
variable vmax(1);
variable x(1,n);
variable y(1,n);
% variable l(1,ni);


obj = 0;


for i=1:ni
    indexes = (u == i);
    if sum(indexes) > 0
        obj = obj + (sum((x(indexes) - xi(i)).^2) + sum((y(indexes) - yi(i)).^2 + sum(indexes)*h^2) -  E(i));
    end
end

for i = 1:n-1
    (x(i+1) - x(i))<= (t(i+1)- t(i))*vmax;
    x(i+1) - x(i)>= -(t(i+1)- t(i))*vmax;
    (y(i+1) - y(i))<= (t(i+1)- t(i))*vmax;
    y(i+1) - y(i)>= -(t(i+1)- t(i))*vmax;
    
%     (x(i+1) - x(i))^2 + (y(i+1) - y(i))^2<= t(i)^2*vmax^2;
end

(x(1) - x0) <= t(1)*vmax;
x(1) - x0 >= -t(1)*vmax;
(y(1) - y0) <= t(1)*vmax;
y(1) - y0 >= -t(1)*vmax;
% (x(1) - x0).^2 + (y(1) - y0).^2 <= t(1)^2*vmax^2;

(xf - x(n)) <= (T - t(n))*vmax;
xf - x(n) >= -(T - t(n))*vmax;
(yf - y(n)) <= (T - t(n))*vmax;
yf - y(n) >= -(T - t(n))*vmax;

% (xf - x(n))^2 + (yf - y(n))^2 <= t(n)^2*vmax^2;

% for i=1:ni
%     indexes = (u == i);
%     if sum(indexes) > 0
%         sum((x(indexes) - xi(i)).^2) + sum((y(indexes) - yi(i)).^2 + sum(indexes)*h^2) <= E(i);
%     end
% end


minimize(obj );


cvx_end;
e = zeros(ni,n+1);
e(:,1) = E;
for i=1:n
    e(:,i + 1) = e(:,i); 
    e(u(i),i+1) = e(u(i),i) - ((x(i) - xi(u(i))).^2 + (y(i) - yi(u(i))).^2 + h^2);
end
v = vmax;

