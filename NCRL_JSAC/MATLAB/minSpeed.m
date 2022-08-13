function [v,opt_AoI] = minSpeed(T,E,x0,y0,xf,yf,xi,yi,h,lambda)
[v,u] = upper_bound_minSpeed(T,E,x0,y0,xf,yf,xi,yi,h);
vprev = v;
vnext = v/2
EPS = 1;
opt_AoI = 1;
while vprev - vnext > EPS
    [~,~,~,~,cvx_optval] = AoIminimizer_multiIoT(T,vnext,E,x0,y0,xf,yf,xi,yi,h,lambda,u);
    if (cvx_optval == Inf) || isnan(cvx_optval)
        vnext = (vprev+vnext)/2;
    else
        vprev = vnext;
        vnext = vprev/2
        opt_AoI = cvx_optval;
    end
end
v = vnext;