function h = subplotfill(nrows,ncols,index)

dx = 1/ncols;
dy = 1/nrows;

A = 1:nrows*ncols;
A = reshape(A,ncols,nrows)';

[row,col] = find(A == index);

x = (col - 1) * dx;
y = 1 - row * dy;

h = axes('outerposition',[x, y, dx, dy]);

return