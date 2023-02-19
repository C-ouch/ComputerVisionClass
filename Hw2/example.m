[H,W] = size(I);
​
c1 = [1,1,1]';
c2 = [W,1,1]';
c3 = [1,H,1]';
c4 = [W,H,1]';
​
cp1 = A*c1;
cp2 = A*c2;
cp3 = A*c3;
cp4 = A*c4;
​
xp1 = cp1(1); yp1 = cp1(2);
xp2 = cp2(1); yp2 = cp2(2);
xp3 = cp3(1); yp3 = cp3(2);
xp4 = cp4(1); yp4 = cp4(2);
​
​
Ap = [min( [1,xp1,xp2,xp3,xp4] ), min( [1,yp1,yp2,yp3,yp4] )];
Bp = [min( [1,xp1,xp2,xp3,xp4] ), max( [yp1,yp2,yp3,yp4] )];
Cp = [max( [xp1,xp2,xp3,xp4] ), min( [1,yp1,yp2,yp3,yp4] )];
Dp = [max( [xp1,xp2,xp3,xp4] ), max( [yp1,yp2,yp3,yp4] )];
​
minx = Ap(1); miny = Ap(2);
maxx = Cp(1); maxy = Dp(2);
​
[Xprime,Yprime] = meshgrid( minx:maxx, miny:maxy );
​
​
pprimematrix = [Xprime(:)';Yprime(:)';ones(1,heightIprime*widthIprime)];
phatmatrix = inv(A) * pprimematrix;
​
xlongvector = phatmatrix(1,:) ./ phatmatrix(3,:);
ylongvector = phatmatrix(2,:) ./ phatmatrix(3,:);
​
xmatrix = reshape( xlongvector', heightIprime, widthIprime );
ymatrix = reshape( ylongvector', heightIprime, widthIprime );
​
Iprime = interp2( I,xmatrix,ymatrix );