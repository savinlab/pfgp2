function x = getRandomWalkRing(eps, T, r0,r1,x0)
% random walk of length T with speed given by eps
% within unit box, reflection at borders
assert(r1>r0)

if nargin <5
    tmp = 2*pi*rand(2,1);
    x0 = [cos(tmp), sin(tmp)]*(r1+r0)/2;
end

xt=x0;
x = zeros(T,2);
x(1,:)  = xt;
for i=2:T
    xtnew = xt+ eps*randn(1,2);
    if(all(sqrt(xtnew*xtnew')*[1 -1]+[-r0,r1]>0))
        xt = xtnew;
    end
    x(i,:) =xt;
end