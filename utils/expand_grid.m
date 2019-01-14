function xx = expand_grid(x, inc)
% Expand the 2D array of function values by a constant factor.
%
% Each element in the original array 'x' is assigned to an 'inc' by 'inc'
% square of values in the new array 'xx'.
%
% CS, could be done better
% assert(inc==2)
% a = zeros(size(aa)*2);
% a(2:2:end,2:2:end) = aa;
% b = a + (convn(a,[1 0 1],'same')+convn(a,[1 0 1]','same'))/2;
% b = b+ convn(a,[1 0 1;0 0 0; 1 0 1], 'same')/4;
% 
% b(1,:) = 2*b(1,:);
% b(:,1) = 2*b(:,1);

N = size(x,1);
z = repmat(reshape(repmat(x(:),1,inc)',N*inc,N), inc,1);
xx = reshape(z(:), N*inc,N*inc);

end
