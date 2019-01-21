function [y] = expand_grid_3d(x, inc)
% Expand 3D grid by constant factor.
%
% The 'expanded' grid y is defined such that:
%     y(i, j, k) = x(floor(i / inc), floor(j / inc), floor(k / inc))
%
% Args:
%     x (QxRxS array): Initial grid
%     inc (int): Factor to expand by
%
% Returns:
%     y ((Q*inc)x(R*inc)x(S*inc) array): Expanded grid
    
assert( ...
    ndims(x) == 3, ...
    'expand_grid_3d:incorrect_dims', ...
    sprintf('ndims=%d not supported', ndims(x)) ...
);
y = ex_3(ex_2(ex_1(x, inc), inc), inc);

end


function [y] = ex_2(x, inc)
% Expand 2nd dimension

[s1, s2, s3] = size(x);
x_rep = repmat(x, inc, 1);
y = reshape(x_rep, s1, inc * s2, s3); 

end


function [y] = ex_1(x, inc)
% Expand 1st dimension

swap = @(x) permute(x, [2, 1, 3]);
y = swap(ex_2(swap(x), inc));

end

function [y] = ex_3(x, inc)
% Expand 3rd dimension

swap = @(x) permute(x, [1, 3, 2]);
y = swap(ex_2(swap(x), inc));

end
