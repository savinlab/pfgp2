function [y] = expand_grid_2d(x, inc)
% Expand 2D grid by constant factor.
%
% The 'expanded' grid y is defined such that:
%     y(i, j) = x(floor(i / inc), floor(j / inc))
%
% Args:
%     x (MxN array): Initial grid
%     inc (int): Factor to expand by
%
% Returns:
%     y ((M*inc)x(N*inc) array): Expanded grid

assert( ...
    ndims(x) == 2, ...
    'expand_grid_2d:incorrect_dims', ...
    sprintf('ndims=%d not supported', ndims(x)) ...
);

y = ex_2(ex_1(x, inc), inc);

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
