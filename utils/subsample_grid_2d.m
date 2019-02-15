function [y] = subsample_grid_2d(x, inc)
% Subsample 2D grid by constant factor.
%
% Args:
%     x (MxN array): Initial grid
%     inc (int): Factor to subsample by
%
% Returns:
%     y ((M / inc)x(N / inc) array): Subsampled grid

assert( ...
    ndims(x) == 2, ...
    'subsample_grid_2d:incorrect_dims', ...
    sprintf('ndims=%d not supported', ndims(x)) ...
);

y = x(inc:inc:end, inc:inc:end);

end
