function [y] = subsample_grid_3d(x, inc)
% Subsample 3D grid by constant factor.
%
% Args:
%     x (PxQxR array): Initial grid
%     inc (int): Factor to subsample by
%
% Returns:
%     y ((P/inc)x(Q/inc)x(R/inc) array): Subsampled grid

assert( ...
    ndims(x) == 3, ...
    'subsample_grid_3d:incorrect_dims', ...
    sprintf('ndims=%d not supported', ndims(x)) ...
);

y = x(inc:inc:end, inc:inc:end, inc:inc:end);

end
