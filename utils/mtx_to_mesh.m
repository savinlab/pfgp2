function [mesh] = mtx_to_mesh(mtx, dims)
% Create mesh grid from matrix representing Cartesian product of grid vectors.
%
% We use this function in order to have a consistent way to convert between
% 'matrix' and 'mesh' representations, so we can be sure that we're not 
% accidentally scrambling values.
%
% Args:
%     mtx (NxD float array): Cartesian product of D grid vectors. Each row
%         in matrix contains the coordinate of a point on a complete grid.
%     dims (1xD float array): Dimensions of grid.
%
% Returns:
%     mesh (1xD cell array): Mesh representation of grid. Each element of
%         cell array is D-dimensional float array representing single
%         coordinate in mesh format.
% 
% Throws:
%     error ('mtx_to_mesh:incorrect_dims'): If dimensions are not compatible
%         with matrix 

assert( ...
    size(mtx, 1) == prod(dims), ...
    'mtx_to_mesh:incorrect_dims', ...
    'Dimensions not compatible with matrix'...
);

for i = 1:size(mtx, 2)
    mesh{i} = vec_to_arr(mtx(:, i), dims);
end

end
