function [x_mesh] = mtx_to_mesh(x_mtx, dims)
% Create mesh grid from matrix representing Cartesian product of grid vectors.
%
% We use this function in order to have a consistent way to convert between
% 'matrix' and 'mesh' representations, so we can be sure that we're not 
% accidentally scrambling values.
%
% Args:
%     x_mtx (NxD float array): Cartesian product of D grid vectors. Each row
%         in matrix contains the coordinate of a point on a complete grid.
%     dims (1xD float array): Dimensions of grid.
%
% Returns:
%     x_mesh (1xD cell array): Mesh representation of grid. Each element of
%         cell array is D-dimensional float array representing single
%         coordinate in mesh format.

for i = 1:size(x_mtx, 2)
    x_mesh{i} = vec_to_arr(x_mtx(:, i), dims);
end

end
