function [v_2] = interpolate_grid(x_1_mesh, v_1, x_2_mesh)
% Interpolate the coarsely computed function over a finer grid.
%
% Args:
%     x_1_mesh (1xD cell array): Mesh grid for coarse grid. Each element of
%         cell array is D-dimensional float array representing part of mesh
%     v_1 (D-dimensional float array): Function values on coarse grid
%     x_2_mesh (1xD cell array): Mesh grid for fine grid to interpolate on.
%
% Returns:
%     v_2 (D-dimensional float array): Interpolated values on fine grid

f = griddedInterpolant(x_1_mesh{:}, v_1, 'linear');
v_2 = f(x_2_mesh{:});

end
