function [v_2] = interpolate_grid(x_1_mesh, v_1, x_2_mesh)

f = griddedInterpolant(x_1_mesh{:}, v_1, 'linear');
v_2 = f(x_2_mesh{:});

end
