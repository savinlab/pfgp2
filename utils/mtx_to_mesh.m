function [x_mesh] = mtx_to_mesh(x_mtx, dims)

for i = 1:size(x_mtx, 2)
    x_mesh{i} = vec_to_arr(x_mtx(:, i), dims);
end

end
