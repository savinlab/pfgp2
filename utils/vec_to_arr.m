function [arr] = vec_to_arr(vec, dims)
% Standardized function for converting vector to array, using dimensions.
%
% Args:
%     vec (Nx1 array): Vector representation of multidimensional function
%     dims (1xD array): Dimensions of array
% 
% Returns:
%     arr (array): Array with given dimensions (if dimensions are compatible)

arr = reshape(vec, dims);

end
