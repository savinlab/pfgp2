function tests = test_vec_to_arr

tests = functiontests(localfunctions);

end


function test_1d(testCase)
% Test that funciton leaves vector unchanged if dims are current dimensions.

vec = [1, 2, 3, 4]';

result = vec_to_arr(vec, size(vec));
verifyEqual(testCase, result, vec);

end


function test_2d(testCase)
% Test that function works for 2D arrays.

vec = [1, 2, 3, 4]';
arr = [1, 3; 2, 4];

result = vec_to_arr(vec, [2, 2]);
verifyEqual(testCase, result, arr);

end


function test_3d(testCase)
% Test that function works for 3D arrays.

vec = [1, 2, 3, 4, 5, 6, 7, 8]';
arr(:, :, 1) = [1, 3; 2, 4];
arr(:, :, 2) = [5, 7; 6, 8];

result = vec_to_arr(vec, [2, 2, 2]);
verifyEqual(testCase, result, arr);

end


function test_bad_dims(testCase)
% Test that function throws error if dimensions don't match vector

vec = [1, 2, 3, 4]';
dims = [2, 3];

verifyError(testCase, @() vec_to_arr(vec, dims), 'vec_to_arr:incorrect_dims');

end
