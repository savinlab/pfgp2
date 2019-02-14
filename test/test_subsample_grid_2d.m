function tests = test_subsample_grid_2d

tests = functiontests(localfunctions);

end


function test_1(testCase)
% Test function on 2x2 grid expanded by factor of 2.

inc = 2;
f_orig = [ ...
    [1, 1, 2, 2]; ...
    [1, 1, 2, 2]; ...
    [3, 3, 4, 4]; ...
    [3, 3, 4, 4]; ...
];
f_subsmp = [ ...
    [1, 2]; ...
    [3, 4] ...
];
result = subsample_grid_2d(f_orig, inc);
verifyEqual(testCase, result, f_subsmp);

end

function test_2(testCase)
% Test that subsampling expanded grid returns original grid.

inc = 2;
f_orig = [ ...
    [1, 2]; ...
    [3, 4] ...
];
result = subsample_grid_2d(expand_grid_2d(f_orig, inc), inc);
verifyEqual(testCase, result, f_orig);

end


