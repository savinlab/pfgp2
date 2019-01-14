function tests = test_expand_grid

tests = functiontests(localfunctions);

end


function test_1(testCase)
% Test function on basic 2x2 grid expanded by factor of 2.

inc = 2;
f_orig = [ ...

    [1, 2]; ...
    [3, 4] ...
];
f_exp = [ ...
    [1, 1, 2, 2]; ...
    [1, 1, 2, 2]; ...
    [3, 3, 4, 4]; ...
    [3, 3, 4, 4]; ...
];

result = expand_grid(f_orig, inc);
verifyEqual(testCase, result, f_exp);

end
