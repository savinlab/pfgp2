function tests = test_rnd_proj_mtx

tests = functiontests(localfunctions);

end


function test_orthogonal(testCase)

mtx = get_rnd_proj_mtx(10);
v1 = mtx(1, :)';
v2 = mtx(2, :)';

verifyEqual(testCase, v1' * v2, 0, 'AbsTol', 1e-10);

end


function test_normal(testCase)

mtx = get_rnd_proj_mtx(10);
v1 = mtx(1, :)';
v2 = mtx(2, :)';

verifyEqual(testCase, norm(v1, 2), 1, 'AbsTol', 1e-10);
verifyEqual(testCase, norm(v2, 2), 1, 'AbsTol', 1e-10);

end
