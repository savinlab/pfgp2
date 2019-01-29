function [mtx] = get_rnd_proj_mtx(n)
% Compute a random projection matrix from R^n to R^2.

v1 = randn([n, 1]);
v2 = randn([n, 1]);

u1 = v1 / norm(v1, 2);
v3 = v2 - (v2' * u1) * u1;
u2 = v3 / norm(v3, 2);

mtx = [u1, u2]';

end
