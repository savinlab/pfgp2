function [opt] = default_options_3d()
% Default options for 3D pfgp2 functions.

opt.x_min = [1.0, 1.0, 1.0];
opt.x_max = [256.0, 256.0, 256.0];
opt.ng = [32, 32, 10];
opt.ne = [32, 32, 10];
opt.inc_slow = 2;
opt.use_se = false;
opt.sm_q = 5;
opt.n_hyp_restarts = 1;

end
