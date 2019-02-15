function [opt] = default_options_2d()
% Default options for 2D pfgp2 functions.

% Limits for both grid dimensions
opt.x_min = 1.0;
opt.x_max = 256.0;

% Number of points per dimension for kernel grid
opt.ng = 256;

% Number of points per dimension for prediction grid
opt.ne = 64;

% Subsampling factor for slow inference prediction grid
opt.inc_slow = 2;

% Whether to use squared exponential (SE) kernel
opt.use_se = false;

% Number of mixture components for spectral mixture (SM) kernel
opt.sm_q = 5;

% Number of random restarts for hyperparameter optimization
opt.n_hyp_restarts = 1;

% Percentage of data to use for cross-validation for hyperparameters
opt.val_pct = 0.1;

end
