function [results, dbg] = pfgp_3d(y, x, opt, hyp)
% Run 3D Gaussian process regression on neural spike data.
%
% This function assumes a Gaussian process (GP) model with a Poisson 
% likelihood and a spectral mixture kernel. There are two ways to invoke the 
% function: 
%     1. `[results] = pfgp_2d(y, x, opt)`: 
%            In this case, where no values for the hyperparameters are passed,
%            the function computes a maximum likelihood estimate of the
%            hyperparameters given the data, and then uses these 
%            hyperparameters to perform GP inference.
%     2. `[results] = pfgp_2d(y, x, opt, hyp)`: 
%            In this case, the function uses the hyperparameters passed in to
%            perform GP inference on the data. 
%
% Args:
%     y (Nx1 array): Spike counts
%     x (Nx3 array): Position values
%     opt (struct): Optional parameters
%         x_min (1x3 float array): Minimum position values 
%             (default value: [1.0, 1.0, 1.0])
%         x_max (1x3 float array): Maximum position values
%             (default value: [256.0, 256.0, 256.0])
%         ng (1x3 int array): Number of points to use for each dimension of
%             kernel grid (default value: [32, 32, 10])
%         ne (1x3 int array): Number of points to use for each dimension of
%             test grid (default value: [32, 32, 10])
%         inc_slow (int): Factor by which number of coarse grid points is 
%             smaller than number of fine grid points (default value: 2)
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel (default value: false)
%         sm_q (int): Number of spectral mixture components (if SM kernel is
%             being used).
%     hyp (struct, optional): Hyperparameters (see GPML docs). If 
%         hyperparameters are not passed in, function will compute maximum 
%         likelihood estimate of hyperparameters.
%    
% Returns:
%     results (struct): Contains the following fields:
%         hyp (struct): Hyperparameter struct (see GPML docs)
%         x_test ((ne(1)*ne(2)*ne(3))x3 array): Test points used for inference
%         x_test_mesh (3x1 cell array): Test points used for inference, in
%             meshgrid format.
%         fmu (ne(1) x ne(2) x ne(3) array): Posterior mean of latent function
%         fsd2 (ne(1) x ne(2) x ne(3) array): Posterior var of latent function
%         mtuning (ne(1) x ne(2) x ne(3) array): Posterior mean of tuning
%             function
%         vartuning (ne(1) x ne(2) x ne(3) array): Posterior var of tuning
%             function
%     dbg (struct): Debug information

% Set defaults for opt
opt_default = default_options_3d();
opt = set_opt_defaults(opt, opt_default);
dbg.opt = opt;

assert( ...
    all(mod(opt.ne, opt.inc_slow) == 0), ...
    'opt.ne needs to be divisible by opt.inc_slow' ...
);

if nargin < 4
    hyp = [];
end

% Get GP model from opt values
model = get_gp_model_3d(opt);

% Compute MLE estimate of hyperparameters (if required)
if isempty(hyp)
    fprintf('Computing hyperparameter estimate...\n');
    [hyp, hyp_dbg] = mle_hyp_3d(y, x, opt);
    dbg.hyp = hyp_dbg;
    fprintf('Done. Estimation took %f seconds\n', hyp_dbg.time);
end

% Grid of test points
x_test_vecs = { ...
    linspace(opt.x_min(1), opt.x_max(1), opt.ne(1))', ...
    linspace(opt.x_min(2), opt.x_max(2), opt.ne(2))', ...
    linspace(opt.x_min(3), opt.x_max(3), opt.ne(3))', ...
};
x_test = apxGrid('expand', x_test_vecs);
x_test_dims = opt.ne;
x_test_mesh = mtx_to_mesh(x_test, x_test_dims);

% Coarse grid for slow inference
x_slow_vecs = { ...
    x_test_vecs{1}(opt.inc_slow:opt.inc_slow:end), ...
    x_test_vecs{2}(opt.inc_slow:opt.inc_slow:end) ...
    x_test_vecs{3}(opt.inc_slow:opt.inc_slow:end) ...
};
x_slow = apxGrid('expand', x_slow_vecs);
x_slow_dims = [ ...
    size(x_slow_vecs{1}, 1), ...
    size(x_slow_vecs{2}, 1), ...
    size(x_slow_vecs{3}, 1) ...
];

fprintf('Computing posterior mean using fast inference...\n');
[inf_fast_results, inf_fast_time] = gp_inf_fast(x, y, x_test, model, hyp);
dbg.inf_fast.results = inf_fast_results;
dbg.inf_fast.time = inf_fast_time;
dbg.inf_fast.x_test = x_test;
fprintf('Done. Fast inference took %f seconds\n', inf_fast_time);

fprintf('Computing posterior var using slow inference...\n');
[inf_slow_results, inf_slow_time] = gp_inf_slow(x, y, x_slow, model, hyp);
dbg.inf_slow.results = inf_slow_results;
dbg.inf_slow.time = inf_slow_time;
dbg.inf_slow.x_test = x_slow;
fprintf('Done. Slow inference took %f seconds\n', inf_slow_time);

% Expand fast inference mean to grid
m_f_fast = vec_to_arr(inf_fast_results.m_f, x_test_dims);

% Check that slow and fast means agree
m_f_slow_coarse = vec_to_arr(inf_slow_results.m_f, x_slow_dims);
m_f_fast_coarse = subsample_grid_3d(m_f_fast, opt.inc_slow);
diff = abs(m_f_fast_coarse - m_f_slow_coarse);
max_diff = max(diff(:));
if max_diff > 1.0
    warning('Posterior means from fast and slow inference do not agree.');
end
dbg.m_f_fast_coarse = m_f_fast_coarse;
dbg.m_f_slow_coarse = m_f_slow_coarse;
dbg.max_fast_slow_diff = max_diff;

% Use fast mean for latent mean
m_f = m_f_fast;

% Interpolate slow var on fast grid to get latent variance
v_f_slow = vec_to_arr(inf_slow_results.v_f, x_slow_dims);
v_f = expand_grid_3d(v_f_slow, opt.inc_slow);

% Compute tuning mean and variance (lognormal distribution)
m_t = exp(m_f + v_f ./ 2);
v_t = (exp(v_f) - 1) .* exp(2 .* m_f + v_f);

results.hyp = hyp;
results.x_test = x_test;
results.x_test_mesh = x_test_mesh;
results.fmu = m_f;
results.fsd2 = v_f;

results.mtuning = m_t;
results.vartuning = v_t;

end
