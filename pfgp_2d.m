function [results, dbg] = pfgp_2d(y, x, opt, hyp)
% Run 2D Gaussian process regression on neural spike data.
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
%     x (Nx2 array): Position values
%     opt (struct): Optional parameters
%         x_min (float): Minimum position value (default value: 1.0)
%         x_max (float): Maximum position value (default value: 256.0)
%         ng (int): Number of points to use for each dimension of kernel grid 
%             (default value: 256)
%         ne (int): Number of points to use for each dimension of test grid
%             (default value: 64)
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
%         x_test ((ne^2)x2 array): Test points used for inference
%         x_test_mesh (2x1 cell array): Test points used for inference, in
%             meshgrid format
%         fmu (ne x ne array): Posterior mean of latent function
%         fsd2 (ne x ne array): Posterior var of latent function
%         mtuning (ne x ne array): Posterior mean of tuning function
%         vartuning (ne x ne array): Posterior var of tuning function
%     dbg (struct): Debug information

% Set defaults for opt
opt_default = default_options_2d();
opt = set_opt_defaults(opt, opt_default);
dbg.opt = opt;

assert( ...
    mod(opt.ne, opt.inc_slow) == 0, ...
    'opt.ne needs to be divisible by opt.inc_slow' ...
);

if nargin < 4
    hyp = [];
end

% Compute MLE estimate of hyperparameters (if required)
if isempty(hyp)
    fprintf('Computing hyperparameter estimate...\n');
    [hyp, hyp_dbg] = mle_hyp_2d(y, x, opt);
    dbg.hyp = hyp_dbg;
    fprintf('Done. Estimation took %f seconds\n', hyp_dbg.time);
end

% Grid of test points
x_test_vecs = { ...
    linspace(opt.x_min, opt.x_max, opt.ne)', ...
    linspace(opt.x_min, opt.x_max, opt.ne)' ...
};
x_test = apxGrid('expand', x_test_vecs);
x_test_dims = [opt.ne, opt.ne];
x_test_mesh = mtx_to_mesh(x_test, x_test_dims);

% Coarse grid for slow inference
x_slow_vecs = { ...
    x_test_vecs{1}(opt.inc_slow:opt.inc_slow:end), ...
    x_test_vecs{2}(opt.inc_slow:opt.inc_slow:end) ...
};
x_slow = apxGrid('expand', x_slow_vecs);
x_slow_dims = [size(x_slow_vecs{1}, 1), size(x_slow_vecs{2}, 1)];

% GP model parameters
model = get_gp_model_2d(opt);

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

% Use fast mean for latent mean
m_f = vec_to_arr(inf_fast_results.m_f, x_test_dims);

% Expand slow var to size of fast grid to get latent variance
v_f_slow = vec_to_arr(inf_slow_results.v_f, x_slow_dims);
v_f = expand_grid_2d(v_f_slow, opt.inc_slow);

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
