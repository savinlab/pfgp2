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
%     hyp (struct, optional): Hyperparameters (see GPML docs). If 
%         hyperparameters are not passed in, function will compute maximum 
%         likelihood estimate of hyperparameters.
%    
% Returns:
%     results (struct): Contains the following fields:
%         hyp (struct): Hyperparameter struct (see GPML docs)
%         x_test ((ne^2)x2 array): Test points used for inference
%         fmu (ne x ne array): Posterior mean of latent function
%         fsd2 (ne x ne array): Posterior var of latent function
%         mtuning (ne x ne array): Posterior mean of tuning function
%         vartuning (ne x ne array): Posterior var of tuning function
%     dbg (struct): Debug information

% Set defaults for opt
if ~isfield(opt, 'x_min'), opt.x_min = 1.0; end
if ~isfield(opt, 'x_max'), opt.x_max = 256.0; end
if ~isfield(opt, 'ng'), opt.ng = 256; end
if ~isfield(opt, 'ne'), opt.ne = 64; end
if ~isfield(opt, 'inc_slow'), opt.inc_slow = 2; end
if ~isfield(opt, 'use_se'), opt.use_se = false; end
dbg.opt = opt;

if nargin < 4
    hyp = [];
end

% Grid for kernel
kg_1 = linspace(opt.x_min, opt.x_max, opt.ng)';
kg_2 = linspace(opt.x_min, opt.x_max, opt.ng)';
kg = {kg_1, kg_2};

% GP model parameters
model.lik = {@likPoisson, 'exp'};
model.mean = {@meanZero};
if opt.use_se
    model.cov = {@apxGrid, {{@covSEiso}, {@covSEiso}}, kg};
else
    model.sm_q = 5;
    model.cov = {@apxGrid, {{@covSM, model.sm_q}, {@covSM, model.sm_q}}, kg};
end

% Compute MLE estimate of hyperparameters (if required)
if isempty(hyp)
    fprintf('Computing hyperparameter estimate...\n');
    [hyp, hyp_dbg] = mle_hyp(x, y, model, opt);
    dbg.hyp = hyp_dbg;
    fprintf('Done. Estimation took %f seconds\n', hyp_dbg.time);
end

% Grid of test points
x_test_1 = linspace(opt.x_min, opt.x_max, opt.ne)';
x_test_2 = linspace(opt.x_min, opt.x_max, opt.ne)';
x_test = apxGrid('expand', {x_test_1, x_test_2});

% Coarse grid for slow inference
x_slow_1 = x_test_1(opt.inc_slow:opt.inc_slow:end, :);
x_slow_2 = x_test_2(opt.inc_slow:opt.inc_slow:end, :);
x_slow = apxGrid('expand', {x_slow_1, x_slow_2});

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
m_f = reshape(inf_fast_results.m_f, opt.ne, opt.ne);

% Interpolate slow var on fast grid to get latent variance
v_f_coarse = reshape( ...
    inf_slow_results.v_f, ...
    opt.ne / opt.inc_slow, ...
    opt.ne / opt.inc_slow ...
);
v_f = interpolate_grid(v_f_coarse, opt.inc_slow);

% Compute tuning mean and variance (lognormal distribution)
m_t = exp(m_f + v_f / 2);
v_t = (exp(v_f) - 1) .* exp(2 * m_f + v_f);

results.hyp = hyp;
results.x_test = x_test;
results.fmu = m_f;
results.fsd2 = v_f;
results.mtuning = m_t;
results.vartuning = v_t;

end
