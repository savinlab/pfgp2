function [results] = pfgp_2d(x, y, ng, ne, inc_slow)
% Run 2D GP regression with SM kernels and Poisson likelihood
%
% Args:
%     x (Nx2 array): Position values
%     y (Nx1 array): Spike counts
%     ng (int): Number of points to use for each dimension of kernel grid
%     ne (int): Number of points to use for each dimension of test grid
%     inc_slow (int): Factor by which coarse grid is smaller than full grid
%
% Returns:
%     results (struct): Contains the following fields:
%         hyp (struct): Hyperparameter struct (see GPML docs)
%         hyp_time (double): Hyperparameter estimation time
%         x_test ((ne^2)x2 array): Test points used for inference
%         m_y (ne x ne array): Posterior mean of observations
%         v_y (ne x ne array): Posterior var of observations
%         m_f (ne x ne array): Posterior mean of latent function
%         v_f (ne x ne array): Posterior var of latent function
%         m_t (ne x ne array): Posterior mean of tuning function
%         v_t (ne x ne array): Posterior var of tuning function
%         nll (double): Negative log likelihood value         
%         inf_time (double): GP inference time


% Default value for inc_slow is 2 (don't change)
if nargin < 5, inc_slow = 2; end 

X_MIN = 1.0;
X_MAX = 256.0;

% Grid for kernel
kg_1 = linspace(X_MIN, X_MAX, ng)';
kg_2 = linspace(X_MIN, X_MAX, ng)';
kg = {kg_1, kg_2};

% GP model parameters
model.sm_q = 5;
model.cov = {@apxGrid, {{@covSM, model.sm_q}, {@covSM, model.sm_q}}, kg};
model.mean = {@meanZero};
model.lik = {@likPoisson, 'exp'};

fprintf('Computing hyperparameter estimate...\n');
results_hyp = mle_hyp(x, y, model);
hyp = results_hyp.hyp;
fprintf('Done. Estimation took %f seconds\n', results_hyp.debug.time);

% Grid of test points
x_test_1 = linspace(X_MIN, X_MAX, ne)';
x_test_2 = linspace(X_MIN, X_MAX, ne)';
x_test = apxGrid('expand', {x_test_1, x_test_2});

% Coarse grid for slow inference
x_slow_1 = x_test_1(inc_slow:inc_slow:end, :);
x_slow_2 = x_test_2(inc_slow:inc_slow:end, :);
x_slow = apxGrid('expand', {x_slow_1, x_slow_2});

fprintf('Computing posterior mean using fast inference...\n');
[inf_fast_results, inf_fast_time] = gp_inf_fast(x, y, x_test, model, hyp);
fprintf('Done. Fast inference took %f seconds\n', inf_fast_time);

fprintf('Computing posterior var using slow inference...\n');
[inf_slow_results, inf_slow_time] = gp_inf_slow(x, y, x_slow, model, hyp);
fprintf('Done. Slow inference took %f seconds\n', inf_slow_time);

% Use fast mean for latent mean
m_f = reshape(inf_fast_results.m_f, ne, ne);

% Interpolate slow var on fast grid to get latent variance
v_f_coarse = reshape(inf_slow_results.v_f, ne/inc_slow, ne/inc_slow);
v_f = interpolateGrid(v_f_coarse, inc_slow);

% Compute tuning mean and variance (lognormal dist)
m_t = exp(m_f + v_f / 2);
v_t = (exp(v_f) - 1) .* exp(2 * m_f + v_f);

% Add debug information to results struct
results.debug.hyp.time = results_hyp.debug.time;
results.debug.inf_fast.results = inf_fast_results;
results.debug.inf_fast.time = inf_fast_time;
results.debug.inf_fast.x_test = x_test;
results.debug.inf_slow.results = inf_slow_results;
results.debug.inf_slow.time = inf_slow_time;
results.debug.inf_slow.x_test = x_slow;

% Add return values to results struct
results.hyp = hyp;
results.x_test = x_test;
results.m_f = m_f;
results.v_f = v_f;
results.m_t = m_t;
results.v_t = v_t;

end
