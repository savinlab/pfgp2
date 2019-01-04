function [results, time] = gp_inf_slow(x, y, x_test, model, hyp)
% Perform standard GP inference
%
% Args:
%     x (Nx2 array): Position values
%     y (Nx1 array): Spike counts
%     x_test (array): Test points used for inference
%     model (struct): GP model parameters. Contains the following fields:
%         sm_q (int): Number of mixture components (for SM kernel)
%         cov (GPML cov): Covariance object (see GPML docs)
%         mean (GPML mean): Mean object (see GPML docs)
%         lik (GPML lik): Likelihood object (see GPML docs)
%     hyp (struct): Hyperparameter struct (see GPML docs)
%
% Returns:
%     results (struct): Contains the following fields:
%         m_y (ne x ne array): Posterior mean of observations
%         v_y (ne x ne array): Posterior var of observations
%         m_f (ne x ne array): Posterior mean of latent function
%         v_f (ne x ne array): Posterior var of latent function
%         m_t (ne x ne array): Posterior mean of tuning function
%         v_t (ne x ne array): Posterior var of tuning function
%     time (double): GP inference time

inf_opt = struct('cg_maxit', 500, 'cg_tol', 1e-5);
inf = @(varargin) infGrid(varargin{:}, inf_opt);
gp_params = {inf, model.mean, model.cov, model.lik};

tic;
[m_y, v_y, m_f, v_f] = gp(hyp, gp_params{:}, x, y, x_test);
time = toc;

% Compute mean and variance of tuning function (from lognormal dist)
m_t = exp(m_f + v_f / 2);
v_t = (exp(v_f) - 1) .* exp(2 * m_f + v_f);

results.m_y = m_y;
results.v_y = v_y;
results.m_f = m_f;
results.v_f = v_f;
results.m_t = m_t;
results.v_t = v_t;
end
