function [results, time] = gp_inf_fast(x, y, x_test, model, hyp)
% Perform fast GP inference (error bars will be way off)
%
% Args:
%     x (NxD array): Position values
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
%         m_y ((grid size)xD array): Posterior mean of observations
%         v_y ((grid size)xD array): Posterior var of observations
%         m_f ((grid size)xD array): Posterior mean of latent function
%         v_f ((grid size)xD array): Posterior var of latent function
%         m_t ((grid size)xD array): Posterior mean of tuning function
%         v_t ((grid size)xD array): Posterior var of tuning function
%         nll (double): Negative log likelihood value         
%     time (double): GP inference time

inf_opt = struct('cg_maxit', 500, 'cg_tol', 1e-5, 'stat', true);
tic;
[post, nll] = infGrid(hyp, model.mean, model.cov, model.lik, x, y, inf_opt);
time = toc;
[m_f, v_f, m_y, v_y] = post.predict(x_test);

% Compute mean and variance of tuning function (from lognormal dist)
m_t = exp(m_f + v_f / 2);
v_t = (exp(v_f) - 1) .* exp(2 * m_f + v_f);

results.m_y = m_y;
results.v_y = v_y;
results.m_f = m_f;
results.v_f = v_f;
results.m_t = m_t;
results.v_t = v_t;
results.nll = nll;
end
