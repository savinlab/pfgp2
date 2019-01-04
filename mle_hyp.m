function [results] = mle_hyp(x, y, model)
% Compute maximum likelihood estimate (MLE) of hyperparameters.
%
% Args:
%     x (Nx2 array): Position values
%     y (Nx1 array): Spike counts
%     model (struct): GP model parameters. Contains the following fields:
%         sm_q (int): Number of mixture components (for SM kernel)
%         cov (GPML cov): Covariance object (see GPML docs)
%         mean (GPML mean): Mean object (see GPML docs)
%         lik (GPML lik): Likelihood object (see GPML docs)
%
% Returns:
%     results (struct): Contains the following fields:
%         hyp (struct): Hyperparameter struct (see GPML docs)
%         debug (struct): Contains the following fields:
%             hyp_0 (struct): Initial hyperparameter struct (see GPML docs)
%             inf_opt (struct): Inference options (see GPML docs)
%             time (double): Hyperparameter estimation time

w = ones(model.sm_q, 1) / model.sm_q; 
m1 = (1 / 256) * rand(1, model.sm_q); 
m2 = (1 / 256) * rand(1, model.sm_q); 
v1 = (1 / 256 ^ 2) * rand(1, model.sm_q);
v2 = (1 / 256 ^ 2) * rand(1, model.sm_q);
hyp_0.cov = [log([w; m1(:); v1(:)]); log([w; m2(:); v2(:)])];
hyp_0.mean = [];
hyp_0.lik = [];

inf_opt = struct('cg_maxit', 500, 'cg_tol', 1e-5);
inf = @(varargin) infGrid(varargin{:}, inf_opt);
gp_params = {inf, model.mean, model.cov, model.lik};

tic;
hyp = minimize(hyp_0, @gp, -40, gp_params{:}, x, y);
time = toc;

results.hyp = hyp;
results.debug.time = time;
results.debug.hyp_0 = hyp_0;
results.debug.inf_opt = inf_opt;

end
