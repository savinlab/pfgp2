function [hyp, dbg] = mle_hyp(x, y, model, opt)
% Compute maximum likelihood estimate (MLE) of hyperparameters for 2D data.
%
% Args:
%     x (Nx2 array): Position values
%     y (Nx1 array): Spike counts
%     model (struct): GP model parameters. Contains the following fields:
%         sm_q (int): Number of mixture components (for SM kernel)
%         cov (GPML cov): Covariance object (see GPML docs)
%         mean (GPML mean): Mean object (see GPML docs)
%         lik (GPML lik): Likelihood object (see GPML docs)
%     opt (struct): Additional parameters
%         x_min (float): Minimum position value
%         x_max (float): Maximum position value
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel
%
% Returns:
%     hyp (struct): Hyperparameter struct (see GPML docs)
%     dbg (struct): Debug information

if opt.use_se
    hyp_0 = get_hyp_init_se(opt.x_min, opt.x_max);
else
    hyp_0 = get_hyp_init_sm(model.sm_q, opt.x_min, opt.x_max);
end
dbg.hyp_0 = hyp_0;

inf_opt = struct('cg_maxit', 500, 'cg_tol', 1e-5);
dbg.inf_opt = inf_opt;
inf = @(varargin) infGrid(varargin{:}, inf_opt);
gp_params = {inf, model.mean, model.cov, model.lik};

tic;
hyp = minimize(hyp_0, @gp, -40, gp_params{:}, x, y);
dbg.time = toc;

end


function [hyp_0] = get_hyp_init_sm(sm_q, x_min, x_max)
% Compute initial hyperparameter values for SM kernel
%
% Args:
%     sm_q (int): Number of mixture components for SM kernel
%     x_min (float): Minimum position value
%     x_max (float): Maximum position value
%
% Returns:
%     hyp_0 (struct): Initial hyperparameter struct (see GPML docs)

scl = 1 / (x_max - x_min);
w = ones(sm_q, 1) / sm_q; 
m1 = scl * rand(1, sm_q); 
m2 = scl  * rand(1, sm_q); 
v1 = (scl ^ 2) * rand(1, sm_q);
v2 = (scl ^ 2) * rand(1, sm_q);
hyp_0.cov = [log([w; m1(:); v1(:)]); log([w; m2(:); v2(:)])];
hyp_0.mean = [];
hyp_0.lik = [];

end


function [hyp_0] = get_hyp_init_se(x_min, x_max)
% Compute initial hyperparameter values for SE kernel
%
% Args:
%     x_min (float): Minimum position value
%     x_max (float): Maximum position value
%
% Returns:
%     hyp_0 (struct): Initial hyperparameter struct (see GPML docs)

scl = 1 / (x_max - x_min);
hyp_0.cov = [log([scl; 1]); log([scl; 1])];
hyp_0.mean = [];
hyp_0.lik = [];

end
