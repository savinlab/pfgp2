function [hyp, dbg] = mle_hyp_3d(y, x, model, opt)
% Compute maximum likelihood estimate (MLE) of hyperparameters for 3D data.
%
% Args:
%     x (Nx3 array): Position values
%     y (Nx1 array): Spike counts
%     model (struct): GP model parameters. Contains the following fields:
%         sm_q (int): Number of mixture components (for SM kernel)
%         cov (GPML cov): Covariance object (see GPML docs)
%         mean (GPML mean): Mean object (see GPML docs)
%         lik (GPML lik): Likelihood object (see GPML docs)
%     opt (struct): Additional parameters
%         x_min (1x3 float array): Minimum position values
%         x_max (1x3 float array): Maximum position values
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel
%
% Returns:
%     hyp (struct): Hyperparameter struct (see GPML docs)
%     dbg (struct): Debug information

if ~isfield(opt, 'x_min'), opt.x_min = [1.0, 1.0, 1.0]; end
if ~isfield(opt, 'x_max'), opt.x_max = [256.0, 256.0, 256.0]; end
if ~isfield(opt, 'use_se'), opt.use_se = false; end
if nargin < 5
    hyp_0 = get_hyp_init_3d(opt, model);
end
dbg.opt = opt;
dbg.hyp_0 = hyp_0;

inf_opt = struct('cg_maxit', 500, 'cg_tol', 1e-5);
dbg.inf_opt = inf_opt;
inf = @(varargin) infGrid(varargin{:}, inf_opt);
gp_params = {inf, model.mean, model.cov, model.lik};

tic;
hyp = minimize(hyp_0, @gp, -40, gp_params{:}, x, y);
dbg.time = toc;

end
