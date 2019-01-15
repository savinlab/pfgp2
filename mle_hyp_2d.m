function [hyp, dbg] = mle_hyp(y, x, model, opt, hyp_0)
% Compute maximum likelihood estimate (MLE) of hyperparameters for 2D data.
%
% Args:
%     y (Nx1 array): Spike counts
%     x (Nx2 array): Position values
%     model (struct): GP model parameters. Contains the following fields:
%         sm_q (int): Number of mixture components (for SM kernel)
%         cov (GPML cov): Covariance object (see GPML docs)
%         mean (GPML mean): Mean object (see GPML docs)
%         lik (GPML lik): Likelihood object (see GPML docs)
%     opt (struct): Additional parameters
%         x_min (float): Minimum position value (default: 1.0)
%         x_max (float): Maximum position value (default: 256.0)
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel (default: false)
%     hyp_0 (struct, optional): Initial hyperparameter values for optimizer.
%         If this parameter is not passed, initial value will be selected 
%         based on kernel type and size of domain.
%
% Returns:
%     hyp (struct): Hyperparameter struct (see GPML docs)
%     dbg (struct): Debug information

if ~isfield(opt, 'x_min'), opt.x_min = 1.0; end
if ~isfield(opt, 'x_max'), opt.x_max = 256.0; end
if ~isfield(opt, 'use_se'), opt.use_se = false; end
if nargin < 5
    hyp_0 = get_hyp_init_2d(opt, model);
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
