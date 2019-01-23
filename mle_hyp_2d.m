function [hyp, dbg] = mle_hyp_2d(y, x, opt, hyp_0)
% Compute maximum likelihood estimate (MLE) of hyperparameters for 2D data.
%
% Args:
%     y (Nx1 array): Spike counts
%     x (Nx2 array): Position values
%     opt (struct): Additional parameters
%         x_min (float): Minimum position value (default: 1.0)
%         x_max (float): Maximum position value (default: 256.0)
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel (default: false)
%         sm_q (int): Number of spectral mixture components (if SM kernel is
%             being used).
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
if ~isfield(opt, 'sm_q'), opt.sm_q = 5; end
if nargin < 4
    hyp_0 = [];
end
dbg.opt = opt;

model = get_gp_model_2d(opt);
inf_opt = struct('cg_maxit', 500, 'cg_tol', 1e-5);
dbg.inf_opt = inf_opt;
inf = @(varargin) infGrid(varargin{:}, inf_opt);
gp_params = {inf, model.mean, model.cov, model.lik};

if isempty(hyp_0)
    hyp_0 = get_hyp_init_2d(opt);
end
dbg.hyp_0 = hyp_0;

tic;
hyp = minimize(hyp_0, @gp, -40, gp_params{:}, x, y);
dbg.time = toc;

end
