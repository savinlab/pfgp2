function [hyp, dbg] = mle_hyp_3d(y, x, opt, hyp_0)
% Compute maximum likelihood estimate (MLE) of hyperparameters for 3D data.
%
% Args:
%     x (Nx3 array): Position values
%     y (Nx1 array): Spike counts
%     opt (struct): Additional parameters
%         x_min (1x3 float array): Minimum position values
%         x_max (1x3 float array): Maximum position values
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel
%     hyp_0 (struct, optional): Initial hyperparameter values for optimizer.
%         If this parameter is not passed, initial value will be selected 
%         based on kernel type and size of domain.
%
%
% Returns:
%     hyp (struct): Hyperparameter struct (see GPML docs)
%     dbg (struct): Debug information

if ~isfield(opt, 'x_min'), opt.x_min = [1.0, 1.0, 1.0]; end
if ~isfield(opt, 'x_max'), opt.x_max = [256.0, 256.0, 256.0]; end
if ~isfield(opt, 'use_se'), opt.use_se = false; end
if nargin < 4
    hyp_0 = [];
end
dbg.opt = opt;

model = get_gp_model_3d(opt);
inf_opt = struct('cg_maxit', 500, 'cg_tol', 1e-5);
dbg.inf_opt = inf_opt;
inf = @(varargin) infGrid(varargin{:}, inf_opt);
gp_params = {inf, model.mean, model.cov, model.lik};

if isempty(hyp_0)
    hyp_0 = get_hyp_init_3d(opt, model);
end
dbg.hyp_0 = hyp_0;

tic;
hyp = minimize(hyp_0, @gp, -40, gp_params{:}, x, y);
dbg.time = toc;

end
