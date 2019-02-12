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
%         n_hyp_restarts (int): Number of initializations to use when 
%             optimizing hyperparameters (default: 5)
%         sm_q (int): Number of spectral mixture components (if SM kernel is
%             being used).
%     hyp_0 (struct, optional): Initial hyperparameter values for optimizer.
%         If this parameter is not passed, initial value will be selected 
%         based on kernel type and size of domain.
%
% Returns:
%     hyp (struct): Hyperparameter struct (see GPML docs)
%     dbg (struct): Debug information

% Set defaults for opt
opt_default = default_options_3d();
opt = set_opt_defaults(opt, opt_default);
dbg.opt = opt;

model = get_gp_model_3d(opt);
inf_opt = struct('cg_maxit', 500, 'cg_tol', 1e-8);
dbg.inf_opt = inf_opt;
inf = @(varargin) infGrid(varargin{:}, inf_opt);
gp_params = {inf, model.mean, model.cov, model.lik};

% If hyp_0 is not passed, use multiple restarts to find hyp setting
if nargin < 4

    tic;
    for i = 1:(opt.n_hyp_restarts + 1)
        hyp_0  = get_hyp_init_3d(opt);
        [hyp, fvals] = minimize(hyp_0, @gp, -100, gp_params{:}, x, y);
        hyp_vals(i) = hyp;
        nll_vals(i) = fvals(end);
    end
    dbg.time = toc;

    [~, idx_min] = min(nll_vals);
    hyp = hyp_vals(idx_min);

    dbg.nll = nll_vals(idx_min);
    dbg.nll_vals = nll_vals;
    dbg.hyp_vals = hyp_vals;

% If it is passed, just use hyp_0 for initialization
else

    tic;
    [hyp, fvals] = minimize(hyp_0, @gp, -100, gp_params{:}, x, y);
    dbg.time = toc;
    dbg.nll = fvals(end);

end

end
