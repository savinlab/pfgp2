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

    n_est = opt.n_hyp_restarts + 1;

    % If val_pct is zero, don't do cross-validation
    if opt.val_pct == 0.0

        [hyps, nlls, est_time] = get_hyps(n_est, gp_params, x, y, opt);

        [~, idx_min] = min(nlls);
        hyp = hyps(idx_min);
        dbg.nll = nlls(idx_min);

        dbg.nll_vals = nlls;
        dbg.hyp_vals = hyps;
        dbg.time = est_time;

    % Otherwise, split data and use validation set for model selection
    else

        % Split data into training and validation sets
        [train_data, val_data] = cv_split(x, y, opt.val_pct);

        % Estimate hyperparameters using training data
        [hyps, nlls_train, est_time] = get_hyps( ...
            n_est, gp_params, train_data.x, train_data.y, opt);

        % Compute NLL values on validation data
        for i = 1:length(hyps)
            [nlls_val(i), ~] = gp(hyps(i), gp_params{:}, val_data.x, val_data.y);
        end

        % Choose hyperparameter value with smallest validation NLL
        [~, idx_min] = min(nlls_val);
        hyp = hyps(idx_min);
        dbg.nll_val = nlls_val(idx_min);
        dbg.nll_train = nlls_train(idx_min);

        dbg.nll_vals_val = nlls_val;
        dbg.nll_vals_train = nlls_train;
        dbg.hyp_vals = hyps;
        dbg.time = est_time;
    end

% If it is passed, just use hyp_0 for initialization
else

    tic;
    [hyp, fvals] = minimize(hyp_0, @gp, -100, gp_params{:}, x, y);
    dbg.time = toc;
    dbg.nll = fvals(end);

end

end


function [hyps, nlls, time] = get_hyps(n_est, gp_params, x, y, opt)
% Compute hyperparameter MLE estimates with random restarts

for i = 1:n_est

    hyp_0 = get_hyp_init_3d(opt);

    tic;
    [hyp, fvals] = minimize(hyp_0, @gp, -100, gp_params{:}, x, y);
    times(i) = toc;
    nlls(i) = fvals(end);
    hyps(i) = hyp;
end

time = sum(times);

end
