%%% Ground-truth hyperparameter recovery with correct init

clear all, close all, clc
rng(17434);

% Global parameters
data_dims = [256, 256];
opt.x_min = 1.0;
opt.x_max = 256.0;
opt.use_se = false;
opt.sm_q = 3;
opt.ng = 256;
opt.ne = 64;

% Sample true hyperparameters
hyp_true = get_hyp_init_2d(opt);

% Test hyperparameter estimation with varying number of data points
n_pts_vals = [5000, 7500, 10000, 12500, 15000, 17000, 20000];
for i = 1:size(n_pts_vals, 2)

    n_pts = n_pts_vals(i);

    fr_true = sample_tuning_fn(hyp_true, opt, data_dims, 2);
    x = sample_position_vals(n_pts);
    y = sample_spike_counts(fr_true, x, data_dims);

    % Run hyperparameter estimation with true values as init
    [hyp_est, dbg] = mle_hyp_2d(y, x, opt, hyp_true);

    % Compute error
    hyp_dist = norm(hyp_true.cov - hyp_est.cov, 2);
    hyp_true_norm = norm(hyp_true.cov, 2);
    hyp_est_norm = norm(hyp_est.cov, 2);
    hyp_err_vals(i) = hyp_dist / (hyp_true_norm * hyp_est_norm);

end

% Plot results
plot(n_pts_vals, hyp_err_vals, '-.*');
xlabel('num. data points');
ylabel('normalized error');


function [t] = sample_tuning_fn(hyp, opt, dims, inc)

s_dims = dims / inc;
x_1 = linspace(opt.x_min, opt.x_max, s_dims(1))';
x_2 = linspace(opt.x_min, opt.x_max, s_dims(2))';

[f_s, ~] = sample_hyp_sm_2d(opt.sm_q, hyp, {x_1, x_2}, 1);
f_s_grid = reshape(f_s, s_dims);
f = expand_grid_2d(f_s_grid, inc);

t = exp(f);

end


function [x] = sample_position_vals(n_pts)
% Use random walk to sample realistic trajectory in 2D space

% Generate samples from random walk
rw_smps = get_rnd_walk_ring(0.05, n_pts, 0.25, 0.99, [0, 0.75]);

% Rescale x to unit square
x = ceil(256 / 2 * (rw_smps + 1));
end


function [y] = sample_spike_counts(fr_true, x, dims)
% Sample spikes from Poisson distribution

y = poissrnd(fr_true(sub2ind(dims, x(:,1), x(:,2))));
end


